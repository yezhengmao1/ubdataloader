import os
import time
import json
import queue
import torch
import duckdb
import threading
from dataclasses import dataclass
from transformers import AutoTokenizer
from typing import Optional, List, Dict, Any
from streaming import StreamingDataset, Stream


@dataclass
class TextTokenChunk:
    chunk_state: Dict[str, Any]
    text_chunk: List[str]
    token_chunk: List[int]


class TextChunkDataset:
    def __init__(
        self,
        local_path: List[str],
        remote_path: List[Optional[str]],
        proportion: List[float],
        chunk_size: int = 4096,
        random_seed: int = 42,
    ):
        """
        "local_path": the local path of the dataset
        "remote_path": the remote path of the dataset
        "chunk_size": the chunk size of the dataset
        """
        self.local_path = local_path
        self.remote_path = remote_path
        self.proportion = proportion

        # resume from the checkpoint
        self.chunk_size = chunk_size
        self.epoch = 0
        self.sample_in_epoch = 0
        self.random_seed = random_seed

        self.stream_dataset = None

        self._create_stream_dataset()

    def _create_stream_dataset(self):
        if self.stream_dataset is not None:
            del self.stream_dataset

        streams = []
        for local_path, remote_path, proportion in zip(
            self.local_path, self.remote_path, self.proportion
        ):
            streams.append(
                Stream(
                    remote=remote_path,
                    local=local_path,
                    proportion=proportion,
                )
            )
        self.stream_dataset = StreamingDataset(
            streams=streams,
            shuffle=False,
            shuffle_seed=self.random_seed + self.epoch * 10,
            num_canonical_nodes=1,
            batch_size=self.chunk_size,
            replication=int(os.environ["WORLD_SIZE"]),
        )

    def load_state_dict(self, chunk_state: Dict[str, Any]):
        self.epoch = chunk_state["epoch"]
        self.sample_in_epoch = chunk_state["sample_in_epoch"]

        assert self.chunk_size == chunk_state["chunk_size"]
        assert self.random_seed == chunk_state["random_seed"]
        assert self.remote_path == chunk_state["remote_path"]
        assert self.proportion == chunk_state["proportion"]

        self._create_stream_dataset()
        self.stream_dataset.load_state_dict(chunk_state["streaming_dict"])

    def record_state_dict(self):
        return {
            "streaming_dict": self.stream_dataset.state_dict(
                self.sample_in_epoch, from_beginning=True
            ),
            "epoch": self.epoch,
            "sample_in_epoch": self.sample_in_epoch,
            "chunk_size": self.chunk_size,
            "random_seed": self.random_seed,
            "local_path": self.local_path,
            "remote_path": self.remote_path,
            "proportion": self.proportion,
        }

    def __iter__(self):
        # the stream dataset will be re-created in the next epoch
        # so it's infinite loop
        chunk = []
        state = None
        while True:
            for data in self.stream_dataset:
                if state is None:
                    state = self.record_state_dict()
                chunk.append(data["text"])
                self.sample_in_epoch += 1
                if len(chunk) == self.chunk_size:
                    yield chunk, state
                    chunk = []
                    state = None

            # the stream dataset will be re-created in the next epoch
            self.sample_in_epoch = 0
            self.epoch += 1
            self._create_stream_dataset()


class TextTokenChunkCache:
    def __init__(
        self,
        local_path: List[str],
        remote_path: List[Optional[str]],
        proportion: List[float],
        seq_len_each_sample: int,
        consumed_samples: int,
        batch_size: int,
        tokenizer: AutoTokenizer,
        cache_queue_num: int,
        ckpt_path: str,
        chunk_size: int,
        add_extra_token_to_sequence: bool = True,
    ):
        self.text_dataset = TextChunkDataset(
            local_path=local_path,
            remote_path=remote_path,
            proportion=proportion,
            chunk_size=chunk_size,
        )
        self.tokenizer = tokenizer

        self.consumed_samples = consumed_samples

        self.seq_len_each_sample = seq_len_each_sample
        self.add_extra_token_to_sequence = add_extra_token_to_sequence

        self.cache_queue_num = cache_queue_num
        self.chunk_queue = queue.Queue()
        self.process_chunk: TextTokenChunk = None

        self.ckpt_file_path = ckpt_path
        self.ckpt_db = None

        self.batch_size = batch_size

        self.cached_attention_mask = None
        self.cached_loss_mask = None
        self.cached_position_ids = None

        self.is_master = int(os.environ["RANK"]) == 0

        self.token_it = 0
        self.load_state_dict()

        self.cache_thread = threading.Thread(target=self.cache_chunk, args=())
        self.cache_thread.start()

    def save_state_dict(
        self,
    ):
        # only the master process will save the state dict
        if not self.is_master:
            return

        if self.process_chunk is None:
            self.process_chunk = self.chunk_queue.get(block=True)

        if self.ckpt_db is None:
            self.ckpt_db = duckdb.connect(self.ckpt_file_path)
            self.ckpt_db.execute(
                "CREATE TABLE IF NOT EXISTS state_dict (consumed_samples INTEGER PRIMARY KEY, token_it INTEGER, chunk_state JSON)"
            )

        token_it = self.token_it
        consumed_samples = self.consumed_samples
        chunk_state = json.dumps(self.process_chunk.chunk_state)

        self.ckpt_db.execute(
            "INSERT OR REPLACE INTO state_dict (consumed_samples, token_it, chunk_state) VALUES (?, ?, ?)",
            [consumed_samples, token_it, chunk_state],
        )
        self.ckpt_db.commit()

    def load_state_dict(self):
        # if the consumed_samples is 0, it means the first iteration
        if self.consumed_samples == 0:
            return
        # load from file
        ckpt_db = duckdb.connect(self.ckpt_file_path, read_only=True)
        consumed_samples, token_it, chunk_state = ckpt_db.execute(
            "SELECT consumed_samples, token_it, chunk_state FROM state_dict WHERE consumed_samples = ?",
            [self.consumed_samples],
        ).fetchone()

        chunk_state = json.loads(chunk_state)

        self.consumed_samples = int(consumed_samples)
        self.token_it = int(token_it)
        self.text_dataset.load_state_dict(chunk_state)

    def cache_chunk(self):
        # preload the chunks
        dataset_iter = iter(self.text_dataset)
        while True:
            if self.chunk_queue.qsize() >= self.cache_queue_num:
                time.sleep(0.01)
                continue

            text_chunk, chunk_state = next(dataset_iter)

            tokens_list = self.tokenizer(text_chunk)["input_ids"]
            token_cnt = 0
            token_chunk = []

            for token in tokens_list:
                if token[-1] != self.tokenizer.eos_token_id:
                    token.append(self.tokenizer.eos_token_id)
                token_chunk.extend(token)
                token_cnt += len(token)

            chunk = TextTokenChunk(
                chunk_state=chunk_state,
                text_chunk=text_chunk,
                token_chunk=token_chunk,
            )

            self.chunk_queue.put(chunk)

    def __iter__(self):
        return self

    def __next__(self):
        ret_token_ids = []

        if self.add_extra_token_to_sequence:
            seq_len_each_sample = self.seq_len_each_sample + 1
        else:
            seq_len_each_sample = self.seq_len_each_sample

        seq_len_each_batch = seq_len_each_sample * self.batch_size
        left_need_tokens = seq_len_each_batch

        while len(ret_token_ids) < seq_len_each_batch:
            if self.process_chunk is None:
                self.process_chunk = self.chunk_queue.get(block=True)

            token_chunk = self.process_chunk.token_chunk
            tokens = token_chunk[self.token_it : self.token_it + left_need_tokens]

            tokens_cnt = len(tokens)
            ret_token_ids.extend(tokens)

            self.token_it += tokens_cnt
            left_need_tokens -= tokens_cnt

            if self.token_it >= len(token_chunk):
                self.process_chunk = None
                self.token_it = 0

        assert len(ret_token_ids) == seq_len_each_batch, (
            f"len(ret_token_ids): {len(ret_token_ids)}, seq_len_each_batch: {seq_len_each_batch}"
        )

        self.consumed_samples += 1

        # prepare the mask and others
        text = torch.tensor(
            ret_token_ids, dtype=torch.long, device=torch.device("cpu")
        ).view(self.batch_size, -1)

        ret_tokens = text[:, :-1].contiguous()
        ret_labels = text[:, 1:].contiguous()

        if self.cached_attention_mask is None:
            self.cached_attention_mask = (
                (
                    torch.tril(
                        torch.ones(
                            self.seq_len_each_sample,
                            self.seq_len_each_sample,
                            device=torch.device("cpu"),
                        )
                    )
                    < 0.5
                )
                .unsqueeze(0)
                .repeat(self.batch_size, 1, 1)
            )

        if self.cached_loss_mask is None:
            self.cached_loss_mask = (
                torch.ones(
                    self.seq_len_each_sample,
                    dtype=torch.float,
                    device=torch.device("cpu"),
                )
                .unsqueeze(0)
                .repeat(self.batch_size, 1)
            )

        if self.cached_position_ids is None:
            self.cached_position_ids = (
                torch.arange(
                    self.seq_len_each_sample,
                    dtype=torch.long,
                    device=torch.device("cpu"),
                )
                .unsqueeze(0)
                .repeat(self.batch_size, 1)
            )

        return {
            "tokens": ret_tokens,
            "labels": ret_labels,
            "attention_mask": self.cached_attention_mask,
            "loss_mask": self.cached_loss_mask,
            "position_ids": self.cached_position_ids,
        }
