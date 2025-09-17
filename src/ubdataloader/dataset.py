import os
import torch
import threading
from transformers import AutoTokenizer
from typing import Optional, Tuple
from streaming import StreamingDataset


class TextDataset:
    def __init__(
        self,
        local_path: str,
        batch_size: int,
        remote_path: Optional[str] = None,
    ):
        """
        "local_path": the local path of the dataset
        "remote_path": the remote path of the dataset
        "batch_size": the batch size of the dataset
        """
        self.stream_dataset = StreamingDataset(
            local=local_path,
            remote=remote_path,
            shuffle=False,
            batch_size=batch_size,
            replication=int(os.environ["WORLD_SIZE"]),
        )
        self.batch_size = batch_size

    def __len__(self):
        return int(len(self.stream_dataset) // self.batch_size)

    def __getitem__(self, index):
        batch_start = index * self.batch_size
        batch_data = self.stream_dataset[batch_start : batch_start + self.batch_size]

        batch_data = [f["text"] for f in batch_data]

        if len(batch_data) == 0:
            return None

        return batch_data


class TextTokenChunkCache:
    def __init__(
        self,
        seq_len: int,
        batch_size: int,
        text_dataset: TextDataset,
        tokenizer: AutoTokenizer,
        preload_tokens: int,
    ):
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.text_dataset = text_dataset
        self.tokenizer = tokenizer
        self.text_chunk_window = {}
        self.token_chunk_window = {}

        self.preload_tokens = preload_tokens
        self.cache_total_tokens = 0
        self.cache_free_tokens = 0

        self.token_thread_num = 32
        self.sample_tokens_num = seq_len * batch_size

        self.cache_thread = None

    def clean_old_chunks(self, text_chunk_index: int):
        old_keys = []

        for key in self.text_chunk_window:
            if key < text_chunk_index:
                old_keys.append(key)

        for key in old_keys:
            self.cache_total_tokens -= len(self.token_chunk_window[key])
            del self.text_chunk_window[key]
            del self.token_chunk_window[key]

    def cache_chunk(self, text_chunk_index: int):
        # preload the chunks
        chunk_index = text_chunk_index

        while (
            self.cache_total_tokens < self.preload_tokens
            or self.cache_free_tokens < self.sample_tokens_num
        ):
            if chunk_index in self.text_chunk_window:
                chunk_index += 1
                continue

            text_chunk = self.text_dataset[chunk_index]

            if text_chunk is None:
                break

            self.text_chunk_window[chunk_index] = text_chunk
            self.token_chunk_window[chunk_index] = []

            tokens_list = self.tokenizer(text_chunk)["input_ids"]
            token_cnt = 0

            for token in tokens_list:
                if token[-1] != self.tokenizer.eos_token_id:
                    token.append(self.tokenizer.eos_token_id)
                self.token_chunk_window[chunk_index].extend(token)
                token_cnt += len(token)

            chunk_index += 1
            self.cache_total_tokens += token_cnt
            self.cache_free_tokens += token_cnt

    def __getitem__(self, index: Tuple[int, int]) -> Tuple[torch.LongTensor, int, int]:
        if self.cache_thread is not None and self.cache_thread.is_alive():
            print("token thread is still alive! maybe the cache thread is too slow!")
            self.cache_thread.join()

        # wait for the cache to be ready
        text_chunk_index, token_chunk_index = index
        # clear the old chunks in the window, keep the latest preload_chunk_num chunks
        self.clean_old_chunks(text_chunk_index)
        if (
            text_chunk_index not in self.text_chunk_window
            or self.cache_free_tokens < self.sample_tokens_num
        ):
            self.cache_chunk(text_chunk_index)

        if self.cache_free_tokens < self.sample_tokens_num:
            return None, None, None

        ret_tokens = []
        left_need_tokens = self.sample_tokens_num

        # must ensure the cache free tokens is enough
        # so we do not need to cache the chunk again
        while len(ret_tokens) < self.sample_tokens_num:
            tokens = self.token_chunk_window[text_chunk_index][
                token_chunk_index : token_chunk_index + left_need_tokens
            ]
            tokens_cnt = len(tokens)
            ret_tokens.extend(tokens)

            token_chunk_index += tokens_cnt
            self.cache_free_tokens -= tokens_cnt
            left_need_tokens -= tokens_cnt

            if token_chunk_index >= len(self.token_chunk_window[text_chunk_index]):
                text_chunk_index += 1
                token_chunk_index = 0

        assert len(ret_tokens) == self.sample_tokens_num, (
            f"len(ret_tokens): {len(ret_tokens)}, sample_tokens_num: {self.sample_tokens_num}"
        )

        # asnyc cache the next chunk
        self.cache_thread = threading.Thread(
            target=self.cache_chunk, args=(text_chunk_index + 1,)
        )
        self.cache_thread.start()

        return ret_tokens, text_chunk_index, token_chunk_index


class TokenDataset:
    text_it: int
    token_it: int

    def __init__(
        self,
        local_path: str,
        seq_len: int,
        batch_size: int,
        tokenizer: AutoTokenizer,
        remote_path: Optional[str] = None,
        preload_tokens: Optional[int] = None,
    ):
        """
        NOTE: batch size is the number of global batch size, each node will have batch_size * seq_len tokens,
        so in data parallel, each node need to chose thier own token.
        """
        self.seq_len = seq_len
        self.batch_size = batch_size

        self.tokenizer = tokenizer

        self.local_path = local_path
        self.remote_path = remote_path

        if preload_tokens is None:
            preload_tokens = seq_len * batch_size * 4

        self.text_dataset = TextDataset(
            local_path=local_path,
            remote_path=remote_path,
            batch_size=batch_size,
        )

        self.text_token_chunk_cache = TextTokenChunkCache(
            seq_len=seq_len,
            batch_size=batch_size,
            text_dataset=self.text_dataset,
            tokenizer=tokenizer,
            preload_tokens=preload_tokens,
        )

        # set the now_text_it to the first text chunk
        self.text_it = 0
        self.token_it = 0

    def __len__(self):
        return len(self.text_dataset)

    def __iter__(self):
        return self

    def __next__(self):
        # only support iterator mode, not support getitem mode
        if self.text_it is None or self.token_it is None:
            raise StopIteration

        tokens, self.text_it, self.token_it = self.text_token_chunk_cache[
            (self.text_it, self.token_it)
        ]

        if tokens is None:
            raise StopIteration

        return tokens
