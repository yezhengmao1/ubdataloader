import torch
from transformers import AutoTokenizer
from .dataset import TextTokenChunkCache
from typing import Optional, List


class TokenStreamDataLoader:
    text_it: int
    token_it: int

    def __init__(
        self,
        local_path: List[str],
        remote_path: List[Optional[str]],
        proportion: List[float],
        seq_len: int,
        consumed_samples: int,
        micro_batch_size: int,
        data_parallel_rank: int,
        data_parallel_size: int,
        tokenizer_path: str,
        ckpt_path: str,
        cache_queue_num: Optional[int] = None,
        chunk_size: int = 4096,
    ):
        """
        NOTE: each iteration, we will load the seq_len * micro_batch_size * data_parallel_size tokens
        and each data parallel rank will load the seq_len * micro_batch_size tokens
        1. use the consumed_samples (due it's not consistent with the iteration) to resume the training from the last checkpoint
        """
        self.seq_len = seq_len
        self.micro_batch_size = micro_batch_size
        self.consumed_samples = consumed_samples

        self.data_parallel_rank = data_parallel_rank
        self.data_parallel_size = data_parallel_size

        if cache_queue_num is None:
            cache_queue_num = 1024

        self.text_token_chunk_cache = TextTokenChunkCache(
            local_path=local_path,
            remote_path=remote_path,
            proportion=proportion,
            seq_len_each_sample=seq_len,
            consumed_samples=consumed_samples,
            batch_size=micro_batch_size * data_parallel_size,
            tokenizer=AutoTokenizer.from_pretrained(
                tokenizer_path,
                trust_remote_code=True,
                add_eos_token=True,
            ),
            cache_queue_num=cache_queue_num,
            ckpt_path=ckpt_path,
            chunk_size=chunk_size,
        )

    def __iter__(self):
        return self

    def __next__(self):
        # only support iterator mode, not support getitem mode
        self.text_token_chunk_cache.save_state_dict()

        rank_start_idx = self.data_parallel_rank * self.micro_batch_size
        rank_end_idx = rank_start_idx + self.micro_batch_size
        data = next(self.text_token_chunk_cache)

        return {
            "tokens": data["tokens"][rank_start_idx:rank_end_idx],
            "labels": data["labels"][rank_start_idx:rank_end_idx],
            "attention_mask": data["attention_mask"][rank_start_idx:rank_end_idx],
            "loss_mask": data["loss_mask"][rank_start_idx:rank_end_idx],
            "position_ids": data["position_ids"][rank_start_idx:rank_end_idx],
        }
