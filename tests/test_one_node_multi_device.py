import os
import torch
import argparse
from transformers import AutoTokenizer
from ubdataloader.dataset import TokenDataset


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local-path", type=str, required=True)
    parser.add_argument("--seq-len", type=int, default=4096)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--tokenizer-model", type=str, required=True)
    args = parser.parse_args()

    # init
    torch.distributed.init_process_group(
        backend="gloo",
        world_size=int(os.environ["WORLD_SIZE"]),
        rank=int(os.environ["RANK"]),
    )

    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_model,
        add_eos_token=True,
    )
    token_dataset = TokenDataset(
        local_path=args.local_path,
        seq_len=args.seq_len,
        batch_size=args.batch_size,
        tokenizer=tokenizer,
    )

    for data in token_dataset:
        if os.environ.get("RANK") == "0":
            assert len(data) == args.seq_len * args.batch_size

    print("done")
    # wait all nodes to finish
    torch.distributed.barrier()
