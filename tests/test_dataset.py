import os
import torch
import argparse
from ubdataloader.dataset import TextChunkDataset


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", nargs="*", required=True)
    parser.add_argument("--seq-len", type=int, default=4096)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--tokenizer-model", type=str, required=True)
    parser.add_argument("--ckpt-path", type=str, required=True)
    args = parser.parse_args()

    # data path like below:
    # weight local_path remote_path
    # 1 /tmp/mdsdata0 None
    assert len(args.data_path) % 3 == 0, (
        "data path should be like below: weight local_path remote_path, but got %s"
        % args.data_path
    )

    local_path = []
    remote_path = []
    proportion = []
    for weight, local, remote in zip(
        args.data_path[::3], args.data_path[1::3], args.data_path[2::3]
    ):
        proportion.append(float(weight))
        local_path.append(local)
        if remote_path == "None":
            remote_path.append(None)
        else:
            remote_path.append(remote)

    # init
    torch.distributed.init_process_group(
        backend="gloo",
        world_size=int(os.environ["WORLD_SIZE"]),
        rank=int(os.environ["RANK"]),
    )

    dataset = TextChunkDataset(
        local_path=local_path,
        remote_path=remote_path,
        proportion=proportion,
        chunk_size=16,
    )

    it = iter(dataset)
    print(next(it))

    # wait all nodes to finish
    torch.distributed.barrier()
