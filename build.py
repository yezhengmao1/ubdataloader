import glob
import os
import pyarrow.parquet as pq
from multiprocessing import Pool
from streaming.base import MDSWriter
from typing import Dict, Any, List


def read_parquet(file_path) -> List[Dict[str, Any]]:
    table = pq.read_table(file_path, columns=["text"])
    texts = table["text"].to_pylist()
    return [{"text": text} for text in texts]


def init_worker():
    pid = os.getpid()
    print(f"Initializing worker {pid}")


if __name__ == "__main__":
    columns = {
        "text": "str",
    }

    input_dir = "/volume/pt-data/data/data-deliver/opensource/huggingface.co/datasets/allenai/olmo-mix-1124/v1.0.0-20250829/"
    output_dir = "/volume/pt-train/users/zmye/data/mdsdata"

    parquet_files = glob.glob(os.path.join(input_dir, "*.zstd.parquet"))[:2]

    import multiprocessing

    cpu_count = max(1, multiprocessing.cpu_count() - 8)

    with Pool(initializer=init_worker, processes=cpu_count) as pool:
        with MDSWriter(
            out=output_dir,
            columns=columns,
            compression="zstd:7",
            size_limit=1 << 26,
            hashes=["sha1"],
        ) as writer:
            for texts in pool.imap(read_parquet, parquet_files):
                for text in texts:
                    writer.write(text)
