from streaming.base import MDSWriter


if __name__ == "__main__":
    output_dir = "/tmp/mdsdata2"

    columns = {
        "text": "str",
    }

    with MDSWriter(
        out=output_dir,
        columns=columns,
        compression="zstd:7",
        size_limit=1 << 26,
        hashes=["sha1"],
    ) as writer:
        for i in range(50):
            writer.write({"text": f"text2_{i}"})
