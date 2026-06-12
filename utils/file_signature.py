"""Stable, inexpensive signatures for large local checkpoints."""

import hashlib
import os


def sampled_file_signature(path, sample_bytes=1024 * 1024):
    size = os.path.getsize(path)
    offsets = sorted({
        0,
        max(0, size // 2 - sample_bytes // 2),
        max(0, size - sample_bytes),
    })
    digest = hashlib.sha256()
    digest.update(str(size).encode("ascii"))
    with open(path, "rb") as file:
        for offset in offsets:
            file.seek(offset)
            block = file.read(min(sample_bytes, size - offset))
            digest.update(str(offset).encode("ascii"))
            digest.update(block)
    return {
        "size": size,
        "sample_bytes": sample_bytes,
        "sample_sha256": digest.hexdigest(),
    }


def validate_file_signature(path, expected, name):
    if "sample_sha256" not in expected:
        actual_size = os.path.getsize(path)
        if actual_size != expected.get("size"):
            raise ValueError(
                f"{name} size does not match cached representation: "
                f"{actual_size} != {expected.get('size')}")
        return
    actual = sampled_file_signature(
        path, sample_bytes=int(expected.get("sample_bytes", 1024 * 1024)))
    if actual != expected:
        raise ValueError(
            f"{name} does not match cached representation signature")
