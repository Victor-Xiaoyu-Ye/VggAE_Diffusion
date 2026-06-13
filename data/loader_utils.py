"""DataLoader settings safe for MoXing/MemArts clients."""


def multiprocessing_loader_kwargs(num_workers):
    """Avoid inheriting MemArts gRPC handles through Linux fork."""
    if num_workers <= 0:
        return {}
    return {
        "multiprocessing_context": "spawn",
        "persistent_workers": True,
    }
