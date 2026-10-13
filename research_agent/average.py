def merge(base_weights, task_vectors):
    """Return merged weights for the LoRA-targeted backbone layers."""
    return {
        key: base + sum(vector[key] for vector in task_vectors) / len(task_vectors)
        for key, base in base_weights.items()
    }
