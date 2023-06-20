def invert(dictionary: dict) -> dict:
    """Inverts or reverses a dictionary"""
    return {v: k for k, v in dictionary.items()}
