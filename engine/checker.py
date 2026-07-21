def read_source(filepath):
    """Open filepath, read contents, return the text"""

    with open(filepath, "r") as f:
        return f.read()