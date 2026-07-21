import ast
# ast (Abstract Syntax Tree): lets program read Python code as structured tree instead of plain text
# Python figures out structure: what's a function definition, function call, what's inside a loop, etc

def read_source(filepath):
    """Open filepath, read contents, return the text"""

    with open(filepath, "r") as f:
        return f.read()

def parse_source(source_code):
    """Turn the source code into an ast tree we can inspect"""

    return ast.parse(source_code) # converts code into tree object where Python figures our structure
