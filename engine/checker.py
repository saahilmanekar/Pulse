# ast (Abstract Syntax Tree): lets program read Python code as structured tree instead of plain text
# Python figures out structure: what's a function definition, function call, what's inside a loop, etc
import ast


def read_source(filepath):
    """Open filepath, read contents, return the text"""

    with open(filepath, "r") as f:
        return f.read()

def parse_source(source_code):
    """Turn the source code into an ast tree we can inspect"""

    # converts code into tree object where Python figures our structure
    return ast.parse(source_code) 

def find_dataloader_calls(tree):
    """Walk the tree and find all DataLoader(...) calls"""

    dataloader_calls = []

    # ast.walk() visits every node in the tree
    for node in ast.walk(tree): 

        # ast.Call: node that Python creates whenever code calls a function/method
        if isinstance(node, ast.Call):
        
            # Tries to get "id" from node.func but returns None if it crashes
            func_name = getattr(node.func, "id", None) 

            if func_name == "DataLoader":
                dataloader_calls.append(node)

    return dataloader_calls

def get_keyword_value_from_call_node(call_node, keyword_name):
    """Given a Call node, find the value passed for a specific keyword"""

    # eg: batch_size=32 should return 32

    for keyword in call_node.keywords:
        if keyword.arg == keyword_name:
            return keyword.value
    return None
