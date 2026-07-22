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

    # eg: batch_size=32 should return a form of 32

    for keyword in call_node.keywords:
        if keyword.arg == keyword_name:
            return keyword.value
    return None

def literal_or_none(node):
    """Attempts to read a plain literal value (eg: 0, 32, True) from ast node"""

    # Previous function gives us back raw ast node, not the actual literal value like 32

    if node is None:
        return None

    try:
        return ast.literal_eval(node)
    except (ValueError, TypeError):
        return None

def check_num_workers(tree):
    """Check all DataLoader calls for a missing or zero num_workers"""

    findings = []

    for call_node in find_dataloader_calls(tree):
        num_workers_node = get_keyword_value_from_call_node(call_node, "num_workers")
        num_workers_value = literal_or_none(num_workers_node)

        if num_workers_node is None or num_workers_value == 0:
            findings.append({

                # .lineno provides the line it was written in the source file
                "line": call_node.lineno,
                "message": "num_workers is missing or set to 0. Data loads on a single process, which can leave the GPU waiting. Worth testing higher values (eg: 2,4,8) to see what's fastest for your specific setup."
            })

    return findings
