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

def get_call_name(func_node):
    """Return the full name of the function being called"""

    # If the func_node has a plain name with no dots, just grab the text (.id)
    if isinstance(func_node, ast.Name):
        return func_node.id

    # Recursively call the function in case there is a dot
    elif isinstance(func_node, ast.Attribute):

        # Pass in everything to the left of the last dot
        parent_name = get_call_name(func_node.value)

        if parent_name:
            return f"{parent_name}.{func_node.attr}"
        else:
            return func_node.attr

    return None

def find_dataloader_calls(tree):
    """Walk the tree and find all DataLoader(...) calls"""

    dataloader_calls = []

    # ast.walk() visits every node in the tree
    for node in ast.walk(tree): 

        # ast.Call: node that Python creates whenever code calls a function/method
        if isinstance(node, ast.Call):

            # Tries to get the actual name of the function using the helper function
            func_name = get_call_name(node.func)

            if func_name == "DataLoader" or (func_name and func_name.endswith(".DataLoader")):
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

def find_variable_literal_assignment(tree, var_name):
    """Look for var_name = literal in the file"""

    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):

            # node.targets: list of nodes representing what is being assigned to left side of assignment statement 
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == var_name:
                    value = literal_or_none(node.value)
                    if value is not None:
                        return value
    return None

def resolve_value(value_node, tree):
    """Attempts to resolve a value to a literal"""

    # Case 1: Direct literal (eg: num_workers=0)

    literal_value = literal_or_none(value_node)
    if literal_value is not None:
        return literal_value, True

    # Case 2: Plain local variable (eg: n=0, num_workers=n)

    if isinstance(value_node, ast.Name):
        local_value = find_variable_literal_assignment(tree, value_node.id)
        if local_value is not None:
            return local_value, True
    
    # Case 3: couldn't resolve all

    return None, False

# Check SPECIFIC rules/attributes using the ABOVE helper functions

def check_num_workers(tree):
    """Check DataLoader calls for a missing, zero, or unresolvable num_workers"""

    findings = []

    for call_node in find_dataloader_calls(tree):
        value_node = get_keyword_value_from_call_node(call_node, "num_workers")

        # Case 1: Not set at all
        if value_node is None:
            findings.append({
                "line": call_node.lineno,
                "message": "num_workers is not set at all, meaning it defaults to 0. Data loads on a single process, which can leave the GPU waiting. Worth testing higher values (e.g. 2, 4, 8)."
            })
            continue

        value, resolved = resolve_value(value_node, tree)

        # Case 2: Not resolvable
        if not resolved:
            findings.append({
                "line": call_node.lineno,
                "message": "num_workers isn't a plain literal or simple variable, so Pulse can't determine or test its value. For best results, set it to a plain variable (e.g. num_workers=n, where n=0 is set nearby)."
            })

        # Case 3: Resolvable
        elif value == 0:
            findings.append({
                "line": call_node.lineno,
                "message": "num_workers is set to 0. This can leave the GPU waiting. Worth testing higher values (e.g. 2, 4, 8)."
            })
    return findings

def check_pin_memory(tree):
    """Check all DataLoader calls for a missing, False, or unresolvable pin_memory"""

    findings = []

    for call_node in find_dataloader_calls(tree):
        value_node = get_keyword_value_from_call_node(call_node, "pin_memory")

        # Case 1: Not set at all
        if value_node is None:
            findings.append({
                "line": call_node.lineno,
                "message": "pin_memory is not set at all, meaning it defaults to False. If you're using a GPU, this usually speeds up moving data from CPU to GPU at little to no cost. Worth testing pin_memory=True."
            })
            continue

        value, resolved = resolve_value(value_node, tree)

        # Case 2: Not resolvable
        if not resolved:
            findings.append({
                "line": call_node.lineno,
                "message": "pin_memory isn't a plain literal or simple local variable, so Pulse can't determine or test its value. For best results, use a plain variable (e.g. use_pin = True, then pin_memory=use_pin)."
            })

        # Case 3: Resolvable
        elif value is False:
            findings.append({
                "line": call_node.lineno,
                "message": "pin_memory is set to False. If you're using a GPU, this usually speeds up moving data from CPU to GPU at little to no cost. Worth testing pin_memory=True."
            })

    return findings

def check_amp_usage(source_code):
    """Check whether mixed precision (autocast/GradScaler) appears anywhere in the file"""

    findings = []

    # Check directly in source code

    if "autocast" not in source_code and "GradScaler" not in source_code:
        findings.append({
            "line": 1,
            "message": "No sign of mixed precision (autocast/GradScaler) anywhere in this file. On a compatible GPU, this often speeds up training with minimal accuracy impact, worth testing."
        })

    return findings

def check_excessive_logging(tree):
    """Flag print() calls found inside a training loop (for/while)"""
    
    findings = []

    for node in ast.walk(tree):
        if isinstance(node, (ast.For, ast.While)):
            for child in ast.walk(node):
                if isinstance(child, ast.Call):
                    func_name = get_call_name(child.func)
                    if func_name == "print":
                        findings.append({
                            "line": child.lineno,
                            "message": "print() call found inside a loop. Printing every step can add overhead, especially if it forces GPU synchronization (eg: printing a .item() value). Consider printing every N steps instead."
                        })
    return findings

def check_gpu_cpu_sync(tree):
    """Flag .item(), .cpu(), .numpy() calls found inside a training loop"""

    findings = []

    sync_method_names = {"item", "cpu", "numpy"}

    for node in ast.walk(tree):
        if isinstance(node, (ast.For, ast.While)):
            for child in ast.walk(node):
                if isinstance(child, ast.Call):
                    func_name = get_call_name(child.func)
                    if func_name and func_name.split(".")[-1] in sync_method_names:
                        findings.append({
                            "line": child.lineno,
                            "message": f"'.{func_name.split('.')[-1]}()' call found inside a loop. This forces the GPU to stop and sync with the CPU, which can add up if called every step. Consider calling it less frequently or accumulating values on the GPU instead."
                        })

    return findings

def check_checkpoint_in_loop(tree):
    """Flag torch.save(...) calls found inside a training loop"""

    findings = []

    for node in ast.walk(tree):
        if isinstance(node, (ast.For, ast.While)):
            for child in ast.walk(node):
                if isinstance(child, ast.Call):
                    func_name = get_call_name(child.func)
                    if func_name == "torch.save":
                        findings.append({
                            "line": child.lineno,
                            "message": "torch.save() call found inside a loop. Saving a checkpoint every step can add significant disk I/O overhead. Consider checkpointing every N steps or once per epoch instead."
                        })
    return findings

def check_persistent_workers(tree):
    """If num_workers > 0, check whether persistent_workers is also True"""

    findings = []

    for call_node in find_dataloader_calls(tree):
        
        # Depends on num_workers, so we need to grab this value
        num_workers_node = get_keyword_value_from_call_node(call_node, "num_workers")
        num_workers_value, num_workers_resolved = resolve_value(num_workers_node, tree)

        # persistent_workers is truly beneifical if there is actual num_workers value
        if not num_workers_resolved or num_workers_value is None or num_workers_value <= 0:
            continue


        persistent_workers_node = get_keyword_value_from_call_node(call_node, "persistent_workers")
        persistent_workers_value, persistent_workers_resolved = resolve_value(persistent_workers_node, tree)

        # Case 1: Not set at all
        if persistent_workers_node is None:
            findings.append({
                "line": call_node.lineno,
                "message": f"num_workers is {num_workers_value}, but persistent_workers is not set, so it defaults to False. Worker processes will restart every epoch, adding startup overhead. Worth testing persistent_workers=True."
            })
        
        # Case 2: Not resolvable
        elif not persistent_workers_resolved:
            findings.append({
                "line": call_node.lineno,
                "message": "persistent_workers isn't a plain literal or simple local variable, so Pulse can't determine its value."
            })
        
        # Case 3: Resolvable
        elif persistent_workers_value is False:
            findings.append({
                "line": call_node.lineno,
                "message": f"num_workers is {num_workers_value}, but persistent_workers is set to False. Worker processes will restart every epoch, adding startup overhead. Worth testing persistent_workers=True."
            })

    return findings

def check_non_blocking_transfer(tree):
    """Flag .to(device)/.cuda() calls inside a loop that don't pass non_blocking=True"""

    findings = []

    transfer_method_names = {"to", "cuda"}

    for node in ast.walk(tree):
        if isinstance(node, (ast.For, ast.While)):
            for child in ast.walk(node):
                if isinstance(child, ast.Call):
                    func_name = get_call_name(child.func)

                    if func_name and func_name.split(".")[-1] in transfer_method_names:
                        non_blocking_node = get_keyword_value_from_call_node(child, "non_blocking")
                        non_blocking_value = literal_or_none(non_blocking_node)

                        if non_blocking_node is None or non_blocking_value is False:
                            findings.append({
                                "line": child.lineno,
                                "message": "Tensor transfer (.to()/.cuda()) inside a loop doesn't pass non_blocking=True. If pin_memory=True is also set on your DataLoader, this can allow data transfer and GPU computation to overlap. Worth testing."
                            })
    return findings

# Final function that ties everything together

def run_static_check(filepath):
    """Run all checks against a file and return the combined findings"""

    source_code = read_source(filepath)
    tree = parse_source(source_code)

    all_findings = []

    num_workers_findings = check_num_workers(tree)
    pin_memory_findings = check_pin_memory(tree)
    amp_usage_findings = check_amp_usage(source_code)
    excessive_logging_findings = check_excessive_logging(tree)
    check_gpu_cpu_sync_findings = check_gpu_cpu_sync(tree)
    check_checkpoint_in_loop_findings = check_checkpoint_in_loop(tree)
    check_persistent_workers_findings = check_persistent_workers(tree)
    check_non_blocking_transfer_findings = check_non_blocking_transfer(tree)

    all_findings += num_workers_findings
    all_findings += pin_memory_findings
    all_findings += amp_usage_findings
    all_findings += excessive_logging_findings
    all_findings += check_gpu_cpu_sync_findings
    all_findings += check_checkpoint_in_loop_findings
    all_findings += check_persistent_workers_findings
    all_findings += check_non_blocking_transfer_findings
    
    return all_findings