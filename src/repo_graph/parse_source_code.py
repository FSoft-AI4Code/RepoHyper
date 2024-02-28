from tree_sitter import Language, Parser
import os
import json
import ast
import tqdm
from joblib import Parallel, delayed

PY_LANGUAGE = Language('/datadrive05/huypn16/treesitter-build/python-java.so', 'python')
parser = Parser()
parser.set_language(PY_LANGUAGE)

def get_node_text(start_byte, end_byte, code):
    return code[start_byte:end_byte]

def parse_func_source_code(code, namespace):
    functions = []
    query = """
        (function_definition 
            name: (identifier) @function.def
            parameters: (parameters) @function.parameters
            body: (_) @function.body
        (#not-a-child-of class_definition))
    """
    query = PY_LANGUAGE.query(query)
    tree = parser.parse(bytes(code, "utf8"))
    captures = query.captures(tree.root_node)
    for i, (node, type) in enumerate(captures):
        if type == 'function.def':
            start_byte = node.start_byte
            name = get_node_text(node.start_byte, node.end_byte, code)
            if "(" in name or ":" in name:
                start_byte = node.start_byte - 1
                name = get_node_text(node.start_byte-1, node.end_byte-1, code)
        elif type == 'function.body':                                                                                                                           
            end_byte = node.end_byte
        if (i+1) % 3 == 0:
            functions.append({"name": f"{namespace}.{name}",
                              "context": "def " + get_node_text(start_byte, end_byte, code),
                              "edges": [],
                              "type_edges": []})
    return functions

def parse_class_source_code(code, namespace):
    classes = []
    query = """(class_definition 
            name: (identifier) @class.def
            body: (_) @class.body)"""
    query = PY_LANGUAGE.query(query)
    tree = parser.parse(bytes(code, "utf8"))
    captures = query.captures(tree.root_node)
    for i, (node, type) in enumerate(captures):
        if type == 'class.def':
            start_byte = node.start_byte
            class_name = get_node_text(node.start_byte, node.end_byte, code)
            if "(" in class_name or ":" in class_name:
                start_byte = node.start_byte - 1
                class_name = get_node_text(node.start_byte-1, node.end_byte-1, code)
            if "." in class_name or " " in class_name or "self" in class_name or "(" in class_name or ":" in class_name or ")" in class_name:
                continue
        elif type == 'class.body':
            end_byte = node.end_byte
        if (i+1) % 2 == 0:
            methods = []
            class_code = "class " + str(get_node_text(start_byte, end_byte, code))
            class_tree = parser.parse(bytes(class_code, "utf8"))
            method_captures = PY_LANGUAGE.query("""(function_definition 
                                                    name: (identifier) @function.def
                                                    parameters: (parameters) @function.parameters
                                                    body: (_) @function.body)""").captures(class_tree.root_node)
            for j, (node, type) in enumerate(method_captures):
                if type == 'function.def':
                    method_start_byte = node.start_byte
                    name = get_node_text(node.start_byte, node.end_byte, class_code)
                elif type == 'function.body':
                    method_end_byte = node.end_byte
                if (j+1) % 3 == 0:
                    method_str = "def " + get_node_text(method_start_byte, method_end_byte, class_code) 
                    methods.append({"name": f"{namespace}.{class_name}.{name}", "context": method_str, "edges": [], "type_edges": []})            
            # add edges from class node to other methods
            # if len(methods) > 1:
            #     methods[0]["edges"].extend([method["name"] for method in methods[1:]])
            classes.append({"name": f"{namespace}.{class_name}",
                            "methods": methods, "context": class_code, "edges": [method["name"] for method in methods], "type_edges": [4 for _ in range(len(methods))]})
    return classes

def parse_imports_from_code(code):
    query = """
        (import_statement) @import.module
        (import_from_statement) @import.from_module
    """
    query = PY_LANGUAGE.query(query)
    tree = parser.parse(bytes(code, "utf8"))
    captures = query.captures(tree.root_node)
    imports = []
    for (node, _) in captures:
        import_string = get_node_text(node.start_byte, node.end_byte, code)
        try:
            ast_tree = ast.parse(import_string)
            import_nodes = [node for node in ast.walk(ast_tree) if isinstance(node, ast.ImportFrom)]
            imported_items = [f"{node.module}.{name.name}" for node in import_nodes for name in node.names]
            for item in imported_items:
                imports.append(item)
        except SyntaxError:
            pass
    return imports

def parse_file(file_name):
    type_edges = []
    code = open(file_name).read()
    namespace = file_name.split("/")[-1].split(".")[0]
    if namespace == "__init__":
        namespace = file_name.split("/")[-2] + "." + namespace
    
    classes = parse_class_source_code(code, namespace)
    for class_code in classes:
        code = code.replace(class_code["context"], "")
        
    functions = parse_func_source_code(code, namespace)
    for function_code in functions:
        code = code.replace(function_code["context"], "").strip()
        
    all_names_in_file = [function["name"] for function in functions]
    type_edges += [3 for _ in range(len(functions))]
    for cls in classes:
        all_names_in_file.append(cls["name"])
        type_edges.append(3)        
                   
    imports = parse_imports_from_code(code)
    code = {"name": f"{namespace}.py", "context": code, "edges": all_names_in_file, "imports": imports, "type_edges": type_edges}
    return functions, classes, code

def check_name_in_nodes(name, name_nodes):
    for node in name_nodes:
        if name in node:
            return node
    return False

def check_node_in_names(node, names):
    for name in names:
        if node.endswith(name):
            return name
    return False

def check_name_in_import(_import, names):
    for name in names:
        if name in _import:
            return name
    for name in names:
        if name.split(".")[-1] == _import.split(".")[-1] and not name.endswith(".py"):
            return name
    return False

def check_file_in_nodes(name, name_nodes):
    for node in name_nodes:
        if ("." not in node) and (name.split(".")[0] == node):
            return node

def parse_source(path, call_graph_json_path):
    contexts_files = []
    all_names = []
    
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.endswith(".py"):
                _functions, _classes, _code = parse_file(os.path.join(root, file))
                all_names.extend([function["name"] for function in _functions])
                for cls in _classes:
                    all_names.append(cls["name"])
                    all_names += [method["name"] for method in cls["methods"]]
                all_names.append(_code["name"])
                relative_path = os.path.relpath(os.path.join(root, file), path)
                contexts_files.append({"relative_path": relative_path, "functions": _functions, "classes": _classes, "code": _code})

    graph = json.load(open(call_graph_json_path))
    code2flow_nodes = [node for node in graph.keys()]
    for _file in contexts_files:
        node_counts = 0
        cant_captured_nodes = 0 
        # checking whether function is in the call graph and add call graph edges if there's any
        for function in _file["functions"]:
            node = check_name_in_nodes(function["name"], code2flow_nodes)
            if node:
                for _node in graph[node]:
                    found_name = check_node_in_names(_node, all_names)
                    if found_name:
                        function["edges"].append(found_name)
                        function["type_edges"].append(1)
                node_counts += 1
            else:
                # print(function["name"])
                cant_captured_nodes += 1
        
        for _class in _file["classes"]:
            node = check_name_in_nodes(_class["name"], code2flow_nodes)
            if node:
                for _node in graph[node]:
                    found_name = check_node_in_names(_node, all_names)
                    if found_name:
                        _class["edges"].append(found_name)
                        _class["type_edges"].append(1)
                node_counts += 1
            else:
                # print(_class["name"])
                cant_captured_nodes += 1
            
        # checking whether method is in the call graph and add call graph edges if there's any (eliminate wrongly parsed methods or classes)
        for _class in _file["classes"]:
            for method in _class["methods"]:
                node = check_name_in_nodes(method["name"], code2flow_nodes)
                if node:
                    for _node in graph[node]:
                        found_name = check_node_in_names(_node, all_names)
                        if found_name:
                            method["edges"].append(found_name)
                            method["type_edges"].append(1)
                    node_counts += 1
                else:
                    # print(method["name"])
                    cant_captured_nodes += 1
        
        # checking whether file is in the graph nodes and add call graph edges if there's any
        node = check_file_in_nodes(_file["code"]["name"], code2flow_nodes)
        if node:
            for _node in graph[node]:
                found_name = check_node_in_names(_node, all_names)
                if found_name:
                    _file["code"]["edges"].append(found_name)
                    _file["code"]["type_edges"].append(2)
        
        # checking whehther import is in the parsed names
        for _import in _file["code"]["imports"]:
            found_name = check_name_in_import(_import, all_names)
            if found_name:
                _file["code"]["edges"].append(found_name)
                _file["code"]["type_edges"].append(0)
        
        # get name from imports, search it inside the implementation, if found, add edges (in case we the call graph is broken)
        for _class in _file["classes"]:
            for method in _class["methods"]:
                for _import in _file["code"]["imports"]:
                    found_name = check_name_in_import(_import, all_names)
                    search_name = _import.split(".")[-1]
                    if (search_name in method["context"]) and (found_name not in method["edges"]) and found_name:
                        method["edges"].append(found_name)
                        method["type_edges"].append(1)
        
        for _function in _file["functions"]:
            for _import in _file["code"]["imports"]:
                found_name = check_name_in_import(_import, all_names)
                search_name = _import.split(".")[-1]
                if (search_name in _function["context"]) and (found_name not in _function["edges"]) and found_name:
                    _function["edges"].append(found_name)
                    _function["type_edges"].append(1)
    # print("Total nodes: ", node_counts)
    # print("Can't captured nodes: ", cant_captured_nodes)
    
    return contexts_files


if __name__ == '__main__':
    # contexts = parse_source("data/perplexity/repos/bitcoinlib")
    file_name = "/datadrive05/huypn16/knn-transformers/data/repobench/repos/bitcoinlib/bitcoinlib/blocks.py"
    code = open(file_name).read()
    classes = parse_class_source_code(code, "blocks")
