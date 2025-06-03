import ast
import random
import subprocess
import pickle

vnames = pickle.load(open("../vnames.pkl", "rb"))
keywords = ['and', 'as', 'assert', 'break', 'class', 'continue', 'def', 'del', 'elif', 'else',
        'except', 'False', 'finally', 'for', 'from', 'global', 'if', 'import', 'in', 'is',
        'lambda', 'None', 'nonlocal', 'not', 'or', 'pass', 'raise', 'return', 'True', 'try',
        'while', 'with', 'yield']
built_in_functions = ['abs', 'all', 'any', 'ascii', 'bin', 'bool', 'bytearray', 'bytes', 'callable', 'chr',
        'classmethod', 'compile', 'complex', 'delattr', 'dict', 'dir', 'divmod', 'enumerate',
        'eval', 'exec', 'filter', 'float', 'format', 'frozenset', 'getattr', 'globals', 'hasattr',
        'hash', 'help', 'hex', 'id', 'input', 'int', 'isinstance', 'issubclass', 'iter', 'len',
        'list', 'locals', 'map', 'max', 'memoryview', 'min', 'next', 'object', 'oct', 'open',
        'ord', 'pow', 'print', 'property', 'range', 'repr', 'reversed', 'round', 'set', 'setattr',
        'slice', 'sorted', 'staticmethod', 'str', 'sum', 'super', 'tuple', 'type', 'vars', 'zip']
preserve_words = set(["self", "super", "Exception", "__init__", "__main__"])
preserve_words.union(set(keywords))
preserve_words.union(set(built_in_functions))
variable_names = [x for (x, _) in vnames.most_common(50000) if x not in preserve_words]


class CodeMasker(ast.NodeTransformer):
    def __init__(self, lineno, tree_variables, state):
        self.lineno = lineno
        self.tree_variables = tree_variables
        self.state = state

    def visit_FunctionDef(self, node):
        if node.lineno in self.lineno:
            return None
        return self.generic_visit(node)

    def visit_Import(self, node):
        if node.lineno in self.lineno:
            return None
        return self.generic_visit(node)

    def visit_ImportFrom(self, node):
        if node.lineno in self.lineno:
            return None
        return self.generic_visit(node)

    def visit_Assign(self, node):
        if node.lineno in self.lineno:
            return None
        return self.generic_visit(node)

    def visit_For(self, node):
        if node.lineno in self.lineno:
            if isinstance(node.target, ast.Tuple):
                new_elements = [ast.Name(id='_', ctx=ast.Store()) for _ in node.target.elts]
                node.target.elts = new_elements
            else:
                node.target.id = '_'
        return self.generic_visit(node)

    def visit_Return(self, node):
        if node.lineno in self.lineno:
            return None
        return self.generic_visit(node)

    def visit_Expr(self, node):
        if node.lineno in self.lineno:
            return None
        return self.generic_visit(node)

    def visit_AugAssign(self, node):
        if node.lineno in self.lineno:
            return None
        return self.generic_visit(node)

    def visit_If(self, node):
        if node.lineno in self.lineno:
            name = "[MASK]"
            new_cond = ast.parse(name, mode='eval').body
            node.test = new_cond

        for else_node in node.orelse:
            if hasattr(else_node, 'lineno') and else_node.lineno - 1 in self.lineno:
                name = "[MASK]"
                new_cond = ast.parse(name, mode='eval').body
                node.test = new_cond

        return self.generic_visit(node)


def mask_remove_deadcode(code, line_number, language, state):
    if language == "python":
        if state == "train":
            tree = ast.parse(code)
            tree_variables = set()
            for node in ast.walk(tree):
                if isinstance(node, ast.Name):
                    tree_variables.add(node.id)

            transformer = CodeMasker(line_number, tree_variables, state)
            new_tree = transformer.visit(tree)
            new_code = ast.unparse(new_tree)
            return new_code
        elif state == "shapley":
            new_code = []
            for lineno in line_number:
                tree = ast.parse(code)
                tree_variables = set()
                for node in ast.walk(tree):
                    if isinstance(node, ast.Name):
                        tree_variables.add(node.id)

                transformer = CodeMasker([lineno], tree_variables, state)
                new_tree = transformer.visit(tree)
                new_code.append(ast.unparse(new_tree))
            return new_code

    elif language == "java":
        lines = ','.join([str(x) for x in line_number])
        command = ['java', '-jar', "mask_dcode_java_jar/dcode_java.jar", code, lines, state]
        process = subprocess.Popen(command, shell=False, stdout=subprocess.PIPE, text=True)
        masked_code, error = process.communicate()
        if error:
            raise Exception("Error executing command: " + error)
        if state == "train":
            return masked_code
        elif state == "shapley":
            return masked_code.split("------------------------------------------------")[:-1]

