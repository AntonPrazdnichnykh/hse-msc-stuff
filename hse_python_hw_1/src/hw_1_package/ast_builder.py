import ast
import networkx as nx
import matplotlib.pyplot as plt
from typing import Any, Optional, NoReturn, Tuple
from collections import defaultdict


class ASTVisualizer(ast.NodeVisitor):
    def __init__(self):
        self.stack = []
        self.ast = nx.Graph()
        self.label_dict = {}
        self.class_count = defaultdict(int)

    def extend_ast(self, node: ast.AST, node_label: str):
        class_name = node.__class__.__name__
        node_name = f"{class_name}_{self.class_count[class_name]}"
        self.class_count[class_name] += 1
        parent_name = None
        if self.stack:
            parent_name = self.stack[-1]

        self.stack.append(node_name)
        self.ast.add_node(node_name)
        self.label_dict[node_name] = node_label

        if parent_name:
            self.ast.add_edge(parent_name, node_name)

        super(self.__class__, self).generic_visit(node)
        self.stack.pop()

    def visit_FunctionDef(self, node: ast.FunctionDef) -> Any:
        label = f"{node.__class__.__name__}\nname: {node.name}"
        self.extend_ast(node, label)

    def visit_arg(self, node: ast.arg) -> Any:
        label = f"{node.__class__.__name__}\nname: {node.arg}"
        self.extend_ast(node, label)

    def visit_Name(self, node: ast.Name) -> Any:
        label = f"{node.__class__.__name__}\nid: {node.id}"
        self.extend_ast(node, label)

    def visit_Constant(self, node: ast.Constant) -> Any:
        label = f"{node.__class__.__name__}\nvalue: {node.value}"
        self.extend_ast(node, label)

    def generic_visit(self, node: ast.AST) -> Any:
        label = node.__class__.__name__
        self.extend_ast(node, label)

    def visualize(self, save_fn: Optional[str] = None, figsize: Optional[Tuple[int, int]] = None) -> NoReturn:
        plt.figure(1, figsize=figsize)
        nx.draw(self.ast, labels=self.label_dict, with_labels=True)
        if save_fn:
            plt.savefig(save_fn)
        plt.show()


def visualize_ast(input_fn: str, output_fn: Optional[str] = None, figsize: Optional[Tuple[int, int]] = None):
    with open(input_fn, 'r') as fin:
        code = fin.read()

    root = ast.parse(code)
    print(ast.dump(root,  indent=4))
    ast_vis = ASTVisualizer()
    ast_vis.visit(root)
    ast_vis.visualize(save_fn=output_fn, figsize=figsize)


if __name__ == "__main__":
    visualize_ast("fibonacci.py", "artifacts/fibonacci_ast.png", figsize=(15, 15))
