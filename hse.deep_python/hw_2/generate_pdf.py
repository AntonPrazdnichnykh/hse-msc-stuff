import subprocess
from hw_1_package import ast_builder


def insert_img(img_path: str, scale: float = 1.):
    return f"\\includegraphics[scale={scale}]{{{img_path}}}"


def generate_pdf(path_to_tex="artifacts/full.tex"):
    tex_str = "\\documentclass{article}\\usepackage[utf8]{inputenc}\\usepackage{graphicx}\\begin{document}"
    with open("artifacts/table_from_list.tex", 'r') as f:
        tex_str += f.read()
    ast_builder.visualize_ast("../hw_1/fibonacci.py", "artifacts/fibonacci_ast.png", figsize=(15, 15))
    tex_str += insert_img("artifacts/fibonacci_ast.png", scale=0.3)
    tex_str += "\\end{document}"
    with open(path_to_tex, 'w') as f:
        f.write(tex_str)
    return_value = subprocess.call(['pdflatex', '-output-directory', 'artifacts/', path_to_tex], shell=False)
    print(return_value)


if __name__ == "__main__":
    generate_pdf()
