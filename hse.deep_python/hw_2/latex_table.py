def create_latex_table(lst, save_fn="artifacts/table_from_list.tex"):
    n_cols = len(lst[0])
    for row in lst:
        assert len(row) == n_cols, "Given list is not a rectangular table"

    out = "\\begin{center}\\begin{tabular}{" + "|c" * n_cols + "|}"
    for row in lst:
        out += " \\hline " + ' & '.join(row) + "\\\\"
    out += " \\hline\\end{tabular}\\end{center}"

    with open(save_fn, 'w') as f:
        f.write(out)


TEST_CASE = [
        ['1', '2', '3'],
        ['4ksdjjkssvkmvs', '5', '6'],
        ['7', '8', '9']
    ]

if __name__ == "__main__":
    create_latex_table(TEST_CASE)
