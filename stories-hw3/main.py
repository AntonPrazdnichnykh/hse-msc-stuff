from data import read_dataset
from mon_conj import montonous_conjugate
import sys


def main():
    in_path, out_path = sys.argv[1:]
    x, y = read_dataset(in_path)
    diff, std_err, conj_measure = montonous_conjugate(x, y)
    with open(out_path, 'w') as f:
        f.write(f"{round(diff)} {round(std_err)} {round(conj_measure, 2)}")


if __name__ == "__main__":
    main()
