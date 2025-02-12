'''
MTS clustering.

'''

import argparse
import numpy as np


def parse_args():
    '''
    Parses arguments.
    '''
    parser = argparse.ArgumentParser(description="Run find_ek.")

    parser.add_argument('--input', nargs='?', required=True,
                        help='Input file.')

    return parser.parse_args()


def main():
    eks = np.load(args.input)
    attr_num, para_num = eks.shape
    k_ix = 1
    e_ix = 3

    for ixx in range(attr_num):
        print('%d %.1f' % (eks[ixx, k_ix], eks[ixx, e_ix]))

if __name__ == "__main__":
    args = parse_args()
    main()
