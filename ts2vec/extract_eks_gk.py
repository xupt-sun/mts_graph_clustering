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

    parser.add_argument('--ix', type=int, default=1,
                        help='run time id.')

    return parser.parse_args()


def main():
    eks = np.load(args.input)
    ix = args.ix - 1
    eks_ix = eks[ix]
    attr_num, para_num = eks_ix.shape
    k_ix = 1
    e_ix = 3

    for ixx in range(attr_num):
        print('%d %.1f' % (eks_ix[ixx, k_ix], eks_ix[ixx, e_ix]))

if __name__ == "__main__":
    args = parse_args()
    main()
