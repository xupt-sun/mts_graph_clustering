'''
MTS clustering based on graph.

'''

import argparse
import numpy as np
import multiTS as mts
from ordpy import ordinal_distribution, ordinal_sequence
# from ordpy import complexity_entropy, weighted_permutation_entropy
from itertools import permutations
from gensim.matutils import hellinger
import concurrent.futures


def parse_args():
    '''
    Parses arguments.
    '''
    parser = argparse.ArgumentParser(description="Compute HMM distance.")

    parser.add_argument('--input', nargs='?', required=True,
                        help='Input file.')

    parser.add_argument('--output', nargs='?', required=True,
                        help='save file.')

    parser.add_argument('--emb_len', default=3, type=int,
                        help='embedding length for ordinal pattern.')

    parser.add_argument('--delay', default=1, type=int,
                        help='time delay for ordinal pattern.')

    return parser.parse_args()


def load_mts():
    '''
    load multiple-time-series data set.
    '''
    mts_data = {}
    # mts_num = -1
    # label_num = -1
    attr_num = -1

    mid = -1  # my identifier
    label = None
    t_max = -1
    vecs = []

    in_file_name = args.input
    in_file = open(in_file_name, 'r')

    for line in in_file:
        items = line.split()
        if ('#' == line[0]) or (0 == len(items)):
            # skip comment line and empty line
            continue

        if "mts_num" == items[0]:
            # statistical information
            # mts_num = int(items[1])
            # label_num = int(items[3])
            attr_num = int(items[5])
        elif "id" == items[0]:
            # mts id
            mid = int(items[1])
        elif "label" == items[0]:
            # mts label
            label = items[1]
        elif "t_max" == items[0]:
            # maximum "time" point
            t_max = int(items[1])
        else:
            vecs = np.zeros((attr_num, t_max))
            vecs[0, :] = list(items)

            for ix in range(1, attr_num):
                line = in_file.readline()
                items = line.split()
                vecs[ix, :] = list(items)

            mts_i = mts.MultiTS(mid, label, vecs)
            mts_data[mid] = mts_i

    return mts_data


def enumurate_patterns(k):
    seqs = range(k)
    patterns = list(permutations(seqs, k))

    return patterns


def map_patterns(patterns_list, patterns):
    labels = []

    for p in patterns:
        tuple_p = tuple(p)
        ix = patterns_list.index(tuple_p)
        labels.append(ix)

    return labels


def compute_pair_sim_ord(ixx, jxx, mts_i, mts_j, emb_len, delay):
    mts_shape_i = mts_i.shape
    # mts_shape_j = mts_j.shape
    ts_num = mts_shape_i[0]  # both are same
    ts_sims = []
    patterns_list = enumurate_patterns(emb_len)
    patterns_num = len(patterns_list)

    for ix in range(ts_num):
        ### extract permutation patterns
        patterns_i, dist_i = ordinal_distribution(mts_i[ix, :], dx=emb_len, taux=delay, ordered=True,
                                                  return_missing=True, tie_precision=6)
        patterns_j, dist_j = ordinal_distribution(mts_j[ix, :], dx=emb_len, taux=delay, ordered=True,
                                                  return_missing=True, tie_precision=6)
        ix_patterns_i = map_patterns(patterns_list, patterns_i)
        ix_patterns_j = map_patterns(patterns_list, patterns_j)
        ord_dist_i = np.zeros((patterns_num,))
        ord_dist_j = np.zeros((patterns_num,))

        for jx in range(patterns_num):
            ix = ix_patterns_i[jx]
            ord_dist_i[ix] = dist_i[jx]

            ix = ix_patterns_j[jx]
            ord_dist_j[ix] = dist_j[jx]

        ### compute sim
        sim = -hellinger(ord_dist_i, ord_dist_j)
        ts_sims.append(sim)

    return [ixx, jxx, ts_sims]


def compute_pair_sim_ord_pl(para):
    ixx = para[0]
    jxx = para[1]
    mts_i = para[2]
    mts_j = para[3]
    emb_len = para[4]
    delay = para[5]

    rst = compute_pair_sim_ord(ixx, jxx, mts_i, mts_j, emb_len, delay)

    return rst


def compute_pd_sims(mts_objs, emb_len, delay):
    ids = list(mts_objs.keys())
    ids.sort()
    id_num = len(ids)
    attr_num = mts_objs[0].row

    ### collect mts pairs
    mts_sims = np.zeros((attr_num, id_num, id_num))
    mts_pairs = []

    for ix in range(id_num - 1):
        mts_i = mts_objs[ids[ix]].mts_org
        for jx in range(ix + 1, id_num):
            mts_j = mts_objs[ids[jx]].mts_org
            mts_pairs.append([ix, jx, mts_i, mts_j, emb_len, delay])

    ### parallel compute
    chunksize = 100
    with concurrent.futures.ProcessPoolExecutor(max_workers=None) as executor:
        xx_sims = executor.map(compute_pair_sim_ord_pl, mts_pairs, chunksize=chunksize)

    for rst in xx_sims:
        ixx = rst[0]
        jxx = rst[1]
        val = rst[2]

        for kx in range(attr_num):
            mts_sims[kx, ixx, jxx] = val[kx]
            mts_sims[kx, jxx, ixx] = val[kx]

    return mts_sims


def main():
    mts_data = load_mts()

    mts_sims = compute_pd_sims(mts_data, args.emb_len, args.delay)
    np.save(args.output, mts_sims)

    return


if __name__ == "__main__":
    args = parse_args()
    main()
