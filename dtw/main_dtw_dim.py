'''
MTS clustering based on Grey Theory.

'''

import argparse
import numpy as np
import multiTS as mts
from tslearn.metrics import dtw, soft_dtw
import concurrent.futures


def parse_args():
    '''
    Parses arguments.
    '''
    parser = argparse.ArgumentParser(description="Compute DTW distance.")

    parser.add_argument('--input', nargs='?', required=True,
                        help='Input file.')

    parser.add_argument('--output', nargs='?', required=True,
                        help='save file.')

    parser.add_argument('--ts_type', nargs='?', default="dtw",
                        help='view of TS: ALL, EACH.')

    parser.add_argument('--dtw_type', nargs='?', default="dtw",
                        help='type of dtw: dtw, sdtw.')

    parser.add_argument('--gamma', default=1.0, type=float,
                        help='sdtw parameter: gamma')

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
            mts_i.normalize_zscore()
            mts_data[mid] = mts_i

    return mts_data


def compute_pair_sim_dtw(ixx, jxx, mts_i, mts_j, dtw_type, ts_type):
    mts_shape_i = mts_i.shape
    # mts_shape_j = mts_j.shape
    ts_num = mts_shape_i[0]  # both are same
    # print('ts_num: %d' % ts_num)
    ts_sims = []

    if ts_type == 'ALL':
        s1 = mts_i.T
        s2 = mts_j.T
        if dtw_type == 'dtw':
            sim = -dtw(s1, s2)
        elif dtw_type == 'sdtw':
            sim = -soft_dtw(s1, s2, gamma=args.gamma)
        else:
            print('Error DTW type. %s' % dtw_type)
            exit(-1)
        ts_sims = sim
    elif ts_type == 'EACH':
        for ix in range(ts_num):
            s1 = list(mts_i[ix, :])
            s2 = list(mts_j[ix, :])
            if dtw_type == 'dtw':
                sim = -dtw(s1, s2)
            elif dtw_type == 'sdtw':
                sim = -soft_dtw(s1, s2, gamma=args.gamma)
            else:
                print('Error DTW type. %s' % dtw_type)
                exit(-1)
            ts_sims.append(sim)
    else:
        print('Error ts_type: ALL or EACH')
        exit(-1)

    return [ixx, jxx, ts_sims]


def compute_pair_sim_dtw_pl(para):
    ixx = para[0]
    jxx = para[1]
    mts_i = para[2]
    mts_j = para[3]
    dtw_type = para[4]
    ts_type = para[5]

    rst = compute_pair_sim_dtw(ixx, jxx, mts_i, mts_j, dtw_type, ts_type)

    return rst


def compute_dtw_sims(mts_objs, dtw_type, ts_type):
    ids = list(mts_objs.keys())
    ids.sort()
    id_num = len(ids)
    attr_num = mts_objs[0].row
    # print('attr_num: %d' % attr_num)

    ### collect mts pairs
    if ts_type == 'EACH':
        mts_sims = np.zeros((attr_num, id_num, id_num))
    elif ts_type == 'ALL':
        mts_sims = np.zeros((id_num, id_num))
    else:
        print('Error ts_type: ALL or EACH')
        exit(-1)

    mts_pairs = []

    for ix in range(id_num - 1):
        mts_i = mts_objs[ids[ix]].mts_org
        for jx in range(ix + 1, id_num):
            mts_j = mts_objs[ids[jx]].mts_org
            mts_pairs.append([ix, jx, mts_i, mts_j, dtw_type, ts_type])

    ### parallel compute
    chunksize = 100
    with concurrent.futures.ProcessPoolExecutor(max_workers=None) as executor:
        xx_sims = executor.map(compute_pair_sim_dtw_pl, mts_pairs, chunksize=chunksize)

    for rst in xx_sims:
        ixx = rst[0]
        jxx = rst[1]
        val = rst[2]
        # print('ixx=%d, jxx=%d, val_len=%d' % (ixx, jxx, len(val)))
        if ts_type == 'EACH':
            for kx in range(attr_num):
                mts_sims[kx, ixx, jxx] = val[kx]
                mts_sims[kx, jxx, ixx] = val[kx]
        elif ts_type == 'ALL':
            mts_sims[ixx, jxx] = val
            mts_sims[jxx, ixx] = val

    return mts_sims


def main():
    mts_data = load_mts()
    mts_sims = compute_dtw_sims(mts_data, args.dtw_type, args.ts_type)
    mts_sims = np.round(mts_sims, 6)
    np.save(args.output, mts_sims)

    return


if __name__ == "__main__":
    args = parse_args()
    main()
