'''
MTS clustering.

'''

import argparse
import numpy as np
import multiTS as mts
import mtsClustering3 as mts_clust


def parse_args():
    '''
    Parses arguments.
    '''
    parser = argparse.ArgumentParser(description="Run mts-clustering-ours.")

    parser.add_argument('--input', nargs='?', required=True,
                        help='Input file.')

    parser.add_argument('--input_sims', nargs='?', required=True,
                        help='Input file of similarity matrix.')

    parser.add_argument('--eks', nargs='?', required=True,
                        help='Input file of ek.')

    parser.add_argument('--sim_flag', default=0, type=int,
                        help='0, do not use sim threshold; 1, use.')

    parser.add_argument('--combine_flag', default=0, type=int,
                        help='0, single-layer; 1, combine multi-layer by overlay; 2, combine multi-layer by average.')

    parser.add_argument('--min_layer_num', default=1, type=int,
                        help='minimum number of layer connections, for overlay combination.')

    parser.add_argument('--cd_alg', nargs='?', default="louvain",
                        help='community detection algorithm: louvain, nmf, mnmf, or flpa.')

    parser.add_argument('--comnum', default=4, type=int,
                        help='community number (for NMF).')

    parser.add_argument('--nmf_tol', default=1e-6, type=float,
                        help='tolerance level of NMF.')

    parser.add_argument('--nmf_max_it', default=200, type=int,
                        help='maximum iterations of NMF optimization.')

    parser.add_argument('--mnmf_reg_lam', default=0.1, type=float,
                        help='regularization parameter for MNMF.')

    parser.add_argument('--clusters', nargs='?', required=True,
                        help='cluster file.')

    return parser.parse_args()


def load_mts():
    '''
    load multiple-time-series data set.
    '''
    mts_data = {}
    # mts_num = -1
    # label_num = -1
    attr_num = -1

    mid = -1        # my identifier
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


def output_mts(mts_data, file_name):
    ids = list(mts_data.keys())
    ids.sort()
    file_mts = open(file_name, "w")

    for key in ids:
        mts = mts_data[key]

        line = ("# mts %d\n" % mts.mid)
        file_mts.write(line)

        line = ("id %d\n" % mts.mid)
        file_mts.write(line)

        line = ("label %s\n" % mts.label)
        file_mts.write(line)

        line = ("t_max %d\n" % mts.column)
        file_mts.write(line)

        for rix in range(mts.row):
            line = ''

            for cix in range(mts.column):
                line += (" %.4f" % mts.mts_norm[rix, cix])

            line += '\n'
            file_mts.write(line)

        line = '\n'
        file_mts.write(line)

    file_mts.close()

    return


def output_clusters(clusters, graph_id_map):
    cluster_file = open(args.clusters, "w")
    cluster_ids = list(clusters.keys())
    cluster_ids.sort()

    for id in cluster_ids:
        cluster_mems = clusters[id]
        for mem in cluster_mems:
            mts_id = graph_id_map[mem]
            line = '%d %d\n' % (mts_id, id)
            cluster_file.write(line)

    cluster_file.close()

    return


def restore_coms(high_coms, node_com_map):
    real_coms = {}

    for cix in high_coms.keys():
        low_cixs = high_coms[cix]
        real_mems = []

        for lcix in low_cixs:
            real_mems += node_com_map[lcix]

        real_mems.sort()
        real_coms[cix] = real_mems

    return real_coms


def print_coms(coms, graph_id_map):
    cids = list(coms.keys())
    cids.sort()

    for id in cids:
        line = '%d: ' % id
        mems = coms[id]

        for mid in mems:
            rmid = graph_id_map[mid]
            line += ' %d' % rmid

        print(line)

    return


def cluster_mts(mts_data, sim_scores, eks):
    cluster = mts_clust.MtsClustering(mts_data)

    if args.combine_flag == 0:
        # single layer network
        knn = int(eks[0])
        sim_th = 0.0
        if args.sim_flag == 1:
            sim_th = eks[1]

        mts_graph, all_same_flag, graph_id_map = cluster.build_mts_graph_knn_sim(sim_scores, knn, sim_th)

        if all_same_flag == 1:
            print("all similarity scores are same. not necessary.")
            exit(0)

        coms = cluster.cluster_mtsset(mts_graph, args.cd_alg, args.comnum, args.nmf_max_it, args.nmf_tol)

    else:
        ### multi layer network
        if args.combine_flag == 2:  ### combine by average.
            knn = int(eks[0])
            sim_th = 0.0
            if args.sim_flag == 1:
                sim_th = eks[1]

            mts_graph, all_same_flag, graph_id_map = cluster.combine_multilayer_graphes_avg(sim_scores,knn, sim_th)

            if all_same_flag == 1:
                print("all similarity scores are same. not necessary.")
                exit(0)

            coms = cluster.cluster_mtsset(mts_graph, args.cd_alg, args.comnum, args.nmf_max_it, args.nmf_tol)
        else:
            knns = np.array(eks[:, 0]).astype(int)
            sim_ths = np.zeros(knns.shape)
            if args.sim_flag == 1:
                sim_ths = eks[:, 1]

            mts_graphes, all_same_flags, graph_id_map = cluster.build_mts_graph_multilayer_knn_sim(sim_scores, knns, sim_ths)

            if args.combine_flag == 1:  ### combine by overlay
                mts_graph, graph_id_map = cluster.combine_multilayer_graphes_overlay(mts_graphes, all_same_flags, args.min_layer_num)
                coms = cluster.cluster_mtsset(mts_graph, args.cd_alg, args.comnum, args.nmf_max_it, args.nmf_tol)
            elif args.combine_flag == 3:  ### multi-layer
                coms = cluster.cluster_mtsset_mnmf(mts_graphes, all_same_flags, args.comnum, args.nmf_max_it, args.nmf_tol,
                                                   args.mnmf_reg_lam)
            else:
                print("error combine flag: %d not in [1, 2, 3]")
                exit(-1)

    output_clusters(coms, graph_id_map)
    print_coms(coms, graph_id_map)

    return


def normalize_sims(sim_scores):
    max_sim = np.nanmax(sim_scores)
    min_sim = np.nanmin(sim_scores)

    if (max_sim - 0.0) > 1e-6:
        print('Error: has negative distance.')
        exit(-1)

    if max_sim == np.NaN:
        print('Error: all scores are NaN.')
        exit(-1)

    if (max_sim - min_sim) > 1e-6:
        sims_scores_norm = (sim_scores - min_sim) / (max_sim - min_sim)
    else:
        sims_scores_norm = np.where(sim_scores != np.NaN, 1.0, sim_scores)

    return sims_scores_norm


def main():
    mts_data = load_mts()
    sim_scores = np.load(args.input_sims)
    sim_scores_norm = normalize_sims(sim_scores)
    eks = np.loadtxt(args.eks)
    cluster_mts(mts_data, sim_scores_norm, eks)


if __name__ == "__main__":
    args = parse_args()
    main()
