'''
MTS clustering.

'''

import argparse
import numpy as np
import networkx as nx
import multiTS as mts
import mtsClustering3 as mts_clust


def parse_args():
    '''
    Parses arguments.
    '''
    parser = argparse.ArgumentParser(description="Run find_ek.")

    parser.add_argument('--input', nargs='?', required=True,
                        help='Input file.')

    parser.add_argument('--sim_file', nargs='?', required=True,
                        help='Sim file.')
    
    parser.add_argument('--output', nargs='?', required=True,
                        help='Output file of ek.')

    parser.add_argument('--combine_flag', type=int, default=0,
                        help='Combine flag: 0 (ALL), 2(Average)')

    parser.add_argument('--k', type=int, required=True,
                        help='Given k.')
                        
    parser.add_argument('--max_min_comp', type=int, default=1,
                        help='Maximum of minimum component number.')

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

    return mts_data, attr_num



def test_e(sim_scores, cluster, k, min_comp):
    ### find e
    e_set = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0]
    e_max = 0.0
    num_node = 0
    num_edge = 0
    num_comp = 0
    mts_graph, all_same_flag, graph_id_map = cluster.build_mts_graph_knn_sim(sim_scores, k, 0.0)
    num_node = nx.number_of_nodes(mts_graph)
    num_edge_sim0 = nx.number_of_edges(mts_graph)
    
    for e in e_set:
        mts_graph, all_same_flag, graph_id_map = cluster.build_mts_graph_knn_sim(sim_scores, k, e)
        num_comp = nx.number_connected_components(mts_graph)
        num_edge_sime = nx.number_of_edges(mts_graph)

        if num_comp == min_comp:
            e_max = e
            break

    if num_comp != min_comp:
        print('Warning: min_comp=%d; k=%d' % (num_comp, k))    

    return num_node, k, num_edge_sim0, e_max, num_edge_sime, min_comp
        

def find_e(mts_data, sim_scores, combine_flag, k, max_min_comp):
    cluster = mts_clust.MtsClustering(mts_data)

    if combine_flag == 0:  # single layer network
        num_node, k, num_edge_sim0, e_max, num_edge_sime, min_comp = test_e(sim_scores, cluster, k, max_min_comp)
        eks = np.ndarray((1, 6))
        eks[0] = np.array([num_node, k, num_edge_sim0, e_max, num_edge_sime, min_comp])

        print('combine_flag: %d' % combine_flag)
        print('# num_node, k, num_edge_sim0, e_max, num_edge_sime, min_comp')
        print('%d\t%d\t%d\t%.1f\t%d\t%d' % (num_node, k, num_edge_sim0, e_max, num_edge_sime, min_comp))
    elif combine_flag == 2:  # multi layer network, combine by average
        ### normalize sims and then average them
        sims_scores_avg = np.mean(sim_scores, axis=0)

        ###
        num_node, k, num_edge_sim0, e_max, num_edge_sime, min_comp = test_e(sims_scores_avg, cluster, k, max_min_comp)
        eks = np.ndarray((1, 6))
        eks[0] = np.array([num_node, k, num_edge_sim0, e_max, num_edge_sime, min_comp])

        print('combine_flag: %d' % combine_flag)
        print('# num_node, k, num_edge_sim0, e_max, num_edge_sime, min_comp')
        print('%d\t%d\t%d\t%.1f\t%d\t%d' % (num_node, k, num_edge_sim0, e_max, num_edge_sime, min_comp))
    else:
        print('Error emb_type, should be: ALL or AVG')
        exit(-1)
        
    return eks


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
    mts_data, attr_num = load_mts()
    sim_scores = np.load(args.sim_file)
    #ixs = np.argwhere(np.isnan(sim_scores))
    #nan_num = np.count_nonzero(np.isnan(sim_scores))
    sim_scores_norm = normalize_sims(sim_scores)
    es = find_e(mts_data, sim_scores_norm, args.combine_flag, args.k, args.max_min_comp)
    np.save(args.output, es)

    return


if __name__ == "__main__":
    args = parse_args()
    main()
    
