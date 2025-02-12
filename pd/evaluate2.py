'''
Evaluate clustering results.

'''

import argparse
from sklearn.metrics.cluster import rand_score
from sklearn.metrics.cluster import adjusted_rand_score
from sklearn.metrics.cluster import normalized_mutual_info_score
from sklearn.metrics.cluster import adjusted_mutual_info_score


def parse_args():
    '''
    Parses arguments.
    '''
    parser = argparse.ArgumentParser(description="Run Evaluate.")

    parser.add_argument('--clusters', nargs='?', required=True,
                        help='cluster file.')

    parser.add_argument('--labels', nargs='?', required=True,
                        help='label file')

    return parser.parse_args()


def load_labels(file_name):
    '''
        load real label for each mts.
    '''
    mts_label_ids = {}
    mts_label_names = {}
    label_clusters = {}

    in_file = open(file_name, 'r')

    ###
    for line in in_file:
        items = line.split()
        mts_id = items[0]
        label_id = items[1]

        if len(items) == 3:
            label_name = items[2]
        else:
            label_name = 'None'

        mts_label_ids[mts_id] = label_id
        mts_label_names[mts_id] = label_name

        if label_id in label_clusters.keys():
            mems = label_clusters[label_id]
            mems.add(mts_id)
        else:
            mems = set()
            mems.add(mts_id)
            label_clusters[label_id] = mems

    return mts_label_ids, label_clusters, mts_label_names


def load_clusters(file_name):
    '''
        load predicted label for each mts.
    '''
    mts_cluster_ids = {}
    clusters = {}

    in_file = open(file_name, 'r')

    ###
    for line in in_file:
        items = line.split()
        mts_id = items[0]
        cluster_id = items[1]

        mts_cluster_ids[mts_id] = cluster_id

        if cluster_id in clusters.keys():
            mems = clusters[cluster_id]
            mems.add(mts_id)
        else:
            mems = set()
            mems.add(mts_id)
            clusters[cluster_id] = mems

    return mts_cluster_ids, clusters


def evaluate_ri(mts_cluster_ids, mts_label_ids):
    # compute rand index
    mts_ids = list(mts_cluster_ids.keys())
    mts_ids.sort()
    clusters = []
    labels = []

    for id in mts_ids:
        cluster_id = mts_cluster_ids[id]
        label_id = mts_label_ids[id]
        clusters.append(cluster_id)
        labels.append(label_id)

    rand_index = rand_score(labels, clusters)
    rand_index_adjusted = adjusted_rand_score(labels, clusters)
    return rand_index, rand_index_adjusted


def evaluate_nim(mts_cluster_ids, mts_label_ids):
    # compute Normalized Mutual Information
    mts_ids = list(mts_cluster_ids.keys())
    mts_ids.sort()
    clusters = []
    labels = []

    for id in mts_ids:
        cluster_id = mts_cluster_ids[id]
        label_id = mts_label_ids[id]
        clusters.append(cluster_id)
        labels.append(label_id)

    nmi = normalized_mutual_info_score(labels, clusters, average_method="geometric")
    nmi_adjust = adjusted_mutual_info_score(labels, clusters, average_method="geometric")

    return nmi, nmi_adjust


def print_clusters(clusters):
    cids = list(clusters.keys())
    cids.sort()

    for id in cids:
        line = '%d: ' % id
        mems = clusters[id]

        for mid in mems:
            rmid = mems[mid]
            line += ' %d' % rmid

        print(line)

    return


def main(args):
    mts_cluster_ids, mts_clusters = load_clusters(args.clusters)
    mts_label_ids, label_clusters, mts_label_names = load_labels(args.labels)

    rand_index, rand_index_adjusted = evaluate_ri(mts_cluster_ids, mts_label_ids)
    nmi, nmi_adjusted = evaluate_nim(mts_cluster_ids, mts_label_ids)

    # print('RI \t ARI \t NMI \t ANMI')
    print('%.4f\t%.4f\t%.4f\t%.4f' % (rand_index, rand_index_adjusted, nmi, nmi_adjusted))


if __name__ == "__main__":
    args = parse_args()
    main(args)

