'''
Class MtsClustering
'''


import numpy as np
import scipy.spatial.distance as spd
from sklearn.decomposition import NMF
from sklearn.utils import check_random_state
from ordpy import complexity_entropy, tsallis_complexity_entropy, renyi_complexity_entropy
from operator import itemgetter
import networkx as nx
import networkx.algorithms.community as nx_comm
import matplotlib.pyplot as plt
import concurrent.futures


class MtsClustering:
    # MTS clustering using complex network
    def __init__(self, mts_objs):
        self.mts_objs = mts_objs

    def build_mts_graph_multilayer_knn(self, mts_sims, knn):
        ids = list(self.mts_objs.keys())
        ids.sort()
        id_num = len(ids)
        layer_num = mts_sims.shape[0]
        mts_graphes = {}
        graph_id_map = {}

        ###
        for nid in range(id_num):
            graph_id_map[nid] = ids[nid]

        ### create as many graphes as layer number
        for gid in range(layer_num):
            mts_graphes[gid] = nx.Graph()

        ### add nodes
        for gid in range(layer_num):
            graph = mts_graphes[gid]
            for nid in range(id_num):
                graph.add_node(nid)
                graph.nodes[nid]["label"] = ids[nid]

        ### add edges
        for gid in range(layer_num):
            graph = mts_graphes[gid]
            graph_sims = mts_sims[gid]

            for nid in range(id_num):
                sims = np.array(graph_sims[nid, :])
                sorted_ix = sims.argsort()
                sorted_ix = sorted_ix[::-1]     # usually, the first is self. its sim = 0.0

                # add the first K connections
                add_num = 0
                for kx in range(id_num):
                    if kx == nid:
                        continue    # skip self
                    nbid = sorted_ix[kx]
                    graph.add_edge(nid, nbid)
                    add_num += 1

                    if add_num >= knn:
                        break

            ### show the graph
            # self.show_graph0(graph)

        return mts_graphes, graph_id_map

    def build_mts_graph_multilayer_knn_sim(self, mts_sims, knn, sim_th):
        ids = list(self.mts_objs.keys())
        ids.sort()
        id_num = len(ids)
        layer_num = mts_sims.shape[0]
        mts_graphes = {}
        graph_id_map = {}

        ###
        for nid in range(id_num):
            graph_id_map[nid] = ids[nid]

        ### create as many graphes as layer number
        for gid in range(layer_num):
            mts_graphes[gid] = nx.Graph()

        ### add nodes
        for gid in range(layer_num):
            graph = mts_graphes[gid]
            for nid in range(id_num):
                graph.add_node(nid)
                graph.nodes[nid]["label"] = ids[nid]

        ### add edges
        max_sim = 0.0
        min_sim = np.amin(mts_sims)

        for gid in range(layer_num):
            graph = mts_graphes[gid]
            graph_sims = mts_sims[gid]

            for nid in range(id_num):
                sims = np.array(graph_sims[nid, :])

                # normalize
                sims_norm = sims
                if max_sim != min_sim:
                    sims_norm = (sims - min_sim) / (max_sim - min_sim)
                else:
                    sims_norm[:] = 1.0

                sorted_ix = sims_norm.argsort()  # usually, the first is self. its sim = 1.0
                sorted_ix = sorted_ix[::-1]

                if sorted_ix[0] != nid:  # in case there are same mtses as nid
                    for ix in range(1, id_num):
                        if sorted_ix[ix] == nid:
                            sorted_ix[ix] = sorted_ix[0]
                            sorted_ix[0] = nid

                # add the first K connections
                nbid = sorted_ix[1]  # skip self
                graph.add_edge(nid, nbid)
                add_num = 1

                ### add other connections if possible
                for ix in range(2, len(sims_norm)):
                    nbid = sorted_ix[ix]
                    score = sims_norm[nbid]
                    if score >= sim_th:
                        graph.add_edge(nid, nbid)
                        add_num += 1
                        if add_num >= knn:
                            break
                    else:
                        break

            ### show the graph
            # self.show_graph0(graph)

        return mts_graphes, graph_id_map

    def combine_multilayer_graphes_overlay(self, mts_graph_multilayer):
        ids = list(self.mts_objs.keys())
        ids.sort()
        id_num = len(ids)
        ids_layer = list(mts_graph_multilayer.keys())
        ids_layer.sort()
        layer_num = len(ids_layer)
        graph_combined = nx.Graph()

        ### add nodes
        for nid in range(id_num):
            graph_combined.add_node(nid)
            graph_combined.nodes[nid]["label"] = ids[nid]

        ### add edges
        for nid1 in range(id_num-1):
            for nid2 in range(nid1+1, id_num):
                con_num = 0

                for gid in range(layer_num):
                    graph = mts_graph_multilayer[gid]
                    if graph.has_edge(nid1, nid2):
                        con_num += 1

                if con_num >= 1:
                    graph_combined.add_edge(nid1, nid2, weight=con_num)

        ### show the graph
        # self.show_graph0(graph_combined)

        return graph_combined

    def combine_multilayer_graphes_avg(self, mts_sims, knn, sim_th):
        ### average similarity score of each attribute.
        ids = list(self.mts_objs.keys())
        ids.sort()
        id_num = len(ids)
        layer_num = mts_sims.shape[0]
        graph_combined = nx.Graph()
        graph_id_map = {}
        topk_neighbors = {}

        ### node id and name map
        for nid in range(id_num):
            graph_id_map[nid] = ids[nid]

        ### add nodes
        for nid in range(id_num):
            graph_combined.add_node(nid)
            graph_combined.nodes[nid]["label"] = ids[nid]

        ### add edges
        #### compute average similarity score
        avg_sims = np.mean(mts_sims, axis=0)

        #### add connections
        for nid in range(id_num):
            sims_score = avg_sims[nid, :]
            scores = []
            for ix in range(id_num):
                if ix == nid:
                    continue
                scores.append((sims_score[ix], None, ix))

            scores_sorted = sorted(scores, key=itemgetter(0), reverse=True)
            topk_neighbors[nid] = scores_sorted[0:knn]

            ####
            scores_conn = []
            scores_conn.append(scores_sorted[0])    ### add the most similar

            for ix in range(1, len(scores_sorted)):
                sim_score = scores_sorted[ix][0]
                if sim_score >= sim_th:
                    scores_conn.append(scores_sorted[ix])

            if len(scores_conn) == 0:
                continue

            add_num = 0

            for ix in range(0, len(scores_conn)):
                nnid = scores_conn[ix][2]
                if not graph_combined.has_edge(nid, nnid):
                    graph_combined.add_edge(nid, nnid)
                add_num += 1
                if add_num >= knn:
                    break

        # show the graph
        # MtsClustering.show_graph0(graph_combined)

        return graph_combined, graph_id_map

    def cluster_mtsset(self, mtsset_graph, alg, comnum, nmf_max_it, nmf_tol):
        mtsset_clusters = {}

        if alg == "louvain":
            # need python 3.0 or above
            louvain_coms = nx_comm.louvain_communities(mtsset_graph, weight='weight', resolution=1.0)
            # louvain_coms = nx_comm.louvain_communities(mtsset_graph, resolution=1.0)
            louvain_com_num = len(louvain_coms)

            for ix in range(louvain_com_num):
                com = louvain_coms[ix]
                com_list = list(com)
                com_list.sort()
                mtsset_clusters[ix] = com_list

        elif alg == "nmf":
            # ajd_mat = np.array(nx.adjacency_matrix(mtsset_graph).todense())
            ajd_mat = nx.to_numpy_array(mtsset_graph)
            # W, H = non_negative_factorization(ajd_mat, n_components=comnum, init='random')
            model = NMF(n_components=comnum, init='random', tol=float(nmf_tol), max_iter=nmf_max_it)
            W = model.fit_transform(ajd_mat)
            node_num, col = W.shape

            for nid in range(node_num):
                prob_com = np.array(W[nid, :])
                com_id = np.argmax(prob_com)

                if com_id not in mtsset_clusters.keys():
                    mtsset_clusters[com_id] = [nid]
                else:
                    mems = mtsset_clusters[com_id]
                    mems.append(nid)
        else:
            print('Error community detection algorithm.')
            exit(-1)

        return mtsset_clusters

    def cluster_mtsset_mnmf(self, mts_graph_multilayer, com_num, mnmf_max_it, mnmf_tol, reg_lambda):
        max_iter = 200
        graph_ids = list(mts_graph_multilayer.keys())
        graph_ids.sort()
        graph_num = len(graph_ids)
        node_num = mts_graph_multilayer[0].number_of_nodes()

        ### build graph matrices
        graph_matrices = np.ndarray((graph_num, node_num, node_num))

        for gid in graph_ids:
            graph = mts_graph_multilayer[gid]
            ajd_mat = nx.to_numpy_array(graph)
            graph_matrices[gid, :, :] = ajd_mat

        ### initialization
        avg = np.sqrt(graph_matrices.mean() / com_num)
        rng = check_random_state(None)

        P = avg * rng.standard_normal(size=(node_num, com_num)).astype(graph_matrices.dtype, copy=False)
        np.abs(P, out=P)

        Q = {}
        for jx in range(graph_num):
            Q[jx] = avg * rng.standard_normal(size=(node_num, com_num)).astype(graph_matrices.dtype, copy=False)
            np.abs(Q[jx], out=Q[jx])

        ### updating
        converged_flag = False
        prev_j = 1e6
        for it in range(mnmf_max_it):
            #### update Qjx
            for jx in range(graph_num):
                Aj = graph_matrices[jx]
                Qj = Q[jx]
                mat_num = np.dot(Aj.T, P)                         # numerator
                mat_den = np.dot(np.dot(Qj, P.T), P) + np.dot(reg_lambda, Qj)   # denominator
                for k in range(node_num):
                    for l in range(com_num):
                        Qj[k, l] = Qj[k, l] * mat_num[k, l] / mat_den[k, l]

            #### update P
            A0 = graph_matrices[0]
            Q0 = Q[0]
            mat_num = np.dot(A0, Q0)
            for jx in range(1, graph_num):
                Aj = graph_matrices[jx]
                Qj = Q[jx]
                mat_num += np.dot(Aj, Qj)

            mat_den = np.dot(reg_lambda, P)
            for jx in range(0, graph_num):
                Qj = Q[jx]
                mat_den += np.dot(np.dot(P, Qj.T), Qj)

            for i in range(node_num):
                for k in range(com_num):
                    P[i ,k] = P[i ,k] * mat_num[i, k] / mat_den[i, k]

            #### compute loss
            term1 = 0.0
            for jx in range(graph_num):
                Aj = graph_matrices[jx]
                Qj = Q[jx]
                mat = Aj - np.dot(P, Qj.T)
                term1 += np.linalg.norm(mat)

            term2 = np.linalg.norm(P)
            for jx in range(graph_num):
                Qj = Q[jx]
                term2 += np.linalg.norm(Qj)
            term2 = term2 * reg_lambda

            J = 0.5 * (term1 + term2)

            #### check obj
            delta_j = prev_j - J
            if delta_j <= float(mnmf_tol):
                converged_flag = True
                print('MNMF converged: it=%d, loss=%.2f' % (it, J))
                break
            else:
                prev_j = J
                # print('MNMF converging: it=%d, loss=%.2f' % (it, J))

        ### extract communities
        if converged_flag == False:
            print("MNMF doesn't converge.")
            exit(-1)

        mtsset_clusters = {}

        for nid in range(node_num):
            prob_com = np.array(P[nid, :])
            com_id = np.argmax(prob_com)

            if com_id not in mtsset_clusters.keys():
                mtsset_clusters[com_id] = [nid]
            else:
                mems = mtsset_clusters[com_id]
                mems.append(nid)

        return mtsset_clusters

    def post_mts_clusters(self, cluster_mtsset, topk_neighbors):
        ###
        com_keys = list(cluster_mtsset.keys())
        com_keys.sort()
        com_num = len(com_keys)
        ids = list(self.mts_objs.keys())
        id_num = len(ids)
        mts_cid = -1 * np.ones((id_num,), dtype=int)

        for cid in com_keys:
            com_mems = cluster_mtsset[cid]
            for mem in com_mems:
                mts_cid[mem] = cid

        ###
        for cid in com_keys:
            com_mems = cluster_mtsset[cid]

            if len(com_mems) == 1:     # singleton
                nid = com_mems[0]
                #### add to the community of its most similar neighbor (NOT singleton) belongs to.
                add_flag = False
                topk = topk_neighbors[nid]
                for score in topk:
                    nnid = score[2]
                    nnid_cid = mts_cid[nnid]
                    mems_nnid_cid = cluster_mtsset[nnid_cid]
                    if len(mems_nnid_cid) > 1:
                        mts_cid[nid] = nnid_cid
                        add_flag = True
                        break

                if add_flag is False:
                    print('rectify community for mts %d failure.' % nid)

        ### create new community
        unique_mts_cid = list(set(list(mts_cid)))
        unique_mts_cid.sort()
        com_num_new = len(unique_mts_cid)
        coms_new = {}

        for cid in range(com_num_new):
            coms_new[cid] = []

        for ix in range(len(mts_cid)):
            nid = ix
            cid = mts_cid[ix]
            cid_new = unique_mts_cid.index(cid)
            mems = coms_new[cid_new]
            mems.append(nid)

        return coms_new

    @staticmethod
    def show_graph0(graph):
        plt.close()
        plt.figure(figsize=(14, 10))
        pos = nx.spring_layout(graph)
        node_ids = nx.get_node_attributes(graph, "label")
        # weights = nx.get_edge_attributes(graph, "weight")
        nx.draw_networkx_nodes(graph, pos, node_color="r", node_size=300)
        nx.draw_networkx_edges(graph, pos, connectionstyle='arc3,rad = 0.2', arrowsize=15)
        nx.draw_networkx_labels(graph, pos, labels=node_ids)
        # nx.draw_networkx_edge_labels(graph, pos, edge_labels=weights)
        # nx.draw_networkx_edge_labels(graph, pos)
        plt.show()
