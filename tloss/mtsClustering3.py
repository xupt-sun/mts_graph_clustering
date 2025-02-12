'''
Class MtsClustering
'''


import numpy as np
from sklearn.decomposition import NMF
from sklearn.utils import check_random_state
from operator import itemgetter
import networkx as nx
import networkx.algorithms.community as nx_comm
import igraph as ig
import matplotlib.pyplot as plt


class MtsClustering:
    # MTS clustering using complex network
    def __init__(self, mts_objs):
        self.mts_objs = mts_objs

    def build_mts_graph_knn_sim(self, mts_sims,  knn, sim_th):
        ids = list(self.mts_objs.keys())
        ids.sort()
        id_num = len(ids)
        mts_graph = nx.Graph()
        graph_id_map = {}

        ### node id and name map
        for nid in range(id_num):
            graph_id_map[nid] = ids[nid]

        ### add nodes
        for nid in range(id_num):
            mts_graph.add_node(nid)
            mts_graph.nodes[nid]["label"] = ids[nid]

        ### add edges
        max_sim = np.nanmax(mts_sims)
        min_sim = np.nanmin(mts_sims)

        if max_sim == np.NaN:
            ### min_sim is also NaN.
            print('Warning: all scores are Nan. all_sam_flag is 1!')
            all_same_flag = 1  # whether all nodes are equally similar: 0, not; 1, yes.
            ### return a graph with no edge.
        elif (max_sim - min_sim) < 1e-6:
            print('Warning: all socres are equal (except NaN), %.6f. all_sam_flag is 1!' % max_sim)
            all_same_flag = 1   # whether all nodes are equally similar: 0, not; 1, yes.
            ### add knn edges randomly
            for nid in range(id_num):
                sims = np.array(mts_sims[nid, :])
                valid_sim_ixs = []
                for ix in range(len(sims)):
                    if ix == nid:
                        continue
                    if sims[ix] != np.NaN:
                        valid_sim_ixs.append(ix)
                ### randomly add k neighbors
                valid_sim_ixs_perm = np.random.permutation(np.array(valid_sim_ixs))
                valid_sim_ixs_perm = list(valid_sim_ixs_perm)
                for ix in range(knn):
                    if ix < len(valid_sim_ixs_perm):
                        nnid = valid_sim_ixs_perm[ix]
                        if not mts_graph.has_edge(nid, nnid):
                            mts_graph.add_edge(nid, nnid)
        else:
            all_same_flag = 0  # whether all nodes are equally similar: 0, not; 1, yes.
            ### add top similar knn edges
            for nid in range(id_num):
                sims = np.array(mts_sims[nid, :])
                max_sim = np.nanmax(sims)
                min_sim = np.nanmin(sims)

                if max_sim == np.NaN:
                    ### min_sim must be also NaN.
                    print('Warning: node %d has Nan max_sim' % nid)
                    topk_neighbors = []
                elif (max_sim - min_sim) < 1e-6:
                    ### randomly choose k neighbors
                    valid_sim_ixs = []
                    for ix in range(len(sims)):
                        if ix == nid:
                            continue
                        if sims[ix] != np.NaN:
                            valid_sim_ixs.append(ix)

                    valid_sim_ixs_perm = np.random.permutation(np.array(valid_sim_ixs))
                    valid_sim_ixs_perm = list(valid_sim_ixs_perm)
                    topk_neighbors = []
                    for ix in range(knn):
                        if ix < len(valid_sim_ixs_perm):
                            nnid = valid_sim_ixs_perm[ix]
                            topk_neighbors.append((1.0, None, nnid))
                else:
                    sims_norm = (sims - min_sim) / (max_sim - min_sim)
                    ### collect candidate scores
                    scores = []
                    for ix in range(id_num):
                        if ix == nid:   # skip self
                            continue
                        if sims_norm[ix] != np.NaN:
                            scores.append((sims_norm[ix], None, ix))

                    ### find top similar knn neighbors
                    scores_sorted = sorted(scores, key=itemgetter(0), reverse=True)
                    topk_neighbors = []
                    for ix in range(knn):
                        if ix < len(scores_sorted):
                            topk_neighbors.append(scores_sorted[ix])

                #### add edges
                if len(topk_neighbors) == 0:
                    continue

                scores_conn = []
                scores_conn.append(topk_neighbors[0])  ### add the most similar one. at least 1 connection with others.

                for ix in range(1, len(topk_neighbors)):
                    score = topk_neighbors[ix][0]
                    if score >= sim_th:
                        scores_conn.append(topk_neighbors[ix])

                for ix in range(len(scores_conn)):
                    nnid = scores_conn[ix][2]
                    if not mts_graph.has_edge(nid, nnid):
                        mts_graph.add_edge(nid, nnid)

        ### show the graph
        # self.show_graph0(mts_graph)

        return mts_graph, all_same_flag, graph_id_map

    def build_mts_graph_multilayer_knn_sim(self, mts_sims, knns, sim_ths):
        ids = list(self.mts_objs.keys())
        ids.sort()
        layer_num = mts_sims.shape[0]
        mts_graphes = {}
        all_same_flags = {}
        graph_id_map = {}

        ### create each layer graph
        for gid in range(layer_num):
            mts_graphes[gid], all_same_flags[gid], graph_id_map = (
                self.build_mts_graph_knn_sim(mts_sims[gid], knns[gid], sim_ths[gid]))

        return mts_graphes, all_same_flags, graph_id_map

    def combine_multilayer_graphes_overlay(self, mts_graph_multilayer, all_same_flags, min_layer_conns):
        ids = list(self.mts_objs.keys())
        ids.sort()
        id_num = len(ids)
        ids_layer = list(mts_graph_multilayer.keys())
        ids_layer.sort()
        layer_num = len(ids_layer)
        graph_combined = nx.Graph()
        graph_id_map = {}

        ### node id and name map
        for nid in range(id_num):
            graph_id_map[nid] = ids[nid]

        ### add nodes
        for nid in range(id_num):
            graph_combined.add_node(nid)
            graph_combined.nodes[nid]["label"] = ids[nid]

        ### count layer connection number
        layer_conn_num = np.zeros((id_num, id_num))

        for nid1 in range(id_num-1):
            for nid2 in range(nid1+1, id_num):
                con_num = 0
                for gid in range(layer_num):
                    if all_same_flags[gid] == 1:
                        continue    # skip layer that can not differentiate MTS instances.

                    graph = mts_graph_multilayer[gid]

                    if graph.has_edge(nid1, nid2):
                        con_num += 1
                layer_conn_num[nid1, nid2] = con_num
                layer_conn_num[nid2, nid1] = con_num

        ### add edges
        for nid1 in range(id_num - 1):
            for nid2 in range(nid1 + 1, id_num):
                con_num = int(layer_conn_num[nid1,nid2])
                if con_num >= min_layer_conns:
                    graph_combined.add_edge(nid1, nid2, weight=con_num)

        ### show the graph
        # self.show_graph0(graph_combined)

        return graph_combined, graph_id_map

    def combine_multilayer_graphes_avg(self, mts_sims, knn, sim_th):
        mts_sims_avg = np.mean(mts_sims, axis=0)

        ### build combined graphs
        graph_combined, all_same_flag, graph_id_map = self.build_mts_graph_knn_sim(mts_sims_avg, knn, sim_th)

        return graph_combined, all_same_flag, graph_id_map

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
        elif alg == "flpa":
            ig_graph = ig.Graph.from_networkx(mtsset_graph)
            vertex_clusters = ig.Graph.community_label_propagation(ig_graph, variant='fast')
            # vertex_clusters = ig.Graph.community_fast_label_propagation(ig_graph)

            vertex_membership = vertex_clusters.membership

            for ix in range(len(vertex_membership)):
                com_id = vertex_membership[ix]

                if com_id not in mtsset_clusters.keys():
                    mtsset_clusters[com_id] = [ix]
                else:
                    mems = mtsset_clusters[com_id]
                    mems.append(ix)
        else:
            print('Error community detection algorithm.')
            exit(-1)

        return mtsset_clusters

    def cluster_mtsset_mnmf(self, mts_graph_multilayer, all_same_flags, com_num, mnmf_max_it, mnmf_tol, reg_lambda):
        graph_ids = list(mts_graph_multilayer.keys())
        graph_ids.sort()
        graph_num = len(graph_ids)
        node_num = mts_graph_multilayer[0].number_of_nodes()

        ### build graph matrices
        graph_matrices = np.ndarray((graph_num, node_num, node_num))
        ix = 0

        for gid in graph_ids:
            if all_same_flags[gid] == 1:
                continue
            graph = mts_graph_multilayer[gid]
            ajd_mat = nx.to_numpy_array(graph)
            graph_matrices[ix, :, :] = ajd_mat
            ix += 1

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
        
