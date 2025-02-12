# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at

#   http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.


import os
import json
import numpy
import torch
import argparse
import scipy.spatial.distance as spd
import sims
import scikit_wrappers_sims


def load_embedders(save_path, dataset, tag, emb_dim, cuda, gpu):
    """
    Loads and returns embedders from the given parameters.

    @param save_path Path where the model is located.
    @param dataset Name of the dataset.
    @param cuda If True, enables computations on the GPU.
    @param gpu GPU to use if CUDA is enabled.
    """
    embedder = scikit_wrappers_sims.CausalCNNEncoder()
    prefix = dataset + '_' + tag + '_dim' + str(emb_dim) + '_' + str(args.run_id)\
             + '_hyperparameters.json'
    hf = open(os.path.join(save_path, prefix), 'r')
    hp_dict = json.load(hf)
    hf.close()
    hp_dict['cuda'] = cuda
    hp_dict['gpu'] = gpu
    embedder.set_params(**hp_dict)
    prefix = dataset + '_' + tag + '_dim' + str(args.emb_dim) + '_' + str(args.run_id)
    embedder.load_encoder(os.path.join(save_path, prefix))

    return embedder


def parse_arguments():
    parser = argparse.ArgumentParser(
        description='Classification tests for UCR repository datasets, ' +
                    'using the features of several precomputed encoders, ' +
                    'possibly with different hyperparameters, and combining ' +
                    'their computed representations to train an SVM on top ' +
                    'of them.'
    )
    parser.add_argument('--dataset', type=str, metavar='D', required=True,
                        help='dataset name')
    parser.add_argument('--ts_type', type=str, required=True,
                        help='ts type: ALL, EACH')
    parser.add_argument('--dist_func', type=str, required=True,
                        help='dist func: euclidean, cosine')
    parser.add_argument('--sim_file', type=str, required=True,
                        help='similarity file')
    parser.add_argument('--path', type=str, metavar='PATH', required=True,
                        help='path where the dataset is located')
    parser.add_argument('--model_path', type=str, metavar='PATH',
                        required=True,
                        help='path where the folders containing models for ' +
                             'different hyperparameters are located')
    parser.add_argument('--save_path', type=str, metavar='PATH', required=True,
                        help='path where the embedder is/should be saved')
    parser.add_argument('--cuda', action='store_true',
                        help='activate to use CUDA')
    parser.add_argument('--gpu', type=int, default=0, metavar='GPU',
                        help='index of GPU used for computations (default: 0)')
    parser.add_argument('--emb_dim', type=int, default=320,
                        help='embedding dimension')
    parser.add_argument('--run_id', type=int, default=0,
                        help='run identifier')

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_arguments()
    if args.cuda and not torch.cuda.is_available():
        print("CUDA is not available, proceeding without it...")
        args.cuda = False

    # Train datasets
    data = sims.load_dataset(args.path, args.dataset)

    if args.ts_type == 'ALL':
        embedder = load_embedders(args.model_path, args.dataset,
                                  args.ts_type, args.emb_dim, args.cuda, args.gpu)
        reprs = embedder.encode(data)
        ### compute similarities
        mts_num = data.shape[0]
        sims = numpy.zeros((mts_num, mts_num))

        for ix in range(mts_num-1):
            for jx in range(ix+1, mts_num):
                repr_ix = reprs[ix, :]
                repr_jx = reprs[jx, :]
                if args.dist_func == 'euclidean':
                    score = -spd.euclidean(repr_ix, repr_jx)
                elif args.dist_func == 'cosine':
                    score = -spd.cosine(repr_ix, repr_jx)
                else:
                    print('Unknow distance function')
                    exit(-1)
                sims[ix, jx] = score
                sims[jx, ix] = score
    elif args.ts_type == 'EACH':
        mts0 = data[0]
        attr_num, ts_len = mts0.shape
        mts_num = data.shape[0]
        sims = numpy.zeros((attr_num, mts_num, mts_num))
        data_each = numpy.full((mts_num, 1, ts_len), numpy.NaN)

        for kx in range(attr_num):
            data_each[:, 0, :] = data[:, kx, :]
            tag = args.ts_type + str(kx)
            embedder = load_embedders(args.model_path, args.dataset,
                                      tag, args.emb_dim, args.cuda, args.gpu)
            reprs = embedder.encode(data_each)

            for ix in range(mts_num - 1):
                for jx in range(ix + 1, mts_num):
                    repr_ix = reprs[ix, :]
                    repr_jx = reprs[jx, :]
                    if args.dist_func == 'euclidean':
                        score = -spd.euclidean(repr_ix, repr_jx)
                    elif args.dist_func == 'cosine':
                        score = -spd.cosine(repr_ix, repr_jx)
                    else:
                        print('Unknow distance function')
                        exit(-1)
                    sims[kx, ix, jx] = score
                    sims[kx, jx, ix] = score
    else:
        print('Error ts_type: ALL or EACH')
        exit(-1)

    numpy.save(args.sim_file, sims)
