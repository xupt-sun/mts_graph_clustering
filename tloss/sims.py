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
import torch
import numpy
import argparse
from sktime.datasets import load_from_tsfile
import scikit_wrappers_sims


def load_dataset(path, dataset):
    """
    Loads the dataset given in input in numpy arrays.

    @param path Path where the dataset is located.
    @param dataset Name of the dataset.

    @return training set.
    """
    # Initialization needed to load a file with Weka wrappers
    file_name = os.path.join(path, dataset + ".ts")
    data, labels = load_from_tsfile(file_name, replace_missing_vals_with='NaN', return_data_type="nested_univ")

    train_size, nb_dims = data.shape
    length = 0
    for ix in range(train_size):
        tss = data.loc[ix]
        ts_len = len(tss[0])
        if length < ts_len:
            length = ts_len

    train = numpy.full((train_size, nb_dims, length), numpy.NaN)

    for ix in range(train_size):
        tss = data.loc[ix]
        ts_len = len(tss[0])
        for jx in range(nb_dims):
            ts_vals = tss[jx].values
            for kx in range(ts_len):
                train[ix, jx, kx] = ts_vals[kx]

    # Normalizing dimensions independently
    for j in range(nb_dims):
        mean_val = numpy.nanmean(train[:, j])
        std_val = numpy.nanstd(train[:, j])
        train[:, j] = (train[:, j] - mean_val) / std_val

    return train


def fit_hyperparameters(file, train, cuda, gpu, emb_dim,
                        save_memory=False):
    """
    Creates a embedder from the given set of hyperparameters in the input
    file, fits it and return it.

    @param file Path of a file containing a set of hyperparemeters.
    @param train Training set.
    @param cuda If True, enables computations on the GPU.
    @param gpu GPU to use if CUDA is enabled.
    @param save_memory If True, save GPU memory by propagating gradients after
           each loss term, instead of doing it after computing the whole loss.
    """
    embedder = scikit_wrappers_sims.CausalCNNEncoder()

    # Loads a given set of hyperparameters and fits a model with those
    hf = open(os.path.join(file), 'r')
    params = json.load(hf)
    hf.close()
    # Check the number of input channels
    params['in_channels'] = numpy.shape(train)[1]
    params['cuda'] = cuda
    params['gpu'] = gpu
    params['out_channels'] = emb_dim
    params['reduced_size'] = int(emb_dim / 2)
    embedder.set_params(**params)
    embedder.fit_encoder(train, save_memory=save_memory, verbose=True)

    return embedder


def parse_arguments():
    parser = argparse.ArgumentParser(
        description='Training modle for TS similarity calculation'
    )
    parser.add_argument('--dataset', type=str, metavar='D', required=True,
                        help='dataset name')
    parser.add_argument('--ts_type', type=str, required=True,
                        help='ts type: ALL, EACH')
    parser.add_argument('--path', type=str, metavar='PATH', required=True,
                        help='path where the dataset is located')
    parser.add_argument('--save_path', type=str, metavar='PATH', required=True,
                        help='path where the estimator is/should be saved')
    parser.add_argument('--cuda', action='store_true',
                        help='activate to use CUDA')
    parser.add_argument('--gpu', type=int, default=0, metavar='GPU',
                        help='index of GPU used for computations (default: 0)')
    parser.add_argument('--hyper', type=str, metavar='FILE', required=True,
                        help='path of the file of hyperparameters to use; ' +
                             'for training; must be a JSON file')
    parser.add_argument('--load', action='store_true', default=False,
                        help='activate to load the estimator instead of ' +
                             'training it')
    parser.add_argument('--fit_encoder', action='store_true', default=False,
                        help='if not supervised, activate to load the ' +
                             'model and retrain the embedder')
    parser.add_argument('--emb_dim', type=int, default=320,
                        help='embedding dimension')
    parser.add_argument('--run_id', type=int, default=0,
                        help='run identifier')

    return parser.parse_args()


def train_embedder(data, tag):
    embedder = fit_hyperparameters(args.hyper, data, args.cuda, args.gpu, args.emb_dim)

    prefix = args.dataset + '_' + tag + '_dim' + str(args.emb_dim) + '_' + str(args.run_id)
    embedder.save_encoder(os.path.join(args.save_path, prefix))

    prefix = args.dataset + '_' + tag + '_dim' + str(args.emb_dim) + '_' + str(args.run_id)\
             + '_hyperparameters.json'
    with open(os.path.join(args.save_path, prefix), 'w') as fp:
        json.dump(embedder.get_params(), fp)


def train_embedder_all(data):
    tag = args.ts_type
    train_embedder(data, tag)


def train_embedder_each(data):
    mts_num, attr_num, ts_len = data.shape

    for ix in range(attr_num):
        data_each = numpy.full((mts_num, 1, ts_len), numpy.NaN)
        data_each[:, 0, :] = data[:, ix, :]
        tag = args.ts_type + str(ix)
        train_embedder(data_each, tag)


if __name__ == '__main__':
    args = parse_arguments()

    # print('run_id: %d' % args.run_id)

    if args.cuda and not torch.cuda.is_available():
        print("CUDA is not available, proceeding without it...")
        args.cuda = False

    data = load_dataset(args.path, args.dataset)

    if args.ts_type == 'ALL':
        train_embedder_all(data)
    elif args.ts_type == 'EACH':
        train_embedder_each(data)
    else:
        print('Error ts_type: ALL or EACH')
        exit(-1)
