import argparse
import logging
import os
import sys
from os.path import dirname

import sklearn
from sklearn.cluster import KMeans
from sklearn.datasets.base import load_data, load_files

from src import inf
from src.log_config import config_log

config_log()
logger = logging.getLogger(__name__)

__author__ = 'Stefan Morozov (kilokso86@gmail.com)'


def arg_parser():
    parser = argparse.ArgumentParser(description='Tool to calculate informative criteria described in '
                                                 '"A Bayesian information criterion for unsupervised learning based on '
                                                 'an objective prior"')
    parser.add_argument('-o', '--output', type=str, help='Output directory.')
    parser.add_argument('-d', '--dataset_dir', default='.\\', type=str,
                        help='Path to directory where .\\data\\dataset placed. By default ".\\"')
    parser.add_argument('-n', '--dataset_name', default='iris.csv', type=str,
                        help='Name of csv file to be loaded from. By default iris.csv (Fisher\'s Iris), '
                             'see "./data" to find more datasets')
    parser.add_argument('-v', '--verbose', action='store_true', help='Increase logging verbosity')
    return parser.parse_args()


if __name__ == "__main__":
    logger.info(f'Run: {" ".join(sys.argv)}')

    args = arg_parser()
    if args.verbose:
        logger.setLevel(logging.DEBUG)

    logger.info(f'Get dataset from: {os.path.join(os.path.abspath(args.dataset_dir), args.dataset_name)}...')
    X, Y, y_names = load_data(module_path=os.path.abspath(args.dataset_dir), data_file_name=args.dataset_name)
    logger.info(f'Data set was loaded')

    n = len(Y)
    #n = 20
    if n > 100:
        logger.info(f'Too big number {n}, be prepared for a long time...')

    entr_arr = inf.get_k_arr(n)
    logger.info(f'Calculated entropy for parts of number {n} - {entr_arr}')
