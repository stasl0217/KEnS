#!/usr/bin/env python
# coding: utf-8


import src.param as param
param.knowledge = 'rotate'

"""
In all variable names, 0 denotes the target/sparse kg and 1 denotes the support kg.
"""
import os
from os.path import join
import sys

if './' not in sys.path:
    sys.path.append('./')

import src.param as param
import numpy as np
import logging
import pandas as pd
import ast
import time
# from pandarallel import pandarallel
from src.validate import MultiModelTester, filt_hits_at_n, filt_ensemble_hits_at_n, hr2t_from_train_set
from src.ensemble import *
from src.data_loader import *
from src.weightlearning import *
import argparse



def set_logger(param, model_dir):
    '''
    Write logs to checkpoint and console
    '''

    log_file = os.path.join(model_dir, 'test-boost.log')

    logging.basicConfig(
        format='%(asctime)s %(levelname)-8s %(message)s',
        level=logging.INFO,
        datefmt='%Y-%m-%d %H:%M:%S',
        filename=log_file,
        filemode='w'
    )
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s %(levelname)-8s %(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)


def parse_args(args=None):
    parser = argparse.ArgumentParser(
        description='Training and Testing Knowledge Graph Embedding Models',
        usage='test.py [<args>] [-h | --help]'
    )
    parser.add_argument('-l', '--target_language', type=str, choices=['ja', 'el', 'es', 'fr', 'en'], help="target kg")
    parser.add_argument('-m', '--knowledge_model', default='transe', type=str, choices=['transe', 'rotate'])
    parser.add_argument('--model_dir', type=str, help="Directory of trained model")
    parser.add_argument('-d', '--dim', type=int, help="Dimension of the saved model")

    return parser.parse_args(args)


def main(args):
    target_lang = args.target_language
    param.knowledge = args.knowledge_model
    model_dir = args.model_dir

    param.lang = target_lang

    src_langs = ['fr', 'ja', 'es',  'el',  'en']
    src_langs.remove(target_lang)
    # load data
    data_dir = './data/kg'
    entity_dir = './data/entity'
    vocabs = pd.read_csv(join(entity_dir, f'{target_lang}.tsv'), sep='\t', header=None)
    num_entity = len(vocabs)
    hr2t_train = hr2t_from_train_set(data_dir, target_lang)  # used for filtered test setting

    # logging
    set_logger(param, model_dir)  # set logger.
    logging.info('=== Boosting')
    logging.info(f'target_lang: {target_lang}')

    langs = [target_lang] + src_langs


    # load predictions on validation set
    df = pd.read_csv('results-val.tsv', sep='\t')  # (h,r,t) => predictions
    for model in langs:  # convert string to list
        df[model] = df[model].apply(lambda x: ast.literal_eval(x))
        # extract entities
        df[model] = df[model].apply(lambda x: extract_entities(x))

    # load predictions on test set for testing
    test_result_file = join(model_dir,  'results-test.tsv')
    result = pd.read_csv(test_result_file, sep='\t')
    for lang in langs:
        result[lang] = result[lang].apply(lambda x: ast.literal_eval(x))  # convert list string to list
    print(result.head(2))

    # learn weights for all entities
    weight_df = pd.DataFrame(np.arange(num_entity), columns=['e'])
    base = {lang: filt_hits_at_n(result, lang, hr2t_train, n=10) for lang in langs}
    weight_df['weights'] = weight_df['e'].apply(lambda e: learn_entity_specific_weights(target_lang, langs, df, e, num_entity,base))

    wdf = weight_df.copy()
    weights = pd.Series(wdf['weights'].values, index=wdf['e']).to_dict() # {entity: {model:weight}}
    wdf.to_csv(join(model_dir, 'weights.tsv'), sep='\t', index=False)

    logging.info(f'[{target_lang}]')

    logging.info('=== boosting')
    filt_ensemble_hits_at_n(result, weights, langs, hr2t_train, n=10)
    filt_ensemble_hits_at_n(result, weights, langs, hr2t_train, n=3)
    filt_ensemble_hits_at_n(result, weights, langs, hr2t_train, n=1)


if __name__ == "__main__":
    main(parse_args())



