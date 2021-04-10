#!/usr/bin/env python
# coding: utf-8

# In[2]:


"""
Working directory: project root

In all variable names, 0 denotes the target/sparse kg and 1 denotes the source/dense kg.
"""
import src.param as param
param.knowledge = 'rotate'


# In[3]:


import os
from os.path import join

print('Current working dir', os.getcwd())
import sys

if './' not in sys.path:
    sys.path.append('./')

from numpy.linalg import norm
import pandas as pd
import src.param as param


from src.model import save_model_structure, create_alignment_kNN_finder
from src.data_loader import load_support_kgs, load_target_kg, load_align_link_table, take_seed_align_links,     load_all_to_all_seed_align_links
from src.validate import MultiModelTester, TestMode, test_alignment_hits10
from src.modellist import AlignModel
import numpy as np
import logging
from src.validate import filt_hits_at_n, filt_ensemble_hits_at_n
import ast
from collections import defaultdict
from scipy.spatial.distance import cosine as cos_dist
from src.model import extend_seed_align_links


# In[4]:


def set_logger(param, model_dir):
    '''
    Write logs to checkpoint and console
    '''

    log_file = os.path.join(model_dir, 'train.log')

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



param.lang = 'ja'


param.epoch10 = 100
param.epoch11 = 100
param.epoch2 = 5
param.lr = 0.01
param.dim = 1000
param.round = 5
param.align_lr = 1e-4
# param.verbose = args.verbose
param.align_model = AlignModel('same')
param.updating_embedding = False
param.margin = 0.3

param.val_freq = 1


target_lang = param.lang
src_langs = ['fr', 'en', 'es',  'ja',  'el']
src_langs.remove(target_lang)



# load data
data_dir = './data/kg'  # where you put kg data
seed_dir = './data/seed_alignlinks'  # where you put seed align links data
model_dir = join('./model', f'kens-{param.knowledge}', target_lang)  # output
if not os.path.exists(model_dir):
    os.makedirs(model_dir)

    
# logging
set_logger(param, model_dir)  # set logger
logging.info('Knowledge model: %s'%(param.knowledge))
logging.info('target language: %s'%(param.lang))
             
# hyper-parameters
logging.info(f'dim: {param.dim}')
logging.info(f'lr: {param.lr}')
logging.info('margin: %.2f'%param.margin)


# target (sparse) kg
kg0 = load_target_kg(data_dir, target_lang, testfile_suffix='-val.tsv')  # target kg. KnowledgeGraph object
# supporter kgs
supporter_kgs = load_support_kgs(data_dir, seed_dir, target_lang, src_langs)  # list[KnowledgeGraph]

# seed alignment links
seed_alignlinks = load_all_to_all_seed_align_links(seed_dir)  # {(lang1, lang2): 2-col np.array}

all_kgs = [kg0] + supporter_kgs

# build alignment model (all-to-all)
for kg in all_kgs:
    kg.build_alignment_models(all_kgs)  # kg.align_models_of_all {lang: align_model}

# create validator
validator = MultiModelTester(kg0, supporter_kgs)

print('model initialization done')


for i in range(param.round):
    
    # train alignment model
    for kg in all_kgs:
        # align it with everything else
        for other_lang, align_model in kg.align_models_of_all.items():
            if (other_lang, kg.lang) in seed_alignlinks:  # seed_alignlinks {(lang1, lang2): 2-col np.array}
                # use if to avoid retrain the same pair of languages
                align_links = seed_alignlinks[(other_lang, kg.lang)]
                align_model.fit([align_links[:, 0], align_links[:, 1]], np.zeros(align_links.shape[0]),
                                epochs=param.epoch2, batch_size=param.batch_size, shuffle=True, )

    # # # self-learning
    # for kg in all_kgs:
    #     for other_kg in all_kgs:
    #         if other_kg.lang != kg.lang and (other_kg.lang, kg.lang) in seed_alignlinks:
    #             print(f'self learning[{kg.lang}][{other_kg.lang}]')
    #             seeds = seed_alignlinks[(other_kg.lang, kg.lang)]
    #             found = extend_seed_align_links(other_kg, kg, seeds)
    #             if len(found) > 0:  # not []
    #                 new_seeds = np.concatenate([seeds, found], axis=0)
    #                 seed_alignlinks[(other_kg.lang, kg.lang)] = new_seeds

    # train knowledge model
    kg0.model.fit([kg0.h_train, kg0.r_train, kg0.t_train], kg0.y_train,
                  epochs=param.epoch10, batch_size=param.batch_size, shuffle=True, )
    for kg1 in supporter_kgs:
        kg1.model.fit([kg1.h_train, kg1.r_train, kg1.t_train], kg1.y_train,
                      epochs=param.epoch11, batch_size=param.batch_size, shuffle=True, )

    model_weights = []
    if i % param.val_freq == 0:  # validation
        param.updating_embedding = False

        logging.info(f'=== round {i}')
        logging.info(f'[{kg0.lang}]')
        model_weights = []
        hits10_kg0 = validator.test(TestMode.KG0)
        model_weights.append(hits10_kg0)

        for kg1 in supporter_kgs:
            logging.info(f'[{kg1.lang}]')
            hits10 = validator.test(TestMode.KG1, supporter_kg=kg1)
            model_weights.append(hits10)
        print(model_weights)


kg0.save_model(model_dir)
for kg1 in supporter_kgs:
    kg1.save_model(model_dir)

save_model_structure(kg0.kNN_finder, os.path.join(model_dir, 'kNN_finder.json'))

# Populate the alignment set using newly predicted links
for kg1 in supporter_kgs:
    kg1.populate_alignlinks(kg0, seed_alignlinks)

choices = ['-val.tsv', '-test.tsv']  # '-val.tsv' if predict on validation data, '-test.tsv' if on test data
validator = MultiModelTester(kg0, supporter_kgs)

kg0.filtered_reordered_embedding_matrix = None
for kg1 in supporter_kgs:
    kg1.filtered_reordered_embedding_matrix = None

val_df = None
test_df = None

for suffix in choices:
    testfile = join(data_dir, target_lang + suffix)
    output = join(model_dir, 'results'+suffix)
    testcases = pd.read_csv(testfile, sep='\t', header=None).values
    param.n_test = testcases.shape[0]  # test on all training triples
    print('Loaded test cases from: %s'%testfile)

    results_df = validator.test_and_record_results(testcases)
    if suffix == '-val.tsv':
        val_df = results_df
    elif suffix == '-test.tsv':
        test_df = results_df

    results_df.to_csv(join(output), sep='\t', index=False)

