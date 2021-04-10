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
from pandarallel import pandarallel
from src.validate import MultiModelTester, filt_hits_at_n, filt_ensemble_hits_at_n, hits10_with_weight, hits3_with_weight, hits1_with_weight
from src.ensemble import *
from src.data_loader import *


#%%%%%%%%%%%%%%%%%%%%%%%%%%%%

def weight_matrix_from_one_triple(row):
    # weight_matrix[i] is the datapoint weight for the entity pair (true_t, i)
    weight_matrix = np.ones(num_entity)
    return weight_matrix


def rank_matrix_from_one_triple(row):
    """
    rank_matrix shape: [num_entity, len(langs)]
    rank_matrix[i,j]=1 means entity[i] is correctly ranked after true_t by lang j, -1 means wrongly
    rank_matrix[true_t, true_t] = 1
    :param row: a row in df
    :return:
    """
    true_t = row['t']

    rank_matrix = 1 * np.ones(
        [num_entity, len(langs)])  # all initialized as -1 (ranked wrong by default) excpet for true_t

    for j, lang in enumerate(langs):
        predictions = row[lang]

        ths = int(100 / mrrs[target_lang][lang])
        if true_t in predictions:
            # unseen datapoints as ranked correctly
            rank_matrix[:, j] = 1
            rank = predictions.index(true_t)  # start from 0
            if rank > 0:
                # update rank_matrix for entities before true_t (they are ranked wrongly)
                rank_matrix[np.array(predictions[:rank + 1]), j] = -1
        else:
            # assume it's ranked around mean rank
            rank_matrix[np.array(predictions).astype(np.int32), j] = -1  # entities in predictions are wrongly ranked
            # sample the rest
            rest = np.random.randint(0, num_entity, size=(ths - len(predictions)))
            rank_matrix[rest, j] = -1
    return rank_matrix


# In[ ]:


def triples_of_given_entity(df, e):
    return df[(df['h'] == e) | (df['t'] == e)]


#     return df[(df['h'] == e)]


def model_weights_for_one_round(weight_matrix, rank_matrix):
    # learn model weights
    # wrong[i,j] = -a means language j wrongly ranks entity i before the true_t,
    # and the pair (true_t, i) has weight a
    wrong = np.zeros(rank_matrix.shape, dtype=np.float32)
    for j in range(len(langs)):  # copy weights for wrong predictions
        row_indices = (rank_matrix[:, j] == -1)
        wrong[row_indices, j] = weight_matrix[row_indices]
    wrong = np.sum(wrong, axis=0) + 1e-7  # to make it non-zero

    correct = np.zeros(rank_matrix.shape)
    for j in range(len(langs)):
        row_indices = (rank_matrix[:, j] == 1)
        correct[row_indices, j] = weight_matrix[row_indices]
    correct = np.sum(correct, axis=0) + 1e-7

    lang_weights = 0.5 * np.log(correct / wrong)

    return lang_weights


def learn_model_weights(weight_matrix, rank_matrix):
    T = 5
    lang_final_weights = np.zeros(len(langs))

    for i in range(T):
        lang_weights = model_weights_for_one_round(weight_matrix, rank_matrix)

        # choose language (best of this round) that has not been chosen
        lang_order = np.argsort(lang_weights)[
                     ::-1]  # lang_order[k]=j means language j is the kth best (has kth largest weight)

        for lang in lang_order:
            if not lang_final_weights[lang] > 0:
                best_lang = lang
                break
        best_lang_weight = lang_weights[best_lang]
        lang_final_weights[best_lang] += best_lang_weight

        # update datapoint weights according to prediction of the best language
        coeff = np.exp(-best_lang_weight * rank_matrix[:, best_lang])
        weight_matrix = weight_matrix * coeff
        weight_matrix = weight_matrix / np.sum(weight_matrix)  # normalize

    final = {lang: w for lang, w in zip(langs, lang_final_weights)}

    return final


def learn_entity_specific_weights(target_lang, df, e):
    if target_lang == 'en':
        MIN_TRIPLE = 1
    else:
        MIN_TRIPLE = 3


    triples = triples_of_given_entity(df, e)
    if len(triples) < MIN_TRIPLE:  # too few triples, just use mrr
        return mrrs[target_lang]

    subdf = triples.copy().reset_index(drop=True)

    # get entity pair weights and prediction correctness
    subdf['weight_matrix'] = ''
    subdf['weight_matrix'] = subdf.apply(lambda row: weight_matrix_from_one_triple(row), axis=1)
    subdf['rank_matrix'] = subdf.apply(lambda row: rank_matrix_from_one_triple(row), axis=1)

    # concatenate matrices for multiple triples by axis=0
    big_weight_matrix = np.concatenate(subdf['weight_matrix'].values, axis=0)
    big_rank_matrix = np.concatenate(subdf['rank_matrix'].values, axis=0)
    # normalize datapoint weights
    big_weight_matrix = big_weight_matrix / np.sum(big_weight_matrix)

    model_weights = learn_model_weights(big_weight_matrix, big_rank_matrix)
    return model_weights


# when no related triple in validation set, use MRR by default
if param.knowledge == 'transe':
    mrrs = {'el': {'el': 0.463, 'fr': 0.538, 'en': 0.359, 'es': 0.478, 'ja': 0.67},
            'ja': {'ja': 0.385, 'fr': 0.569, 'en': 0.377, 'es': 0.435, 'el': 0.75},
            'es': {'es': 0.425, 'fr': 0.54, 'en': 0.352, 'ja': 0.685, 'el': 0.764},
            'fr': {'fr': 0.359, 'en': 0.38, 'es': 0.459, 'ja': 0.70, 'el': 0.77},
            'en': {'en': 0.297, 'fr': 0.489, 'es': 0.50, 'ja': 0.675, 'el': 0.758}
            }
else:
    mrrs = {
        'en': {'en': 0.37, 'fr': 0.648, 'es': 0.657, 'ja': 0.722, 'el': 0.836},
        'fr': {'ja': 0.34, 'fr': 0.67, 'en': 0.57, 'es': 0.68, 'el': 0.80},
        'ja': {'ja': 0.34, 'fr': 0.67, 'en': 0.57, 'es': 0.68, 'el': 0.80},
        'el': {'el': 0.11, 'fr': 0.74, 'en': 0.55, 'es': 0.71, 'ja': 0.71},
        'es': {'es': 0.43, 'fr': 0.684, 'en': 0.573, 'ja': 0.717, 'el': 0.804}
    }


langs = [ 'fr', 'en', 'es',  'ja',  'el']
#%%%%%%%%%%%%%%%%%%%%%%


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




def run(lang):
    target_lang = lang
    src_langs = ['fr', 'ja', 'es', 'el', 'en']
    src_langs.remove(target_lang)

    suffix = '-val.tsv'  # use validation file for weight learning
    testfile = join(data_dir, target_lang + suffix)

    set_logger(param, model_dir)
    logging.info(f'target_lang: {target_lang}')


    try:
        # target (sparse) kg
        kg0 = load_target_kg(data_dir, target_lang)  # target kg. KnowledgeGraph object
        # supporter kgs
        supporter_kgs = load_support_kgs(data_dir, seed_dir, target_lang, src_langs)  # list[KnowledgeGraph]
    except ValueError:
        print(f'{lang} skipped')

    validator = MultiModelTester(kg0, supporter_kgs)

    # load model
    kg0.load_model(model_dir)
    for kg1 in supporter_kgs:
        kg1.load_model(model_dir)

    # recompute filtered_reordered_embedding_matrix using the latest embeddings
    kg0.filtered_reordered_embedding_matrix = None
    for kg1 in supporter_kgs:
        kg1.filtered_reordered_embedding_matrix = None


    output = join(model_dir, 'results'+suffix)
    testcases = pd.read_csv(testfile, sep='\t', header=None).values
    param.n_test = testcases.shape[0]  # test on all training triples
    print('Loaded test cases from: %s'%testfile)

    results_df = validator.test_and_record_results(testcases)
    results_df.to_csv(output, sep='\t', index=False)


target_lang = 'ja'

param.lang = target_lang

src_langs = ['fr', 'ja', 'es',  'el',  'en']
src_langs.remove(target_lang)




# load data
data_dir = './data/kg'
seed_dir = './data/seed_alignlinks'
model_dir = join('./model', f'kens-{param.knowledge}', target_lang)  # output

output_file = join(model_dir, 'weights.tsv')

entity_dir = './data/entity'
vocabs = pd.read_csv(join(entity_dir, f'{target_lang}.tsv'), sep='\t', header=None)
num_entity = len(vocabs)

# logging
set_logger(param, model_dir)  # set logger.
logging.info('=== Boosting')
logging.info(f'target_lang: {target_lang}')


test_result_file = join(model_dir,  'results-test.tsv')
weights_file = join(model_dir,'weights.tsv')
langs = [target_lang] + src_langs
# === PARAMETERS END ===



def extract_entities(id_score_tuples):
    return [ent_id for ent_id, score in id_score_tuples]

langs =[ 'fr', 'en', 'es',  'ja',  'el']

# use validation set predictions for weight learning
file = join(model_dir, 'results-val.tsv')  # if not generated, change the suffix in train.ipynb and generate it
output = join(model_dir, 'weights.tsv')
M = len(langs)

pandarallel.initialize()

# if os.path.exists(join(model_dir, 'results-val.tsv')):
#     run('ja')

# load predictions on validation set
df = pd.read_csv(file, sep='\t')  # (h,r,t) => predictions
for model in langs:  # convert string to list
    df[model] = df[model].apply(lambda x: ast.literal_eval(x))
    # extract entities
    df[model] = df[model].apply(lambda x: extract_entities(x))
    

# load predictions on test set for testing
test_result_file = join(model_dir,  'results-test.tsv')
result = pd.read_csv(test_result_file, sep='\t')
# result = result.loc[:1000,:]
for lang in langs:
    result[lang] = result[lang].apply(lambda x: ast.literal_eval(x))  # convert list string to list
print(result.head(2))



# multiprocessing weight learning
time0 = time.time()

# learn weights for all entities
weight_df = pd.DataFrame(np.arange(num_entity), columns=['e'])
weight_df['weights'] = weight_df['e'].parallel_apply(lambda e: learn_entity_specific_weights(target_lang, df, e))

print(f'time {time.time()-time0}')

print('# of entities:', len(weight_df))
weight_df.head(5)


def hr2t_from_train_set():
    train_df = pd.read_csv(join(data_dir, f'{target_lang}-train.tsv'), sep='\t')
    tripleset = set([tuple([h,r,t]) for h,r,t in (train_df.values)])

    hr2t = {}  # {(h,r):set(t)}
    for tp in tripleset:
        h,r,t=tp[0],tp[1],tp[2]
        if (h,r) not in hr2t:
            hr2t[(h,r)] = set()
        hr2t[(h,r)].add(t)
    return hr2t


wdf = weight_df.copy()
weights = pd.Series(wdf['weights'].values, index=wdf['e']).to_dict() # {entity: {model:weight}}

logging.info(f'[{target_lang}]')
hr2t_train = hr2t_from_train_set()

logging.info('-- filtered')
filt_ensemble_hits_at_n(result, weights, langs, hr2t_train, n=10)
filt_ensemble_hits_at_n(result, weights, langs, hr2t_train, n=3)
filt_ensemble_hits_at_n(result, weights, langs, hr2t_train, n=1)



