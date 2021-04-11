from __future__ import division
import time
import pandas as pd
import src.param as param
import numpy as np
from enum import Enum
from src.ensemble import voting, voting_with_model_weight, voting_with_model_weight_and_rrf, filt_voting_with_model_weight
from inspect import signature
from src.model import project_t, create_alignment_kNN_finder
import logging
from os.path import join


class TestMode(Enum):
    KG0 = 'kg0'
    KG1 = 'kg1'
    Transfer = 'link_redirect'
    VOTING = 'voting'


class MultiModelTester:
    def __init__(self, target_kg, supporter_kgs):
        """
        :param target_kg: KnowledgeGraph object
        :param support_kgs: list[KnowledgeGraph]
        """
        self.target_kg = target_kg
        self.supporter_kgs = supporter_kgs

    def get_embedding_matrix(self, model):
        return np.squeeze(np.array(model.layers[3].get_weights())).reshape([1, -1])

    def flatten_kNN_finder_output(self, kNN_finder_output):
        """
        :param kNN_finder_output: [array([[id1, id2, ...]]), array([[score1, score2, ...]])]
        :return: [(id1, score1), (id2, score2), ...]
        """
        # indices, scores = kNN_finder_output
        # kNN_finder returns [array([[id1, id2, ...]]), array([[score1, score2, ...]])]
        indices = kNN_finder_output[0][0]  # flatten [[id1, id2, ...]] -> [id1, id2, ...]
        scores = kNN_finder_output[1][0]  # flatten [[score1, score2, ...]] -> [score1, score2, ...]
        topk_indices_scores = [(indices[i], scores[i]) for i in range(len(indices))]
        return topk_indices_scores

    def extract_entities(self, id_score_tuples):
        return [ent_id for ent_id, score in id_score_tuples]

    def predict(self, h, r, mode, supporter_kg=None, voting_function=None):
        """
        If mode is LINK_TRANSFER and h not in align_dict, return [[]]
        :param h: np.int32
        :param r: np.int32
        :param mode:
        :param supporter_kg: needed when mode == KG1 or LINK_REDIRECT. None for KG0 or VOTING
        :return:
        """
        if (mode == TestMode.KG1 or mode == TestMode.Transfer) and (not supporter_kg):
            supporter_kg = self.supporter_kgs[0]
            print('TestMode: %s but no kg specified. Using supporter_kgs[0], language: %s' % (mode, supporter_kg.lang))

        if mode == TestMode.KG0:
            # input shape must be (1,1) to feed h,r into kNN_finder  (batch_size=1, column=1)
            h = np.reshape(h, (1, 1))
            r = np.reshape(r, (1, 1))
            top_k_indices_scores = self.target_kg.kNN_finder.predict(
                [h, r, self.get_embedding_matrix(self.target_kg.model)],
                batch_size=1)  # [array([[id1, id2, ...]]), array([[score1, score2, ...]])]
            top_k = self.flatten_kNN_finder_output(top_k_indices_scores)  # [(id1, score1), (id2, score2), ...]
            return top_k

        elif mode == TestMode.KG1:  # query on supporter kgs
            # input shape must be (1,1) to feed h,r into kNN_finder  (batch_size=1, column=1)
            h = np.reshape(h, (1, 1))
            r = np.reshape(r, (1, 1))
            top_k_indices_scores = supporter_kg.kNN_finder.predict(
                [h, r, self.get_embedding_matrix(supporter_kg.model)], batch_size=1)
            top_k = self.flatten_kNN_finder_output(top_k_indices_scores)  # [(id1, score1), (id2, score2), ...]
            return top_k

        elif mode == TestMode.Transfer:
            return self.__predict_by_knowledge_transfer(h, r, supporter_kg)

        elif mode == TestMode.VOTING:
            choice_lists = []  # list[list[(item,score)]]

            choices0 = self.predict(h, r, mode=TestMode.KG0, supporter_kg=None)
            choice_lists.append(choices0)

            for sup_kg in self.supporter_kgs:
                choices1 = self.predict(h, r, mode=TestMode.Transfer, supporter_kg=sup_kg)
                choice_lists.append(choices1)
            if voting_function is None:
                return voting(choice_lists, param.k)
            else:
                return voting_function(choice_lists, param.k)

    def __predict_by_knowledge_transfer(self, h0, r0, supporter_kg=None):
        """
        :param h0: shape (1,1) [[id]]
        :param r0: shape (1,1) [[id]]
        :return:
        """
        if h0 not in supporter_kg.dict0to1:
            return []
        h1 = supporter_kg.dict0to1[h0]
        r1 = r0

        # filtered setting
        h1 = np.reshape(h1, (1, 1))
        r1 = np.reshape(r1, (1, 1))
        # The entities will directly be entity id in kg0 if we use filtered_reordered_embedding matrix
        t0_and_scores = supporter_kg.kNN_finder.predict(
            [h1, r1, supporter_kg.get_filtered_reordered_embedding_matrix(self.target_kg.num_entity)],
            batch_size=1)
        t0_and_scores = self.flatten_kNN_finder_output(t0_and_scores)  # [(id1, score1), (id2, score2), ...]

        # original setting
        # t1_and_scores = self.predict(h1, r1, mode=TestMode.KG1, supporter_kg=supporter_kg)  # [(id, score), ...]
        # t0_and_scores = [(supporter_kg.dict1to0[t1], score) for (t1,score) in t1_and_scores if t1 in supporter_kg.dict1to0]
        return t0_and_scores

    def test(self, mode, supporter_kg=None, voting_function=None):
        """
        Compute Hits@10 on first param.n_test test samples
        :param mode: TestMode. For LINK_TRANSFER, triples without links will be skipped
        :param supporter_kg: needed when mode == KG1 or LINK_REDIRECT. None for KG0 or VOTING
        :param voting_function: used when mode==VOTING. Default: vote by count
        :return:
        """

        time0 = time.time()
        hits = 0
        samples = min(param.n_test, self.target_kg.h_test.shape[0])

        # used when mode is LINK_TRANSFER
        linked_triple = samples
        retrieved_t0 = 0
        no_retreived = 0

        for i in range(samples):
            if mode == TestMode.KG1:
                if i >= supporter_kg.h_test.shape[0]:
                    break
                # input shape must be (1,1) to feed h,r into kNN_finder  (batch_size=1, column=1)
                h, r, t = supporter_kg.h_test[i], supporter_kg.r_test[i], supporter_kg.t_test[i]
            else:
                # input shape must be (1,1) to feed h,r into kNN_finder  (batch_size=1, column=1)
                h, r, t = self.target_kg.h_test[i], self.target_kg.r_test[i], self.target_kg.t_test[i]

            top_k = self.predict(h, r, mode, supporter_kg=supporter_kg,
                                 voting_function=voting_function)  # list[(idx,score)] Length may be less than k in LINK_TRANSFER MODE
            top_k_entities = self.extract_entities(top_k)
            if t in top_k_entities:
                # print('hit')
                hits += 1
            else:
                if param.verbose:
                    # print the wrong test cases
                    print('Test case with wrong result: h,r,t', h, r, t)
                    print(top_k)

            if mode == TestMode.Transfer:
                if h not in supporter_kg.dict0to1:
                    linked_triple -= 1
                else:
                    retrieved_t0 += len(top_k)
                    if len(top_k) == 0:
                        no_retreived += 1

        # logging.info('===Validation %s===' % mode)
        if mode == TestMode.Transfer:
            hit_ratio = hits / linked_triple
            logging.info(
                'Hits@%d in %d linked triples in %d: %f' % (param.k, linked_triple, samples, hits / linked_triple))
            logging.info('Average retrieved t0: %f' % (retrieved_t0 / linked_triple))
            logging.info('%d queries have h link but no retrieved t' % (no_retreived))
        else:
            hit_ratio = hits / samples
            logging.info('Hits@%d (%d triples): %f' % (param.k, samples, (hits / samples)))

        print('time: %s' % (time.time() - time0))
        return hit_ratio

    def test_and_record_results(self, testcases):
        time0 = time.time()
        hits = 0
        samples = min(param.n_test, testcases.shape[0])

        results = []  # list[(h ,r, t, nominations)].
        # nominations=list[list[(entity_idx, score)]], len(nominations)=1(target)+len(supporter_kgs)

        for i in range(samples):
            # input shape must be (1,1) to feed h,r into kNN_finder  (batch_size=1, column=1)
            h, r, t = testcases[i, 0], testcases[i, 1], testcases[i, 2]

            nominations = []  # list[list[(entity_idx, score)]], len(nominations)=1(target)+len(supporter_kgs)
            kg0_top_k = self.predict(h, r, TestMode.KG0)
            nominations.append(kg0_top_k)

            for kg1 in self.supporter_kgs:
                kg1_top_k = self.predict(h, r, mode=TestMode.Transfer, supporter_kg=kg1)
                nominations.append(kg1_top_k)

            results.append([h, r, t] + nominations)  # [h,r,t, list[(idx,score)], list[(idx,score)], ...]

        results_df = pd.DataFrame(results, columns=['h', 'r', 't', self.target_kg.lang] + [kg1.lang for kg1 in
                                                                                           self.supporter_kgs])
        return results_df



    def test_voting(self, voting_function, model_weights=None):
        """
        Test specially on mode==VOTING
        Compute Hits@10 on first param.n_test test samples
        :param mode: TestMode. For LINK_TRANSFER, triples without links will be skipped
        :return:
        """
        time0 = time.time()
        hits = 0
        samples = param.n_test

        for i in range(samples):

            # input shape must be (1,1) to feed h,r into kNN_finder  (batch_size=1, column=1)
            h, r, t = self.target_kg.h_test[i], self.target_kg.r_test[i], self.target_kg.t_test[i]

            top_k = self.predict_by_voting(h, r, voting_function=voting_function)
            top_k_entities = self.extract_entities(top_k)
            if t in top_k_entities:
                # print('hit')
                hits += 1
            else:
                if param.verbose:
                    # print the wrong test cases
                    print('Test case with wrong result: h,r,t', h, r, t)
                    print(top_k)

        logging.info('Hits@%d in first (%d triples): %f' % (param.k, samples, (hits / samples)))

    def predict_by_voting(self, h, r, voting_function, model_weights=None):
        """
        If mode is LINK_TRANSFER and h not in align_dict, return [[]]
        :param h: np.int32
        :param r: np.int32
        :param mode:
        :param supporter_kg: needed when mode == KG1 or LINK_REDIRECT. None for KG0 or VOTING
        :return:
        """
        choice_lists = []  # list[list[(item,score)]]

        choices0 = self.predict(h, r, mode=TestMode.KG0, supporter_kg=None)
        choice_lists.append(choices0)

        for sup_kg in self.supporter_kgs:
            choices1 = self.predict(h, r, mode=TestMode.Transfer, supporter_kg=sup_kg)
            choice_lists.append(choices1)

        # if param.verbose:
        #     print(param.lang)
        #     print(choices0)
        #     for i in range(1, len(choice_lists)):
        #         print(self.supporter_kgs[i-1].lang)
        #         print(choice_lists[i])

        # check number of parameters of voting_function
        sig = signature(voting_function)
        len_params = len(sig.parameters)
        if len_params == 2:
            return voting_function(choice_lists, param.k)
        else:
            return voting_function(choice_lists, param.k, model_weights)


def hits10_with_weight(results, weight, model_lists):
    """
    :param results: a result dataframe of predictions. [h, r, t, el, ja, en, ...]
    :param weight: dict {entity: {model:weight}}
    :param model_lists: a list. ['el', 'es','fr','ja','en']
    :return:
    """
    hits = 0
    no_weight = 0
    center = 'h'  # the key in weights
    for index, row in results.iterrows():
        t = row['t']
        predictions = [row[lang] for lang in model_lists]
        if row[center] in weight:
            model_weights_dict = weight[row[center]]  # {model:weight}
            model_weights = [model_weights_dict[lang] for lang in model_lists]
            topk = voting_with_model_weight(choice_lists=predictions, k=10, model_weights=model_weights)
        elif param.lang in weight:  # weight is {model:weight}
            model_weights = [weight[lang] for lang in model_lists]
            topk = voting_with_model_weight(choice_lists=predictions, k=10, model_weights=model_weights)
        else:
            topk = voting(choice_lists=predictions, k=10)  # vote by count
            no_weight += 1
        if t in extract_entities(topk):
            hits += 1
    hits_ratio = hits / results.shape[0]
    logging.info('Hits@10 (%d triples): %.4f' % (results.shape[0], hits_ratio))
    logging.info('%d/%d cases do not have learned model weight' % (no_weight, results.shape[0]))


def hits1_with_weight(results, weight, model_lists):
    """
    :param results: a result dataframe of predictions. [h, r, t, el, ja, en, ...]
    :param weight: dict {entity: {model:weight}}
    :param model_lists: a list. ['el', 'es','fr','ja','en']
    :return:
    """
    hits = 0
    no_weight = 0
    center = 'h'  # the key in weights
    for index, row in results.iterrows():
        t = row['t']
        predictions = [row[lang][:1] for lang in model_lists]
        if row[center] in weight:
            model_weights_dict = weight[row[center]]  # {model:weight}
            model_weights = [model_weights_dict[lang] for lang in model_lists]
            topk = voting_with_model_weight(choice_lists=predictions, k=1, model_weights=model_weights)
        elif param.lang in weight:  # weight is {model:weight}
            model_weights = [weight[lang] for lang in model_lists]
            topk = voting_with_model_weight(choice_lists=predictions, k=1, model_weights=model_weights)
        else:
            topk = voting(choice_lists=predictions, k=1)  # vote by count
            no_weight += 1
        if t in extract_entities(topk):
            hits += 1
    hits_ratio = hits / results.shape[0]
    logging.info('Hits@1 (%d triples): %.4f' % (results.shape[0], hits_ratio))
    logging.info('%d/%d cases do not have learned model weight' % (no_weight, results.shape[0]))


def hits3_with_weight(results, weight, model_lists):
    """
    :param results: a result dataframe of predictions. [h, r, t, el, ja, en, ...]
    :param weight: dict {entity: {model:weight}}
    :param model_lists: a list. ['el', 'es','fr','ja','en']
    :return:
    """
    hits = 0
    no_weight = 0
    center = 'h'  # the key in weights
    for index, row in results.iterrows():
        t = row['t']
        predictions = [row[lang][:1] for lang in model_lists]
        if row[center] in weight:
            model_weights_dict = weight[row[center]]  # {model:weight}
            model_weights = [model_weights_dict[lang] for lang in model_lists]
            topk = voting_with_model_weight(choice_lists=predictions, k=3, model_weights=model_weights)
        elif param.lang in weight:  # weight is {model:weight}
            model_weights = [weight[lang] for lang in model_lists]
            topk = voting_with_model_weight(choice_lists=predictions, k=3, model_weights=model_weights)
        else:
            topk = voting(choice_lists=predictions, k=3)  # vote by count
            no_weight += 1
        if t in extract_entities(topk):
            hits += 1
    hits_ratio = hits / results.shape[0]
    logging.info('Hits@3 (%d triples): %.4f' % (results.shape[0], hits_ratio))
    logging.info('%d/%d cases do not have learned model weight' % (no_weight, results.shape[0]))


def hits10_with_weight_and_rrf(results, weight, model_lists):
    """
    :param results: a result dataframe of predictions. [h, r, t, el, ja, en, ...]
    :param weight: dict {entity: {model:weight}}
    :param model_lists: a list. ['el', 'es','fr','ja','en']
    :return:
    """
    hits = 0
    no_weight = 0
    center = 'h'  # the key in weights
    for index, row in results.iterrows():
        t = row['t']
        predictions = [row[lang] for lang in model_lists]
        if row[center] in weight:
            model_weights_dict = weight[row[center]]  # {model:weight}
            model_weights = [model_weights_dict[lang] for lang in model_lists]
            topk = voting_with_model_weight_and_rrf(choice_lists=predictions, k=10, model_weights=model_weights)
        elif param.lang in weight:  # weight is {model:weight}
            model_weights = [weight[lang] for lang in model_lists]
            topk = voting_with_model_weight_and_rrf(choice_lists=predictions, k=10, model_weights=model_weights)
        else:
            topk = voting(choice_lists=predictions, k=10)  # vote by count
            no_weight += 1
        if t in extract_entities(topk):
            hits += 1
    hits_ratio = hits / results.shape[0]
    logging.info('Hits@10 (%d triples): %.4f' % (results.shape[0], hits_ratio))
    print('%d/%d cases do not have learned model weight' % (no_weight, results.shape[0]))



def hits1(results, lang):
    """
    :param results: df, h,r,t, lang
    :return:
    """
    hits = 0
    no_weight = 0
    for index, row in results.iterrows():
        t = row['t']
        predictions = row[lang]  # list[(entity,socre)]
        first = predictions[0][0]
        if t == first:
            hits += 1
    hits_ratio = hits / results.shape[0]
    logging.info('Hits@1 (%d triples): %.4f' % (results.shape[0], hits_ratio))


def hits3(results, lang):
    """
    :param results: df, h,r,t, lang
    :return:
    """
    hits = 0
    no_weight = 0
    for index, row in results.iterrows():
        t = row['t']
        predictions = row[lang]  # list[(entity,socre)]
        first3 = extract_entities(predictions[:3])
        if t in first3:
            hits += 1
    hits_ratio = hits / results.shape[0]
    logging.info('Hits@3 (%d triples): %.4f' % (results.shape[0], hits_ratio))


def hits10(results, lang):
    """
    :param results: df, h,r,t, lang
    :return:
    """
    hits = 0
    no_weight = 0
    for index, row in results.iterrows():
        t = row['t']
        predictions = row[lang]  # list[(entity,socre)]
        first3 = extract_entities(predictions[:10])
        if t in first3:
            hits += 1
    hits_ratio = hits / results.shape[0]
    logging.info('Hits@10 (%d triples): %.4f' % (results.shape[0], hits_ratio))


def extract_entities(id_score_tuples):
    return [ent_id for ent_id, score in id_score_tuples]


def test_alignment_hits10(kg0, kg1):
    time0 = time.time()
    align_links = np.array(list(kg1.dict0to1.items()))
    hit_1 = 0
    hit_10 = 0
    kNN_finder = create_alignment_kNN_finder(kg1.num_entity)
    embedding_matrix0 = kg0.get_embedding_matrix().reshape([kg0.num_entity, param.dim])
    embedding_matrix1 = kg1.get_embedding_matrix()
    print('=== Test alignment model between [%s] and [%s] ===' % (kg0.lang, kg1.lang))
    for e0, e1 in align_links:
        param.k = 10
        top_k_from_kg1 = kNN_finder.predict(
            [embedding_matrix0[e0, :].reshape([1, -1]), embedding_matrix1], batch_size=1)  # list[(entity,score)]
        entities = top_k_from_kg1[0]  # entities
        if e1 in entities:
            hit_10 += 1
        # if hit_at_10(v0, e1, embedding_matrix1):
        #     hit_10 += 1
    print('Target lang: %s. Source lang: %s' % (kg0.lang, kg1.lang))
    # print('Hits@1: %d/%d=%.4f'%(hit_1, align_links.shape[0], (hit_1/align_links.shape[0])))
    print('Hits@10: %d/%d=%.4f' % (hit_10, align_links.shape[0], (hit_10 / align_links.shape[0])))
    print('time: %s' % (time.time() - time0))


def filt_hits_at_n(results, lang, hr2t_train, n):
    """
    Filtered setting Hits@n when testing
    :param hr2t_train: {(h,r):set(t)}
    :param results: df, h,r,t, lang
    :return:
    """
    hits = 0
    for index, row in results.iterrows():
        t = row['t']
        predictions = row[lang]  # list[(entity,socre)]

        predictions = extract_entities(predictions)
        if (row['h'], row['r']) in hr2t_train:  # filter
            h, r = row['h'], row['r']
            predictions = [e for e in predictions if e not in hr2t_train[(h,r)]]
        predictions = predictions[:n]  # top n
        if t in predictions:
            hits += 1
    hits_ratio = hits / results.shape[0]
    logging.info('Hits@%d (%d triples)(filt): %.4f' % (n, results.shape[0], hits_ratio))
    return hits_ratio


def filt_ensemble_hits_at_n(results, weight, model_lists, hr2t_train, n):
    """
    Filtered setting Hits@n
    :param results: a result dataframe of predictions. [h, r, t, el, ja, en, ...]
    :param weight: dict {entity: {model:weight}}
    :param model_lists: a list. ['el', 'es','fr','ja','en']
    :return:
    """
    hits = 0
    no_weight = 0
    center = 'h'  # the key in weights
    for index, row in results.iterrows():
        t = row['t']
        predictions = [row[lang][:n] for lang in model_lists]

        if row[center] in weight:
            model_weights_dict = weight[row[center]]  # {model:weight}
            model_weights = [model_weights_dict[lang] for lang in model_lists]
        elif param.lang in weight:  # weight is {model:weight}
            model_weights = [weight[lang] for lang in model_lists]
        else:
            model_weights = [1 for lang in model_lists]  # majority vote

        if (row['h'], row['r']) in hr2t_train:  # filtered setting
            train_ts = hr2t_train[(row['h'], row['r'])]
        else:
            train_ts = set()  # empty set

        topk = filt_voting_with_model_weight(choice_lists=predictions,
                                             k=n,
                                             train_ts=train_ts,
                                             model_weights=model_weights)

        if t in extract_entities(topk):
            hits += 1
    hits_ratio = hits / results.shape[0]
    logging.info('Hits@%d (%d triples): %.4f' % (n, results.shape[0], hits_ratio))




def hr2t_from_train_set(data_dir, target_lang):
    train_df = pd.read_csv(join(data_dir, f'{target_lang}-train.tsv'), sep='\t')
    tripleset = set([tuple([h,r,t]) for h,r,t in (train_df.values)])

    hr2t = {}  # {(h,r):set(t)}
    for tp in tripleset:
        h,r,t=tp[0],tp[1],tp[2]
        if (h,r) not in hr2t:
            hr2t[(h,r)] = set()
        hr2t[(h,r)].add(t)
    return hr2t




