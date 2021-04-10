from collections import defaultdict
import numpy as np
import src.param as param


def voting(choice_lists, k):
    """
    :param choice_lists: list[list[(item, score_by_model)]]. choices from each model
    :param k: return top k list[(item,score)]
    :return:
    """
    item_scores = defaultdict(lambda: 0)  # {entity_id: score} default score is 0
    for choice_list in choice_lists:  # choice_list [[id1], [id2], ...]
        for entity, score in choice_list:  # entity [single_id]
            item_scores[entity] += 1
    sorted_item_scores = sorted(item_scores.items(), key=lambda x: x[1], reverse=True)  # descending
    topk = sorted_item_scores[:k]  # list[(item,score)], length k
    # items = [pair[0] for pair in topk]
    return topk


def voting_with_score(choice_lists, k):
    """
    :param choice_lists: list[list[(item, score)]]. choices from each model
    :param k: return top k
    :return:
    """
    item_scores = defaultdict(lambda: 0)  # {entity_id: score} default score is 0
    for choice_list in choice_lists:  # choice_list [[id1], [id2], ...]
        for entity, score in choice_list:  # entity [single_id]
            item_scores[entity] += score
    sorted_item_scores = sorted(item_scores.items(), key=lambda x: x[1], reverse=True)  # descending
    topk = sorted_item_scores[:k]  # list[(item,score)], length k
    return topk


def voting_with_item_score_and_model_weight(choice_lists, k, model_weights=None):
    """
    :param choice_lists: list[list[(item, score)]]. choices from each model
    :param model_weights: weight for each choice_list (model). len(model_weights)==len(choice_lists). Default: [1, 0.3, 0.3, ...]
    :param k: return top k.
    :return:
    """
    if model_weights is None:
        # Default: [1, 0.3, 0.3, ...]
        model_weights = [0.3 for i in range(len(choice_lists))]
        model_weights[0] = 1
    item_scores = defaultdict(lambda: 0)  # {entity_id: score} default score is 0
    for choice_list, weight in zip(choice_lists, model_weights):  # choice_list [[id1], [id2], ...]
        for entity, score in choice_list:  # entity [single_id]
            item_scores[entity] += score * weight
    sorted_item_scores = sorted(item_scores.items(), key=lambda x: x[1], reverse=True)  # descending
    topk = sorted_item_scores[:k]  # list[(item,score)], length k
    return topk



def voting_with_model_weight_and_rrf(choice_lists, k, model_weights=None):
    """
    :param choice_lists: list[list[(item, score)]]. choices from each model
    :param model_weights: weight for each choice_list (model). len(model_weights)==len(choice_lists). Default: [1, 0.3, 0.3, ...]
    :param k: return top k.
    :return:
    """
    if model_weights is None:
        # Default: [1, 0.3, 0.3, ...]
        model_weights = [0.3 for i in range(len(choice_lists))]
        model_weights[0] = 1
    item_scores = defaultdict(lambda: 0)  # {entity_id: score} default score is 0
    ranks = np.arange(1, len(choice_lists[0])+1)  # choice_lists[0]: how many entity candidates in each choice_list
    for rank, choice_list, weight in zip(ranks, choice_lists, model_weights):  # choice_list [[id1], [id2], ...]
        for entity, score in choice_list:  # entity [single_id]
            item_scores[entity] += weight*gain_as_rrf(rank)
    sorted_item_scores = sorted(item_scores.items(), key=lambda x: x[1], reverse=True)  # descending
    topk = sorted_item_scores[:k]  # list[(item,score)], length k
    return topk


def gain_as_rrf(rank, GAIN_CONST=param.rrf_const):
    """
    Define the gain of retrieving a true tail as Reciprocal Rank Fusion (RRF)
    :param rank: 0~len(candidates)-1
    :param GAIN_CONST: hyper-parameter for rrf
    """
    # GAIN_CONST a hyper-parameter. Set as 60 in the original paper：
    # Reciprocal Rank Fusion outperforms Condorcet and individual Rank Learning Methods
    rrf = 1 / (GAIN_CONST + rank)  # Reciprocal Rank Fusion
    return rrf

def gain_as_const_minus_rank(rank, GAIN_CONST=10):
    """
    :param rank: 0~len(candidates)-1
    :param GAIN_CONST: hyper-parameter
    :return:
    """
    return GAIN_CONST-rank


def voting_with_model_weight(choice_lists, k, model_weights=None):
    """
    :param choice_lists: list[list[(item, score)]]. choices from each model
    :param model_weights: weight for each choice_list (model). len(model_weights)==len(choice_lists). Default: [1, 0.3, 0.3, ...]
    :param k: return top k.
    :return:
    """
    if model_weights is None:
        # Default: [1, 0.3, 0.3, ...]
        model_weights = [0.3 for i in range(len(choice_lists))]
        model_weights[0] = 1
    item_scores = defaultdict(lambda: 0)  # {entity_id: score} default score is 0
    for choice_list, weight in zip(choice_lists, model_weights):  # choice_list [[id1], [id2], ...]
        for entity, score in choice_list:  # entity [single_id]
            item_scores[entity] += weight
    sorted_item_scores = sorted(item_scores.items(), key=lambda x: x[1], reverse=True)  # descending
    topk = sorted_item_scores[:k]  # list[(item,score)], length k
    return topk


def filt_voting_with_model_weight(choice_lists, k, train_ts, model_weights=None):
    """
    :param choice_lists: list[list[(item, score)]]. choices from each model
    :param model_weights: weight for each choice_list (model). len(model_weights)==len(choice_lists). Default: [1, 0.3, 0.3, ...]
    :param k: return top k.
    :param train_ts: filter the t that has appeared in training set
    :return:
    """
    if model_weights is None:
        model_weights = [1 for i in range(len(choice_lists))]
    item_scores = defaultdict(lambda: 0)  # {entity_id: score} default score is 0
    for choice_list, weight in zip(choice_lists, model_weights):  # choice_list [[id1], [id2], ...]
        for entity, score in choice_list:  # entity [single_id]
            item_scores[entity] += weight
    sorted_item_scores = sorted(item_scores.items(), key=lambda x: x[1], reverse=True)  # descending
    if len(train_ts) > 0:
        sorted_item_scores = [pair for pair in sorted_item_scores if pair[0] not in train_ts]
    topk = sorted_item_scores[:k]  # list[(item,score)], length k
    return topk

