import tensorflow as tf
from tensorflow import keras
import src.param as param
import time
import numpy as np


# ============== Model (TransE, Distmult, ...) Specific Scoring Functions =================


gamma = 24
epsilon = 1e-7


def project_t(hr):
    """
    :param hr: [h_re, h_im, r_re, r_im]
    :return: [projected_t_re, projected_t_im]
    """
    pi = 3.14159265358979323846

    head, relation = hr[0], hr[1]

    re_head, im_head = tf.split(head, 2, axis=2)  # input shape: (None, 1, dim)

    embedding_range = param.rotate_embedding_range()

    #Make phases of relations uniformly distributed in [-pi, pi]
    phase_relation = relation/(embedding_range/pi)

    re_relation = tf.cos(phase_relation)
    im_relation = tf.sin(phase_relation)

    re_tail = re_head * re_relation - im_head * im_relation
    im_tail = re_head * im_relation + im_head * re_relation

    predicted_tail = tf.concat([re_tail, im_tail], axis=-1)

    return predicted_tail


def define_loss(t_true_pred):
    """
    same as TransE
    :param t_true_pred: [t_re, t_im, projected_t_re, projected_t_im]
    :return:
    """
    # Equivalent to RotatE original t-batch
    t_true, t_pred = t_true_pred[0], t_true_pred[1]

    re_pred, im_pred = tf.split(t_pred, 2, axis=-1)
    re_tail, im_tail = tf.split(t_true, 2, axis=-1)
    re_score = re_pred - re_tail + 1e-7
    im_score = im_pred - im_tail + 1e-7
    score = tf.stack([re_score, im_score], axis=0)
    score = tf.norm(score, axis=0)

    dist = tf.reduce_sum(score, axis=2)
    return dist

def find_kNN(t_vec_and_embed_matrix):
    """
    :param t_vec_and_embed_matrix:
    :return: [top_k_entity_idx, top_k_scores(larger is better)] . top_k_entity_idx shape [1,k].
                e.g., [array([[64931, 13553, 20458]]), array([[-1.008282 , -2.0292854, -3.059666]])]
    """
    predicted_t_vec = tf.squeeze(t_vec_and_embed_matrix[0])  # shape (batch_size=1, 1, dim) -> (dim,)
    embedding_matrix = tf.reshape(t_vec_and_embed_matrix[1], [-1, param.dim])  # new shape (num_entity, dim*2), *2 for double embedding
    distance = tf.norm(tf.subtract(embedding_matrix, predicted_t_vec), axis=1)
    top_k_scores, top_k_t = tf.nn.top_k(-distance, k=param.k)  # find indices of k largest score. score = neg(distance)
    return [tf.reshape(top_k_t, [1, param.k]), tf.reshape(top_k_scores, [1, param.k])]  # reshape to one row matrix to fit keras model output


