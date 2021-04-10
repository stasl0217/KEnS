import tensorflow as tf
from tensorflow import keras
import src.param as param
import time
import numpy as np


# ============== Model (TransE, Distmult, ...) Specific Scoring Functions =================

def define_loss(t_true_pred):
    t_true = t_true_pred[0]
    t_pred = t_true_pred[1]

    # tf.norm() will result in nan loss when tf.norm([0])
    # USE tf.reduce_mean(tf.square()) INSTEAD!!!
    # return tf.reduce_mean(tf.square(t_true-t_pred), axis=2)  # input shape: (None, 1, dim)
    return tf.norm(t_true - t_pred + 1e-8, axis=2)  # input shape: (None, 1, dim)


def project_t(hr):
    return hr[0] + hr[1]


def find_kNN(t_vec_and_embed_matrix):
    """
    :param t_vec_and_embed_matrix:
    :return: [top_k_entity_idx, top_k_scores(larger is better)] . top_k_entity_idx shape [1,k].
                e.g., [array([[64931, 13553, 20458]]), array([[-1.008282 , -2.0292854, -3.059666]])]
    """
    predicted_t_vec = tf.squeeze(t_vec_and_embed_matrix[0])  # shape (batch_size=1, 1, dim) -> (dim,)
    embedding_matrix = tf.reshape(t_vec_and_embed_matrix[1], [-1, param.dim])  # new shape (num_entity, dim)
    distance = tf.norm(tf.subtract(embedding_matrix, predicted_t_vec), axis=1)
    top_k_scores, top_k_t = tf.nn.top_k(-distance, k=param.k)  # find indices of k largest score. score = neg(distance)
    return [tf.reshape(top_k_t, [1, param.k]), tf.reshape(top_k_scores, [1, param.k])]  # reshape to one row matrix to fit keras model output




# ============== Model Specific: END =================
