from __future__ import division
import tensorflow as tf
from tensorflow import keras
import src.param as param
import time
import numpy as np
from scipy.spatial.distance import cosine as cos_dist
import logging
import pandas as pd
from os.path import join



def create_kNN_finder(predictor, num_entity):
    """
    kNN finder
    === input: [input_h_query, input_r_query, embedding_matrix]
    === output:[top_k_entity_idx, top_k_scores(larger is better)] . top_k_entity_idx shape [1,k].
                e.g., [array([[64931, 13553, 20458]]), array([[-1.008282 , -2.0292854, -3.059666]])]
    """
    input_h_query = keras.layers.Input(batch_shape=(1, 1))
    input_r_query = keras.layers.Input(batch_shape=(1, 1))
    embedding_matrix = keras.layers.Input(shape=(None,),
                                          name='Embedding_matrix')
    predicted_t = predictor([input_h_query, input_r_query])

    kNN_idx_and_score = keras.layers.Lambda(find_kNN, name='find_kNN', output_shape=(param.k,))(
        [predicted_t, embedding_matrix])

    kNN_finder = keras.Model([input_h_query, input_r_query, embedding_matrix],
                             kNN_idx_and_score)
    return kNN_finder


def create_vec_kNN_finder(num_entity):
    """
    Given a vector, find the kNN entities
    :param num_entity:
    :return:
    """
    input_vec = keras.layers.Input(batch_shape=(1, param.dim))
    embedding_matrix = keras.layers.Input(shape=(num_entity * param.dim,),
                                          name='Embedding_matrix')
    kNN_idx_and_score = keras.layers.Lambda(find_kNN, name='find_kNN', output_shape=(param.k,))(
        [input_vec, embedding_matrix])
    vec_kNN_finder = keras.Model([input_vec, embedding_matrix], kNN_idx_and_score)
    return vec_kNN_finder


def create_alignment_kNN_finder(num_entity):
    """
    kNN finder
    === input: [input_h_query, input_r_query, embedding_matrix]
    === output:[top_k_entity_idx, top_k_scores(larger is better)] . top_k_entity_idx shape [1,k].
                e.g., [array([[64931, 13553, 20458]]), array([[-1.008282 , -2.0292854, -3.059666]])]
    """
    input_t_vec = keras.layers.Input(shape=(param.dim,))
    embedding_matrix = keras.layers.Input(shape=(num_entity * param.dim,),
                                          name='Embedding_matrix')

    kNN_idx_and_score = keras.layers.Lambda(find_kNN, name='find_kNN', output_shape=(param.k,))(
        [input_t_vec, embedding_matrix])

    kNN_finder = keras.Model([input_t_vec, embedding_matrix],
                             kNN_idx_and_score)
    return kNN_finder


def create_knowledge_model(num_entity, num_relation, relation_layer=None):
    """
    model: (h,r,t) -> loss/distance/-score. Main model, used for training embedding.
    predictor: (h,r) -> predicted t vector
    kNN finder: (h,r) -> top k possible t
    :return: model, predictor, kNN_finder
    """
    input_h = keras.layers.Input(shape=(1,), name='input_h')
    input_r = keras.layers.Input(shape=(1,), name='input_r')
    input_t = keras.layers.Input(shape=(1,), name='input_t')

    if param.knowledge == 'rotate':  # double embedding
        emd_range = param.rotate_embedding_range()
        entity_embedding_layer = keras.layers.Embedding(input_dim=num_entity,
                                                        output_dim=param.dim,
                                                        # double embedding. need an even number
                                                        embeddings_initializer=keras.initializers.RandomUniform(
                                                            -emd_range, emd_range)
                                                        )
        rel_embedding_layer = keras.layers.Embedding(input_dim=num_relation,
                                                     output_dim=int(param.dim / 2),  # half entity embedding size
                                                     embeddings_initializer=keras.initializers.RandomUniform(
                                                         -emd_range, emd_range)
                                                     )
    else:
        entity_embedding_layer = keras.layers.Embedding(input_dim=num_entity,
                                                        output_dim=param.dim,
                                                        embeddings_regularizer=keras.regularizers.l2(param.reg_scale),
                                                        embeddings_constraint=keras.constraints.MinMaxNorm(min_value=0.95, max_value=1.0, rate=0.7, axis=1),
                                                        )  # shape

        rel_embedding_layer = keras.layers.Embedding(input_dim=num_relation,
                                                     output_dim=param.dim,
                                                     embeddings_regularizer=keras.regularizers.l2(param.reg_scale),
                                                     embeddings_constraint=keras.constraints.max_norm(1., axis=1),
                                                     )

    h, r, t = entity_embedding_layer(input_h), rel_embedding_layer(input_r), entity_embedding_layer(input_t)

    projection_layer = keras.layers.Lambda(project_t)
    projected_t = projection_layer([h, r])

    loss_layer = keras.layers.Lambda(define_loss, name='compute_loss')
    pos_loss = loss_layer([t, projected_t])

    # negative sampling
    rand_negs = keras.layers.Lambda(
        lambda placeholder: tf.random_uniform(shape=(param.neg_per_pos,), maxval=num_entity - 1),
        name='generate_negative_t_samples')(input_h)  # input_h is just a placeholder
    # random_t_indices = keras.backend.variable(tf.random_uniform(shape=(param.neg_per_pos,), maxval=num_entity - 1))
    neg_ts = entity_embedding_layer(rand_negs)  # (3, 128)
    neg_losses = loss_layer([t, neg_ts])

    if param.knowledge == 'rotate':
        # current loss is actually dist
        gm = param.gamma
        pos_loss1 = keras.layers.Lambda(lambda dist: -tf.math.log(tf.math.softplus(gm - dist)), output_shape=(1,))(pos_loss)
        neg_losses1 = keras.layers.Lambda(lambda dist: -tf.math.log(tf.math.softplus(dist-gm)), output_shape=(param.neg_per_pos,))(neg_losses)
        neg_loss1 = keras.layers.Lambda(lambda x: tf.reduce_mean(x, axis=-1), output_shape=(1,),
                                       name='average_loss_over_neg_samples')(neg_losses1)
        total_loss = keras.layers.Lambda(lambda loss: (loss[0] + loss[1]) / 2, output_shape=(1,))([pos_loss1, neg_loss1])
        # Model2. Main model, used for training embedding. (h,r,t) -> loss/distance/-score.
        model = keras.Model([input_h, input_r, input_t], total_loss)
        model.compile(optimizer=keras.optimizers.Adam(lr=param.lr), loss=lambda y_true, loss: loss)  # use model output as customized loss (something to minimize)
    else:
        neg_loss = keras.layers.Lambda(lambda x: tf.reduce_mean(x, axis=-1), output_shape=(1,),
                                       name='average_loss_over_neg_samples')(neg_losses)

        total_loss = keras.layers.Lambda(lambda loss: tf.maximum(loss[0] - loss[1] + param.margin, 0))(
            [pos_loss, neg_loss])
        # Model2. Main model, used for training embedding. (h,r,t) -> loss/distance/-score.
        model = keras.Model([input_h, input_r, input_t], total_loss)
        model.compile(optimizer=keras.optimizers.Adam(lr=param.lr),
                      loss=lambda y_true, loss: loss)  # use model output as customized loss (something to minimize)

    predictor = keras.Model([input_h, input_r], projected_t)  # Model1: (h,r) -> projected t vector

    # Model3. Used for validation.
    # when using kNN finder, we can only use one input_h and one input_r at a time (batch_size=1)
    kNN_finder = create_kNN_finder(predictor, num_entity)

    return model, predictor, kNN_finder

def project_t(hr):
    if param.knowledge == 'transe':
        return hr[0] + hr[1]
    elif param.knowledge == 'rotate':
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
    t_true = t_true_pred[0]
    t_pred = t_true_pred[1]

    # tf.norm() will result in nan loss when tf.norm([0])
    # USE tf.reduce_mean(tf.square()) INSTEAD!!!
    # return tf.reduce_mean(tf.square(t_true-t_pred), axis=2)  # input shape: (None, 1, dim)
    return tf.norm(t_true - t_pred + 1e-8, axis=2)  # input shape: (None, 1, dim)

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


def define_align_loss(v1v2v3v4):
    e1 = v1v2v3v4[0]
    e0_1 = v1v2v3v4[1]
    loss1 = tf.norm(e1 - e0_1, axis=2)  # e1 Shape (None, 1, dim)

    e0 = v1v2v3v4[2]
    e0_1_0 = v1v2v3v4[3]
    loss2 = tf.norm(e0 - e0_1_0, axis=2)  # e1 Shape (None, 1, dim)
    return loss1 + loss2


def mse(a, b):
    # tf.norm() will result in nan loss when tf.norm([0])
    # USE tf.reduce_mean(tf.square()) INSTEAD!!!
    return tf.reduce_mean(tf.square(a - b), axis=-1)


def l2distance(a, b):
    # dist = tf.sqrt(tf.reduce_sum(tf.square(a-b), axis=-1))
    dist = tf.norm(a - b + 1e-8, axis=-1)
    return dist


def create_alignment_model(knowledge_model0, knowledge_model1):
    input_e0 = keras.layers.Input(shape=(1,), name='input_e1')  # entity id in language 1
    input_e1 = keras.layers.Input(shape=(1,), name='input_e2')  # the id of aligned entity in language 2

    # embedding_layer1 = knowledge_model1.get_layer(name='entity_embedding')
    # embedding_layer2 = knowledge_model2.get_layer(name='entity_embedding')

    embedding_layer0 = knowledge_model0.layers[
        3]  # entity embedding. Don't name it or retrieve by name, otherwise name conflict
    embedding_layer1 = knowledge_model1.layers[3]
    e0 = embedding_layer0(input_e0)
    e1 = embedding_layer1(input_e1)

    my_loss = keras.layers.Lambda(lambda v1v2: l2distance(v1v2[0], v1v2[1]))([e0, e1])

    align_model = keras.Model([input_e0, input_e1], my_loss)

    align_model.compile(optimizer=keras.optimizers.Adam(lr=param.align_lr, amsgrad=True),
                        loss=lambda y_true, loss: loss)  # use model output as customized loss (something to minimize)

    return align_model


def extract_entity_embedding_matrix(knowledge_model):
    """
    Get the embedding matrix and FLATTEN it to ensure successful feed to keras layer
    :param knowledge_model:
    :return:
    """
    return np.squeeze(np.array(knowledge_model.layers[3].get_weights())).reshape([1, -1])


def save_model_structure(model, output_path):
    json_string = model.to_json()
    with open(output_path, 'w') as outfile:
        outfile.write(json_string)



def extend_seed_align_links(kg0, kg1, seed_links):
    """
    Self learning using cross-domain similarity scaling (CSLS) metric for kNN search
    :param kg0: supporter kg
    :param kg1: target kg
    :param seed_links: 2-col np array
    :return:
    """

    def cos(v1, v2):
        return 1 - cos_dist(v1, v2)

    csls_links = []

    aligned0 = set(list(seed_links[:, 0]))
    aligned1 = set(list(seed_links[:, 1]))

    k_csls = 3  # how many nodes in neiborhood
    k_temp = param.k
    param.k = k_csls
    kNN_finder1 = create_alignment_kNN_finder(kg1.num_entity)  # k=k_csls
    kNN_finder0 = create_alignment_kNN_finder(kg0.num_entity)

    embedding_matrix0 = kg0.get_embedding_matrix()
    embedding_matrix0_reshaped = embedding_matrix0.reshape([kg0.num_entity, param.dim])
    embedding_matrix1 = kg1.get_embedding_matrix()
    embedding_matrix1_reshaped = embedding_matrix1.reshape([kg1.num_entity, param.dim])

    # change param.k back for link prediction
    param.k = k_temp

    # find kNN for each e0
    # mean neighborhood similarity
    e0_neighborhood = np.zeros([kg0.num_entity, k_csls])
    e1_neighborhood = np.zeros([kg1.num_entity, k_csls])
    e0_neighborhood_cos = np.zeros(kg0.num_entity)
    e1_neighborhood_cos = np.zeros(kg1.num_entity)

    # find neighborhood
    for e0 in range(kg0.num_entity):
        top_k_from_kg1 = kNN_finder1.predict(
            [embedding_matrix0_reshaped[e0, :].reshape([1, -1]), embedding_matrix1],
            batch_size=1)  # [array(entity), array(score)]
        neighbood = top_k_from_kg1[0]  # list[entity], possible e1
        e0_neighborhood[e0, :] = neighbood
    for e1 in range(kg1.num_entity):
        top_k_from_kg0 = kNN_finder0.predict(
            [embedding_matrix1_reshaped[e1, :].reshape([1, -1]), embedding_matrix0],
            batch_size=1)
        neighbood = top_k_from_kg0[0]  # list[entity], possible e0
        e1_neighborhood[e1, :] = neighbood

    e0_neighborhood = e0_neighborhood.astype(np.int32)
    e1_neighborhood = e1_neighborhood.astype(np.int32)

    # compute neighborhood similarity
    for e0 in range(kg0.num_entity):
        e0_vec = embedding_matrix0_reshaped[e0]
        e0_neighbors = e0_neighborhood[e0, :]  # e0's neighbor in kg1 domain
        neighbor_cos = [cos(embedding_matrix1_reshaped[nb, :], e0_vec) for nb in e0_neighbors]
        e0_neighborhood_cos[e0] = np.mean(neighbor_cos)  # r_S

    for e1 in range(kg1.num_entity):
        e1_vec = embedding_matrix1_reshaped[e1]
        e1_neighbors = e1_neighborhood[e1, :]  # e0's neighbor in kg1 domain
        neighbor_cos = [cos(embedding_matrix0_reshaped[nb, :], e1_vec) for nb in e1_neighbors]
        e1_neighborhood_cos[e1] = np.mean(neighbor_cos)

    nearest_for_e0 = np.full(kg0.num_entity, fill_value=-2)  # -2 for not computed, -1 for not found
    nearest_for_e1 = np.full(kg1.num_entity, fill_value=-2)

    for true_e0 in range(kg0.num_entity):
        if true_e0 not in aligned0:
            e0_neighbors = e0_neighborhood[true_e0, :]  # e0's neighbor in kg1 domain
            nearest_e1 = -1
            nearest_e1_csls = -np.inf
            for e1 in e0_neighbors:
                if e1 not in aligned1:
                    # rT(Wx_s) is the same for all e1 in e0's neighborhood
                    csls = 2 * cos(embedding_matrix0_reshaped[true_e0, :], embedding_matrix1_reshaped[e1, :]) - \
                           e1_neighborhood_cos[e1]
                    if csls > nearest_e1_csls:
                        nearest_e1 = e1
            nearest_for_e0[true_e0] = nearest_e1

            # check if they are mutual neighbors
            if nearest_for_e0[true_e0] != -1:
                e1 = nearest_for_e0[true_e0]
                if nearest_for_e1[e1] == -2:  # e1's nearest number not computed yet. compute it now
                    e1_neighbors = e1_neighborhood[e1, :]  # e0's neighbor in kg1 domain
                    nearest_e0 = -1
                    nearest_e0_csls = -np.inf
                    for e0 in e1_neighbors:
                        if e0 not in aligned0:
                            # rT(Wx_s) is the same for all e1 in e0's neighborhood
                            csls = 2 * cos(embedding_matrix1_reshaped[e1, :], embedding_matrix0_reshaped[e0, :]) - \
                                   e0_neighborhood_cos[e0]
                            if csls > nearest_e0_csls:
                                nearest_e0 = e0
                                nearest_e0_csls = csls
                    nearest_for_e1[e1] = nearest_e0

                if nearest_for_e1[e1] == true_e0:
                    # mutual e1_neighbors
                    csls_links.append([true_e0, e1])


    csls_links = np.array(csls_links)

    return csls_links
