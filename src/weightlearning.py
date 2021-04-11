import numpy as np
import src.param as param

def weight_matrix_from_one_triple(row, num_entity):
    # weight_matrix[i] is the datapoint weight for the entity pair (true_t, i)
    weight_matrix = np.ones(num_entity)
    return weight_matrix


def rank_matrix_from_one_triple(target_lang, row, num_entity, langs, base):
    """
    rank_matrix shape: [num_entity, len(langs)]
    rank_matrix[i,j]=1 means entity[i] is correctly ranked after true_t by lang j, -1 means wrongly
    :param row: a row in df
    :return:
    """
    true_t = row['t']
    rank_matrix = 1 * np.ones([num_entity, len(langs)])

    for j, lang in enumerate(langs):
        predictions = row[lang]

        if true_t in predictions:
            # unseen datapoints as ranked correctly
            rank_matrix[:, j] = 1
            rank = predictions.index(true_t)  # start from 0
            if rank > 0:
                # update rank_matrix for entities before true_t (they are ranked wrongly)
                rank_matrix[np.array(predictions[:rank + 1]), j] = -1
        else:
            rank_matrix[np.array(predictions).astype(np.int32), j] = -1  # entities in predictions are wrongly ranked
            rest = np.random.randint(0, num_entity, size=(int(100 / base[lang]) - len(predictions)))
            rank_matrix[rest, j] = -1
    return rank_matrix


def triples_of_given_entity(df, e):
    return df[(df['h'] == e) | (df['t'] == e)]


def model_weights_for_one_round(langs, weight_matrix, rank_matrix):
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


def learn_model_weights(langs, weight_matrix, rank_matrix):
    T = 5
    lang_final_weights = np.zeros(len(langs))

    for i in range(T):
        lang_weights = model_weights_for_one_round(langs, weight_matrix, rank_matrix)

        # choose language (best of this round) that has not been chosen
        # lang_order[k]=j means language j is the kth best (has kth largest weight)
        lang_order = np.argsort(lang_weights)[::-1]

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

    final = {lang: w+1 for lang, w in zip(langs, lang_final_weights)}

    return final


def learn_entity_specific_weights(target_lang, langs, df, e, num_entity, base):
    triples = triples_of_given_entity(df, e)
    if len(triples) <= 10:
        return {l:1 for l in langs}

    subdf = triples.copy().reset_index(drop=True)

    # get entity pair weights and prediction correctness
    subdf['weight_matrix'] = ''
    subdf['weight_matrix'] = subdf.apply(lambda row: weight_matrix_from_one_triple(row, num_entity), axis=1)
    subdf['rank_matrix'] = subdf.apply(lambda row: rank_matrix_from_one_triple(target_lang, row, num_entity, langs, base), axis=1)

    # concatenate matrices for multiple triples by axis=0
    big_weight_matrix = np.concatenate(subdf['weight_matrix'].values, axis=0)
    big_rank_matrix = np.concatenate(subdf['rank_matrix'].values, axis=0)
    # normalize datapoint weights
    big_weight_matrix = big_weight_matrix / np.sum(big_weight_matrix)

    model_weights = learn_model_weights(langs, big_weight_matrix, big_rank_matrix)
    return model_weights


def extract_entities(id_score_tuples):
    return [ent_id for ent_id, score in id_score_tuples]
