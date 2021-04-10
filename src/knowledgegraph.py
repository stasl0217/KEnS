from src.model import create_knowledge_model, create_alignment_model, create_vec_kNN_finder
from sklearn.model_selection import KFold
import numpy as np
import os
from os.path import join
import src.param as param
from tensorflow import keras
from src.modellist import AlignModel
import json

class KnowledgeGraph:
    def __init__(self, lang, kg_train_data, kg_val_data, dict0to1, dict1to0, num_entity, num_relation):
        self.lang = lang
        self.data = kg_train_data  # training set
        self.val_data = kg_val_data
        self.dict0to1 = dict0to1
        self.dict1to0 = dict1to0
        self.num_entity = num_entity
        self.num_relation = num_relation

        if self.dict0to1 is not None:  # supporter kg
            self.seed_links = self.sample_as_seed_links()  # 2-col np.array, entity0 - entity1

        # keras model
        self.model, _, self.kNN_finder = create_knowledge_model(self.num_entity, self.num_relation)
        if param.knowledge == 'transe':
            self.vec_kNN_finder = create_vec_kNN_finder(self.num_entity)
        self.align_model = None  # Used for target kg alignment validation. Value is always None for target kg.

        self.align_models_of_all = {}  # {'lang': alignment model}

        self.h_train, self.r_train, self.t_train = self.data[:,0], self.data[:,1], self.data[:,2]
        self.h_test, self.r_test, self.t_test = self.val_data[:,0], self.val_data[:,1], self.val_data[:,2]
        if self.is_supporter_kg():
            self.h_train = np.concatenate([self.h_train, self.h_test], axis=0) # full KG for supporter KG
            self.r_train = np.concatenate([self.r_train, self.r_test], axis=0)
            self.t_train = np.concatenate([self.t_train, self.t_test], axis=0)
        self.y_train = np.zeros(self.h_train.shape[0])  # supporter KG results don't count
        self.y_test = np.zeros(self.h_test.shape[0])


        # See get_filtered_reordered_embedding_matrix() for more information
        # This value is initialized as None. call the above funtion will assign value to it
        #    ( It's used in TEST MODE to avoid recomputing)
        self.filtered_reordered_embedding_matrix = None  # shape [1, num_entity_kg0 * dim] (flattened)

    def is_supporter_kg(self):
        if self.dict0to1 is not None:
            return True
        return False

    def build_alignment_models(self, all_kgs):
        """
        assign self.all_models_of_all and self.align_modelg
        :param all_kgs:
        :return:
        """
        for kg in all_kgs:
            if kg.lang != self.lang:  # not itself
                # build alignment model
                align = self.build_alignment_model(other_kg=kg)
                self.align_models_of_all[kg.lang] = align

        if self.lang != param.lang:  # not target kg
            self.align_model = self.align_models_of_all[param.lang]  # use target kg for testing


    def sample_as_seed_links(self):
        """
        Only called during initialization,
        used to assign self.seed_links (2-col np array, kg0 to kg1), seed alignment links used for training
        :return:
        """
        links = np.array(list(self.dict0to1.items()))
        np.random.shuffle(links)  # shuffle
        seed_size = int(len(links)*param.align_ratio)
        return links[:seed_size, :]


    def build_alignment_model(self, other_kg):
        # self.align_model = create_alignment_model(other_kg.model, self.model)
        return create_alignment_model(other_kg.model, self.model)

    def get_embedding_matrix(self):
        """
        return the embedding matrix FLATTENED AS ONE ROW
        """
        return np.squeeze(np.array(self.model.layers[3].get_weights())).reshape([1, -1])

    def get_relation_matrix(self):
        """
        return relation embedding as [n_rel, dim]
        """
        return self.model.layers[4].get_weights()[0]


    def get_filtered_reordered_embedding_matrix(self, num_entity0):
        """
        This should only be used when the KG is a supporter KG!
        Find entities aligned to target KG 1~N (number of entities in target KG0
        E[0,:] will be the vector corresponding to [entity 0 in the target KG]

        ===IMPORTANT
        So, if we find kNN in this embedding matrix, the index will directly be the entity index in KG0

        This function will also assign self.filtered_reordered_embedding_matrix, which was initialized as None.
        if param.updating_embedding=False (it's used in TEST MODE to avoid recomputing),
            self.filtered_reordered_embedding_matrix will never change after assigned the first time
        ( If wrongly set update_embedding=False during training, you will not see results change!! )
        :param num_entity0: number of entity in target kg. filtered kg shape: [num_entity0, dim]
        :output: assign self.filtered_reordered_embedding_matrix
        """
        if not self.is_supporter_kg():  # This should only be used when the KG is a supporter KG!
            print('WARNING: you are using get_filtered_reordered_embedding_matrix() '
                  'when this kg is not a supporter kg!!!'
                  'Original embedding matrix is used in this case.')
            return self.get_embedding_matrix()

        if not param.updating_embedding:
            # We're testing, not training. Embedding matrix will not change.
            if self.filtered_reordered_embedding_matrix is not None:  # computed before
                return self.filtered_reordered_embedding_matrix

        E1 = np.squeeze(np.array(self.model.layers[3].get_weights()))  # original embedding matrix. shape [num_entity1, dim]
        E0 = np.zeros([num_entity0, param.dim])  # shape [num_entity0, dim]
        # If an entity in target kg is not linked to the current kg, the vector stays default (all 0)
        for e0, e1 in self.dict0to1.items():
            E0[e0,:] = E1[e1,:]
        flattened = E0.reshape([1, -1])  # flatten for feed kNN_finder
        self.filtered_reordered_embedding_matrix = flattened
        return flattened


    def get_entity_vec(self, ent):
        ent_matrix = self.model.layers[3].get_weights()[0]
        return ent_matrix[ent, :]

    def get_relation_vec(self, rel):
        rel_matrix = self.model.layers[4].get_weights()[0]
        return rel_matrix[rel, :]


    def save_model(self, output_dir):
        """
        Save the trained knowledge model under output_dir. Filename: 'language.h5'
        """
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        # save the weights for the whole model
        filename_knowledge_model = '%s.h5'%self.lang
        keras.models.save_model(self.model, join(output_dir, filename_knowledge_model))
        # self.model.save(join(output_dir, filename_knowledge_model))

        # save model structure in json
        filename_json = '%s.json'%self.lang
        json_string = self.model.to_json()
        with open(join(output_dir,filename_json), 'w') as outfile:
            outfile.write(json_string)

    def load_model(self, model_dir):
        """
        Recreate the knowledge model and load the weights
        :param path:
        :return:
        """
        self.model.load_weights(join(model_dir, self.lang+'.h5'))

    def load_model_from_json(self, model_dir):
        file = join(model_dir, self.lang+'.vec.json')
        with open(file) as f:
            vecs = json.load(f)
            ent_embeddings = vecs["ent_embeddings"]
            rel_embeddings = vecs["rel_embeddings"]
            ent_embedding_layer = self.model.layers[3]
            rel_embedding_layer = self.model.layers[4]
            ent_embedding_layer.set_weights(np.array([ent_embeddings]))
            rel_embedding_layer.set_weights(np.array([rel_embeddings]))


    def populate_alignlinks(self, kg0, seed_alignlinks):
        self.filtered_reordered_embedding_matrix = None
        if (kg0.lang, self.lang) in seed_alignlinks:
            links = seed_alignlinks[(kg0.lang, self.lang)]
            for pair in links:
                e0, e1 = pair[0], pair[1]
                if e0 not in self.dict0to1 and e1 not in self.dict1to0:
                    self.dict0to1[e0] = e1
                    self.dict1to0[e1] = e0
        else:
            links = seed_alignlinks[(self.lang, kg0.lang)]
            for pair in links:
                e0, e1 = pair[1], pair[0]
                if e0 not in self.dict0to1 and e1 not in self.dict1to0:
                    self.dict0to1[e0] = e1
                    self.dict1to0[e1] = e0


def k_fold_split(triples, k=10):
    """
    :param triples: int np.array (n_triple, 3)
    :param k: k fold. default: 10
    :return: h_train, r_train, t_train, y_train, h_test, r_test, t_test.
            y_train is may be a zero vector, used to minimize the triple loss/maximize triple score
    """
    kf = KFold(n_splits=k, shuffle=True)
    kf.get_n_splits(triples)

    train_index, test_index = next(kf.split(triples))
    X_train, y_train = triples[train_index], np.zeros(
        len(train_index))  # y: distance should be as close to 0 as possible
    X_test = triples[test_index]

    h_train, r_train, t_train = X_train[:, 0], X_train[:, 1], X_train[:, 2]
    h_test, r_test, t_test = X_test[:, 0], X_test[:, 1], X_test[:, 2]

    return h_train, r_train, t_train, y_train, h_test, r_test, t_test