from os.path import join
import pandas as pd
import numpy as np
import os
from src.knowledgegraph import KnowledgeGraph
import numpy as np

def load_data(data_dir, language, testfile_suffix=None):
    """
    :return: triples (n_triple, 3) np.int np.array
    :param testfile_suffix: '-val.tsv' or '-test.tsv'. Default '-val.tsv'
    """
    if testfile_suffix is None:
        testfile_suffix = '-val.tsv'

    train_df = pd.read_csv(join(data_dir, language + '-train.tsv'), sep='\t', header=None, names=['v1', 'relation', 'v2'])
    val_df = pd.read_csv(join(data_dir, language + testfile_suffix), sep='\t', header=None, names=['v1', 'relation', 'v2'])

    # count entity from entity list, not from training set, just in case training set does not include all entities
    entity_list = []
    with open(join(os.path.dirname(data_dir),'entity', language+'.tsv'),encoding='utf-8') as f:
        for line in f:
            ent = line.rstrip()
            if ent:
                entity_list.append(ent)
    # entity_list = set(train_df['v1'].values) | set(train_df['v2'].values)

    parent_dir = os.path.dirname(data_dir)
    relation_list = [line.rstrip() for line in open(join(parent_dir, 'relations.txt'))]
    triples_train = train_df.values.astype(np.int)
    triples_val = val_df.values.astype(np.int)
    return triples_train, triples_val, entity_list, relation_list


def load_align_links(link_dir, target_lang, src_lang):
    """
    :return: two column int np.array. 1st col: id in the target_lang. 2nd col: id in the src_lang
    """
    file1 = '%s-%s.tsv' % (src_lang, target_lang)
    file2 = '%s-%s.tsv' % (target_lang, src_lang)

    if os.path.exists(join(link_dir, file1)):
        path = join(link_dir, file1)
        source2target_link_df = pd.read_csv(path, sep='\t', header=None)
        source2target = source2target_link_df.values.astype(np.int)
        # swap the two columns
        target2source = np.ones(source2target.shape, dtype=np.int)
        target2source[:, 0], target2source[:, 1] = source2target[:, 1], source2target[:, 0]
    elif os.path.exists(join(link_dir, file2)):
        path = join(link_dir, file2)
        target2source = pd.read_csv(path, sep='\t', header=None).values.astype(np.int)
    else:
        raise FileNotFoundError('Cannot find alignment file %s or %s.'%(file1, file2))
    return target2source




def load_support_kgs(data_dir, align_dir, target_lang, src_langs, testfile_suffix=None):
    """
    :param target_lang: e.g., 'ja'
    :param src_langs: e.g., ['en', 'de', 'es']
    :param testfile_suffix: '-val.tsv' or '-test.tsv'
    :return: support_kgs list[SuppoterKG]
    """
    supporter_kgs = []

    for lang in src_langs:
        kg1_train_data, kg1_val_data, entity_list1, relation_list1 = load_data(data_dir,lang, testfile_suffix)  # use suffix 1 for supporter kg, 0 for target kg

        # two column np.array. 1st col: target lang. 2nd: source
        align_links = load_align_links(align_dir, target_lang, lang)
        align_dict_0to1 = {target2src[0]: target2src[1] for target2src in align_links}
        align_dict_1to0 = {target2src[1]: target2src[0] for target2src in align_links}

        supporter = KnowledgeGraph(lang, kg1_train_data, kg1_val_data, align_dict_0to1, align_dict_1to0, len(entity_list1), len(relation_list1))
        supporter_kgs.append(supporter)
    return supporter_kgs

def load_target_kg(data_dir, target_lang, testfile_suffix=None):
    """
    :param testfile_suffix: '-val.tsv' or '-test.tsv'
    :return:
    """
    kg_train_data, kg_val_data, entity_list1, relation_list1 = load_data(data_dir, target_lang, testfile_suffix)  # use suffix 1 for supporter kg, 0 for target kg
    kg = KnowledgeGraph(target_lang, kg_train_data, kg_val_data, None, None, len(entity_list1), len(relation_list1))
    return kg

def load_align_link_table(filepath):
    """
    load align links for all languages
    :param filepath
    :return:
    """
    df = pd.read_csv(filepath, sep='\t')  # header [en, ja, ...] (all languages)
    return df

def take_seed_align_links(link_df, lang1, lang2, seed_ratio):
    """
    :return:
    """
    links = link_df[[lang1, lang2]].dropna().values.astype(np.int)
    # entity size
    entity_size = min(link_df[lang1].count(), link_df[lang2].count())  # choose the min one between two kg entity sizes

    # np.random.shuffle(links)
    return links[:int(entity_size*seed_ratio),:]

def load_all_to_all_seed_align_links(align_dir):
    seeds = {}  # { (lang1, lang2): 2-col np.array }
    for f in os.listdir(align_dir):  # e.g. 'el-en.tsv'
        lang1 = f[0:2]
        lang2 = f[3:5]
        links = pd.read_csv(join(align_dir, f), sep='\t').values.astype(int)

        seeds[(lang1, lang2)] = links
    return seeds
