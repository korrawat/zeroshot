import numpy as np
import sys
sys.path.insert(0, '../models/research/slim/')
from datasets import imagenet
import heapq
import os

# download glove vectors
# return dictionary mapping word to vector
def load_pre_trained_vector(weight_filename):
    words = []
    vectors = []
    for l in open(weight_filename):
        t = l.strip().split()
        words.append(t[0])
        vectors.append(list(map(float, t[1:])))
    wordvecs = np.array(vectors, dtype=np.double)
    word2id = {word:i for i, word in enumerate(words)}
    word2vec = {word:vectors[i] for i, word in enumerate(words)}
    return word2vec

# adding parent to dic_parent[child], and child to dic_child[parent]
# mutate dic_parent and dic_child
# return None
def add_child(parent, child, dic_parent = {}, dic_child = {}):
    temp_child = dic_child.get(parent, [])
    temp_child.append(child)
    dic_child[parent] = temp_child
    temp_parent = dic_parent.get(child, [])
    temp_parent.append(parent)
    dic_parent[child] = temp_parent

# ====================================================================
# loading dictionary mapping word to vector
print "Loading Glove..."
MAIN_WORD2VEC = load_pre_trained_vector("glove.6B.50d.txt")
print "Finished loading Glove"

# create dictionary mapping synset to list of labels
ID_LABELS = {}
with open('map_wordnet.txt') as f:
    for line in f:
        ID_LABELS[line[:9]] = line[10:-1].split(', ')

# create dictionary mapping each synset to its children and parents
DIC_CHILD = {}
DIC_PARENT = {}
with open('parent_child.txt') as f:
    for line in f:
        parent = line.split()[0]
        child = line.split()[1]
        add_child(parent, child, DIC_PARENT, DIC_CHILD) 
# ====================================================================

# return set of ancestors at exactly level levels above
def find_ancestor(node, level, dic_parent = DIC_PARENT):
    if level == 0:
        return set([node])
    else:
        ancestors = set([])
        if node not in dic_parent:
            return set([])
        parents = dic_parent[node]
        for parent in parents:
            ancestors = ancestors.union(find_ancestor(parent, level-1))
    return ancestors

# return set of descendants at exactly level levels below
def find_descendant(node, level, dic_child = DIC_CHILD):
    if level == 0:
        return set([node])
    else:
        descendants = set([])
        if node not in dic_child:
            return set([])
        children = dic_child[node]
        for child in children:
            descendants = descendants.union(find_descendant(child, level-1))
    return descendants

# return set of words from going up and down (up = up, down = down) without itself
def hop(node, up, down, dic_parent = DIC_PARENT, dic_child = DIC_CHILD):
    ancestors = find_ancestor(node, up, dic_parent)
    set_hop = set([])
    for ancestor in ancestors:
        set_hop = set_hop.union(find_descendant(ancestor, level, dic_child))
    if node in set_hop:
        set_hop.remove(node)
    return set_hop

# return True if node is a leaf
def is_leaf(node, dic_child = DIC_CHILD):
    return node not in dic_child

# return True if word is one word without '-'
def valid_one_word(word, main_word2vec = MAIN_WORD2VEC):
    return (' ' not in word) and ('-' not in word) and (word in main_word2vec)

# return True if words of synset contains one word label
def is_one_word(synset, main_word2vec = MAIN_WORD2VEC, id_labels = ID_LABELS):
    labels = id_labels[synset]
    res = 1
    for label in labels:
        if valid_one_word(label, MAIN_WORD2VEC):
            res *= 0
        else:
            res *= 1
    return not res

# return synset which is the most recent ancestor that has one-word label
# ancestor including itself
def get_recent_oneword_ancestor(synset, dic_parent = DIC_PARENT, main_word2vec = MAIN_WORD2VEC):
    level = 0
    while level >= 0:
        for ancestor in find_ancestor(synset, level, dic_parent):
            if is_one_word(ancestor, main_word2vec):
                return ancestor
            level += 1

# return any one-word label from the most recent ancestor (including itself)
def get_one_word(synset, dic_parent = DIC_PARENT, main_word2vec = MAIN_WORD2VEC, id_labels = ID_LABELS):
    curr = get_recent_oneword_ancestor(synset, dic_parent, main_word2vec)
    labels = id_labels[curr]
    for label in labels:
        if valid_one_word(label, main_word2vec):
            return label, is_one_word(synset, main_word2vec)

# return set of words having distance dist from node
def hop_dist(node, dist, start_at_1 = False, dic_parent = DIC_PARENT, dic_child = DIC_CHILD):
    start_up = start_at_1
    set_hop = set([])
    for d in range(dist+1):
        for up in range(start_up, d+1):
            down = d - up
            ancestors = find_ancestor(node, up, dic_parent)
            for ancestor in ancestors:
                set_hop = set_hop.union(find_descendant(ancestor, down, dic_child))
    return set_hop

# ==== word2Vec & model2 ====================================================================

with open("1k_synsets.txt", "r") as file:
    ALL_IDS = file.read().split()

def index_to_1k_id(index, all_ids = ALL_IDS):
    return all_ids[index]

def similarity_score(this, that, normalized = True, euclidean=False):
    """
    Args:
        this - a numpy array representing the first vector 
        that - a numpy array representing the second vector 
        normalized - indicates if the score needs to be normalized
    """
    if euclidean:
        score = - np.linalg.norm(this - that)
        return score
    if normalized:
        score = np.dot(this, that)/(np.linalg.norm(this)*np.linalg.norm(that))
    else:
        score = np.dot(this, that)
    return score

def get_vec(word, model_dict = MAIN_WORD2VEC):
    try:
        return np.array(model_dict[word])
    except KeyError:
        print "label \"{0}\" is not in the vocabulary list".format(word)
        raise KeyError

def max_similarity_score_vec_id(vec, label_id, id_labels = ID_LABELS, main_word2vec = MAIN_WORD2VEC):
    max_similarity = -1000
    names = id_labels[label_id]
    for name in names:
        if valid_one_word(name, main_word2vec):
            word_embed_name = get_vec(name, main_word2vec)
            similarity = similarity_score(vec, word_embed_name)
            max_similarity = max(similarity, max_similarity)
    if max_similarity == -1000:
        max_similarity = similarity_score(get_vec(get_one_word(label_id, main_word2vec)[0]), vec)
    return max_similarity



def get_synthetic_vec(probability_distribution, T, all_ids = ALL_IDS, dic_parent = DIC_PARENT, main_word2vec = MAIN_WORD2VEC, id_labels = ID_LABELS):
    '''
    Args:
        probability_distribution - a numpy array of size 1000 that represents the probability of being each label in 
                                   the training dataset
        T : the number of highest probability that we will take into account when contructing the synthetic word embedding
        
        # not using# id - the id of the test image
        # not using # dict_id_to_prob_dist_from_CNN - the dictionary that map image's id to the dictionary of probability distribution
    '''
    #probability_distribution = dict_id_to_prob_dist_from_CNN[image_id]

    # sorted_inds = [ind[0] for ind in sorted(enumerate(-probabilities), key=lambda x:x[1])]
    sorted_indices = [i[0] for i in sorted(enumerate(-np.array(probability_distribution)), key=lambda x:x[1])]
    
    highest_T_prediction = sorted_indices[:T]
    highest_T_probability = np.array([probability_distribution[highest_T_prediction[i]] for i in range(T)])
    word_embedding_vectors = list()
    for training_id in highest_T_prediction:
        synset_id = index_to_1k_id(training_id, all_ids)
        one_word_rep = get_one_word(synset_id, dic_parent, main_word2vec, id_labels)[0]
        word_embedding_vectors.append(get_vec(one_word_rep, main_word2vec))
    word_embedding_vectors = np.array(word_embedding_vectors)
    normalize_factor = np.sum(highest_T_probability)
    synthetic_word_embedding_vector = np.dot(highest_T_probability/normalize_factor, word_embedding_vectors)
    return synthetic_word_embedding_vector

def nearest_neighbor(label_pool, synthetic_vector, k, main_word2vec = MAIN_WORD2VEC, id_labels = ID_LABELS):
    """
    Args:
        label pool - a list of synset id from which we want to predict; each synset id must have valid-one-word
        synthetic_vector - a numpy array representing the word embedding of an image
        k - we will return the synset ids that have k highest similarity
    """
    k_highest_similarity_labels = [(-1000,None,None)]*k
    heapq.heapify(k_highest_similarity_labels)
    for label_id in label_pool:
        names = id_labels[label_id]
        is_very_similar = False
        candidate_similarity = -1000
        for name in names:
            if valid_one_word(name, main_word2vec):
                word_embed_name = get_vec(name, main_word2vec)
                similarity = similarity_score(synthetic_vector, word_embed_name)
                if similarity > k_highest_similarity_labels[0][0]:
                    is_very_similar = True
                    candidate_similarity = max(similarity, candidate_similarity)
                    best_name = name
        if is_very_similar:
            heapq.heapreplace(k_highest_similarity_labels,(candidate_similarity, label_id, best_name))
    nearest_labels = list()
    for pair in k_highest_similarity_labels:
        nearest_labels.append((pair[1],pair[2],pair[0]))
    return nearest_labels


def nearest_neighbor_with_threshold(probability_distribution, top_k, label_pool, threshold, T=10,\
                                    all_ids = ALL_IDS, dic_parent = DIC_PARENT, main_word2vec = MAIN_WORD2VEC, id_labels = ID_LABELS):

    synthetic_vector = get_synthetic_vec(probability_distribution, T, all_ids, dic_parent, main_word2vec, id_labels)
    nearest_label_first_guess = nearest_neighbor(label_pool, synthetic_vector, 1, main_word2vec, id_labels)[0]
    first_guess_id, first_guess_name, first_guess_score = nearest_label_first_guess
    

    new_prob_dist = [probability_distribution[i] * (max_similarity_score_vec_id(get_vec(first_guess_name, main_word2vec), index_to_1k_id(i, all_ids), id_labels, main_word2vec) > threshold)\
        for i in range(len(probability_distribution))]
    if sum(new_prob_dist) == 0:
        return None

    new_synthetic_vector = get_synthetic_vec(new_prob_dist, T, all_ids, dic_parent, main_word2vec, id_labels)
    nearest_label_final_guesses = nearest_neighbor(label_pool, new_synthetic_vector, top_k, main_word2vec, id_labels)

    return nearest_label_final_guesses


def accuracy(threshold, testing_file, top_k = 100, all_ids = ALL_IDS, dic_parent = DIC_PARENT, main_word2vec = MAIN_WORD2VEC, id_labels = ID_LABELS,\
            probs_result_dir, words_result_dir):
    # e.g. testing_file = "available_hop2.txt", probs_result_dir = "/Volumes/Kritkorn/results", words_result_dir = "/Volumes/Kritkorn/words"
    with open(testing_file,"r") as testing_synsets:
        hop2_synset_ids = testing_synsets.read().split()
    label_pool = hop2_synset_ids

    count_total = 0
    count_correct = 0
    for probs_file in os.listdir(probs_result_dir):
        probability_distribution = np.loadtxt(os.path.join(probs_result_dir, probs_file))

        probability_distribution = probability_distribution[1:]

        nns = nearest_neighbor_with_threshold(probability_distribution, top_k, label_pool, threshold, all_ids, dic_parent, main_word2vec, id_labels)
        if nns is None:
            continue
        nn_ids = [x[0] for x in nns]

        count_total += 1
        if testing_wnid in nn_ids:
            count_correct += 1

    return (count_correct, count_total)











