import numpy as np
import sys
import heapq
import os
import timeit

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
# MAIN_WORD2VEC = load_pre_trained_vector("../glove.6B.50d.txt")
MAIN_WORD2VEC = None
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
        if valid_one_word(label, main_word2vec):
            res *= 0
        else:
            res *= 1
    return not res

# return synset which is the most recent ancestor that has one-word label
# ancestor including itself
def get_recent_oneword_ancestor(synset, dic_parent = DIC_PARENT, main_word2vec = MAIN_WORD2VEC, id_labels = ID_LABELS):
    level = 0
    while level >= 0:
        for ancestor in find_ancestor(synset, level, dic_parent):
            if is_one_word(ancestor, main_word2vec, id_labels):
                return ancestor
            level += 1

# return any one-word label from the most recent ancestor (including itself)
def get_one_word(synset, dic_parent = DIC_PARENT, main_word2vec = MAIN_WORD2VEC, id_labels = ID_LABELS):
    curr = get_recent_oneword_ancestor(synset, dic_parent, main_word2vec, id_labels)
    labels = id_labels[curr]
    for label in labels:
        if valid_one_word(label, main_word2vec):
            return label, is_one_word(synset, main_word2vec, id_labels)

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

def max_similarity_score_vec_id(vec, label_id, id_labels = ID_LABELS, main_word2vec = MAIN_WORD2VEC, dic_parent = DIC_PARENT, euclidean=False):
    max_similarity = -1000
    names = id_labels[label_id]
    for name in names:
        if valid_one_word(name, main_word2vec):
            word_embed_name = get_vec(name, main_word2vec)
            similarity = similarity_score(vec, word_embed_name,euclidean=euclidean)
            max_similarity = max(similarity, max_similarity)
    if max_similarity == -1000:
        max_similarity = similarity_score(get_vec(get_one_word(label_id, dic_parent, main_word2vec, id_labels)[0], main_word2vec), vec, euclidean=euclidean)
    return max_similarity

def create_mini_glove(glove_filename, mini_glove_output_path, id_labels = ID_LABELS):
    '''
    big_word2vec = load_pre_trained_vector(glove_filename)
    mini_dict = dict()
    for index in range(1000):
        word_1k = get_one_word(index_to_1k_id(index))
        mini_dict[word_1k] = big_word2vec[word_1k]
    synset_pool = list()
    with open(available_hop_filename) as f:
        for line in f:
            synset_pool.append(line[:-1])

    for synset in synset_pool:
        all_synset_labels = id_labels[synset]
        for word in all_synset_labels:
            if word in big_word2vec:
                mini_dict[word] = big_word2vec[word]

    '''
    big_word2vec = load_pre_trained_vector(glove_filename)
    mini_dict = dict()
    for synset_id in id_labels:
        described_words = id_labels[synset_id]
        for word in described_words:
            if word in big_word2vec:
                mini_dict[word] = big_word2vec[word]

    with open(mini_glove_output_path, 'w') as f:
        for word in mini_dict:
            word_embed = mini_dict[word]
            line_to_be_printed = word
            for value in word_embed:
                line_to_be_printed += " {0:.5f}".format(value)
            f.write(line_to_be_printed + "\n")





def get_synthetic_vec(probability_distribution, T, all_ids = ALL_IDS, dic_parent = DIC_PARENT, main_word2vec = MAIN_WORD2VEC, id_labels = ID_LABELS):
    '''
    Args:
        probability_distribution - a numpy array of size 1000 that represents the probability of being each label in 
                                   the training dataset
        T : the number of highest probability that we will take into account when constructing the synthetic word embedding
        
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

def new_get_synthetic_vec(probability_distribution, T, a,b,c, threshold, compared_vector, all_ids = ALL_IDS, dic_parent = DIC_PARENT, main_word2vec = MAIN_WORD2VEC, id_labels = ID_LABELS):
    sorted_indices = [i[0] for i in sorted(enumerate(-np.array(probability_distribution)), key=lambda x:x[1])]
    
    #highest_T_prediction = sorted_indices[:T]
    #highest_T_prediction = sorted_indices[:3*T/2]
    highest_T_prediction = sorted_indices[:a+b]

    T_similarity_score = [max_similarity_score_vec_id(compared_vector, \
                          index_to_1k_id(highest_T_prediction[i], all_ids), id_labels, main_word2vec, dic_parent, euclidean=False) for i in range(a,a+b)]
    the_middle_highest_similarity = sorted(T_similarity_score,reverse=True)[b-c]

    should_included = [True]*(a) + [T_similarity_score[i] <= the_middle_highest_similarity for i in range(b)]
    #highest_T_probability = np.array([probability_distribution[highest_T_prediction[i]] for i in range(T)])
    highest_T_probability = np.array([ probability_distribution[highest_T_prediction[i]]*(should_included[i]) \
                                         for i in range(a+b)])
    #highest_T_probability = np.array([ probability_distribution[highest_T_prediction[i]]*(T_similarity_score[i]>=threshold) \
    #                                     for i in range(T)])


    word_embedding_vectors = list()
    for training_id in highest_T_prediction:
        synset_id = index_to_1k_id(training_id, all_ids)
        one_word_rep = get_one_word(synset_id, dic_parent, main_word2vec, id_labels)[0]
        word_embedding_vectors.append(get_vec(one_word_rep, main_word2vec))
    word_embedding_vectors = np.array(word_embedding_vectors)
    normalize_factor = np.sum(highest_T_probability)
    synthetic_word_embedding_vector = np.dot(highest_T_probability/normalize_factor, word_embedding_vectors)
    return synthetic_word_embedding_vector

# TODO
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
        best_name_score = -1000
        for name in names:
            if valid_one_word(name, main_word2vec):
                word_embed_name = get_vec(name, main_word2vec)
                similarity = similarity_score(synthetic_vector, word_embed_name)
                if similarity > k_highest_similarity_labels[0][0]:
                    is_very_similar = True
                    candidate_similarity = max(similarity, candidate_similarity)
                    if similarity > best_name_score:
                        best_name_score = similarity
                        best_name = name
        if is_very_similar:
            heapq.heapreplace(k_highest_similarity_labels,(candidate_similarity, label_id, best_name))
    nearest_labels = list()
    k_highest_similarity_labels = sorted(k_highest_similarity_labels,key = lambda x : -x[0])
    for pair in k_highest_similarity_labels:
        nearest_labels.append((pair[1],pair[2],pair[0]))
    return nearest_labels


def nearest_neighbor_with_threshold(probability_distribution, top_k, label_pool, threshold, T, a, b, c, \
                                    all_ids = ALL_IDS, dic_parent = DIC_PARENT, main_word2vec = MAIN_WORD2VEC, id_labels = ID_LABELS):

    #synthetic_vector = get_synthetic_vec(probability_distribution, T, all_ids, dic_parent, main_word2vec, id_labels)
    synthetic_vector = get_synthetic_vec(probability_distribution, a, all_ids, dic_parent, main_word2vec, id_labels)
    #nearest_label_first_guess = nearest_neighbor(label_pool, synthetic_vector, 1, main_word2vec, id_labels)[0]
    #first_guess_id, first_guess_name, first_guess_score = nearest_label_first_guess
    
    #compared_vector = get_vec(first_guess_name, main_word2vec)
    compared_vector = synthetic_vector

    #the_T_th_value = sorted(probability_distribution,reverse = True)[T-1]

    #new_prob_dist = [probability_distribution[i] * (max_similarity_score_vec_id(get_vec(first_guess_name, main_word2vec), index_to_1k_id(i, all_ids), id_labels, main_word2vec, dic_parent) > threshold)\
    #    for i in range(len(probability_distribution))]
    #new_prob_dist = [probability_distribution[i] * (max_similarity_score_vec_id(get_vec(first_guess_name, main_word2vec), index_to_1k_id(i, all_ids), id_labels, main_word2vec, dic_parent) > threshold) * (probability_distribution[i] >= the_T_th_value) \
    #    for i in range(len(probability_distribution))]

    #if sum(new_prob_dist) == 0:
    #    return None

    #new_synthetic_vector = get_synthetic_vec(new_prob_dist, T, all_ids, dic_parent, main_word2vec, id_labels)
    #new_synthetic_vector = get_synthetic_vec(probability_distribution, T, all_ids, dic_parent, main_word2vec, id_labels)
    new_synthetic_vector = new_get_synthetic_vec(probability_distribution, T, a,b,c, threshold, compared_vector, all_ids, dic_parent, main_word2vec, id_labels)
    nearest_label_final_guesses = nearest_neighbor(label_pool, new_synthetic_vector, top_k, main_word2vec, id_labels)

    return nearest_label_final_guesses


def accuracy_one_synset(threshold, T, a, b, c, testing_wnid, probs_result_dir, words_result_dir,\
                        label_pool, overwrite=False, get_all_nns=False, top_k = 100, all_ids = ALL_IDS, dic_parent = DIC_PARENT,\
                        main_word2vec = MAIN_WORD2VEC, id_labels = ID_LABELS, num_sub_sample = 200):
    """
        probs_result_dir - contains n*/n*_*.txt, which contains 1k lines of 
        probabilities from CNN inference
    """
    probs_result_dir_synset = os.path.join(probs_result_dir, testing_wnid)
    words_result_rank_filename = os.path.join(words_result_dir, testing_wnid + '.txt')
    if not overwrite and os.path.exists(words_result_rank_filename):
        return (0, 0)

    count_total = 0
    count_correct = 0
    for probs_file in os.listdir(probs_result_dir_synset)[:num_sub_sample]:
        # print "Processing %s" % probs_file
        probability_distribution = np.loadtxt(os.path.join(probs_result_dir_synset, probs_file))

        probability_distribution = probability_distribution[1:]

        top_k_nns = top_k
        if get_all_nns:
            top_k_nns = len(label_pool)
        nns = nearest_neighbor_with_threshold(probability_distribution, top_k_nns, label_pool, threshold, T, a,b,c, all_ids, dic_parent, main_word2vec, id_labels)
        if nns is None:
            continue
        nn_ids = [x[0] for x in nns]

        count_total += 1
        if testing_wnid in nn_ids:
            rank = nn_ids.index(testing_wnid) # index starts from 0
            if rank < top_k:
                count_correct += 1
            with open(words_result_rank_filename, 'a') as f:
                f.write(probs_file + '\t' + str(rank) + '\n')

    return (count_correct, count_total)


def accuracy(threshold, T, a, b, c, testing_wnids, probs_result_dir, words_result_dir,\
            label_pool, error_log_file, output_log_file=None, debug=False, overwrite=False, get_all_nns=False, top_k = 100, all_ids = ALL_IDS, dic_parent = DIC_PARENT,\
            main_word2vec = MAIN_WORD2VEC, id_labels = ID_LABELS):
    # e.g. probs_result_dir = "/Volumes/Kritkorn/results", words_result_dir = "/Volumes/Kritkorn/words"
    count_total = 0
    count_correct = 0
    for testing_wnid in testing_wnids:
        start_time = timeit.default_timer()
        try:
            count_correct_set, count_total_set = accuracy_one_synset(threshold, T, a,b,c, testing_wnid, probs_result_dir, words_result_dir,\
                        label_pool, overwrite, get_all_nns, top_k, all_ids, dic_parent, main_word2vec, id_labels)

            count_total += count_total_set
            count_correct += count_correct_set
            if count_total_set != 0:
                accuracy_set = 1.0 * count_correct_set / count_total_set
            else:
                accuracy_set = 0

            end_time = timeit.default_timer()
            elapsed_time = end_time - start_time
            average_time = elapsed_time / count_total_set
            if debug:
                output_message = "wnid: %s\n" % testing_wnid
                output_message += "Time: %.3f s, avg time: %.3f s\n" % (elapsed_time, average_time)
                output_message += "Accuracy: %.3f, total: %d, top %d: %d\n" %\
                    (accuracy_set, count_total_set, top_k, count_correct_set)
                print output_message
                with open(output_log_file, 'a') as f:
                    f.write(output_message)

        except Exception as e:
            with open(error_log_file, "a") as f:
                f.write(testing_wnid + "\n")
                f.write(str(e) + "\n")
            continue
    return (count_correct, count_total)

