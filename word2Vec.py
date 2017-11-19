
# coding: utf-8

# In[47]:

import os
import gzip
import numpy as np
import gensim
from chang_hierarchy_label import get_one_word, id_labels, valid_one_word, main_word2vec


# In[2]:


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


# In[78]:


# print len(main_word2vec)


# In[3]:


def check_availibility(word_list):
    not_in_vocab = list()
    for word in word_list:
        try:
            x = get_vec(word)
        except:
            #print "not all words in the list of words are in the vocavulary list"
            not_in_vocab.append(word)
    return not_in_vocab


# In[3]:


# Load Google's pre-trained Word2Vec model.
# google_model = gensim.models.KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin.gz', binary=True)  
# def get_google_word2vec(model):
#     google_word2vec = dict()
#     vocab = model.vocab.keys()
#     for word in vocab:
#         google_word2vec[word.lower()] = model.wv[word]
#     return google_word2vec
# google_word2vec = get_google_word2vec(google_model)


# In[31]:


def get_vec(word, model_dict = main_word2vec):
    try:
        return np.array(model_dict[word])
    except KeyError:
        print "label \"{0}\" is not in the vocabulary list".format(word)
        raise KeyError


# In[28]:


def similarity_score(this, that, normalized = True):
    """
    Args:
        this - a numpy array representing the first vector 
        that - a numpy array representing the second vector 
        normalized - indicates if the score needs to be normalized
    """
    if normalized:
        score = np.dot(this, that)/(np.linalg.norm(this)*np.linalg.norm(that))
    else:
        score = np.dot(this, that)
    return score


# In[9]:


def nearest(vec, ve):
    vnorm = norm(vec)
    scores = []
    for i in range(len(words)):
        wvnorm = norm(wordvecs[i])
        if not dot:
            scores.append(np.dot(wordvecs[i], vec) / (vnorm * wvnorm))
        else:
            scores.append(np.dot(wordvecs[i], vec))
    score_ids = [(s, i) for i, s in enumerate(scores)]
    score_ids.sort()
    score_ids.reverse()
    return score_ids


# In[30]:


# print similarity_score("mit","harvard")


# In[18]:


# print get_vec("newyork")


# In[19]:


# print google_word2vec["white_rabbit"]


# In[10]:


#Not using
def count_available_synset(synset_dict):
    count = 0
    available_synset = list()
    for synset in synset_dict:
        labels = synset_dict[synset]
        available = False
        for label in labels:
            if label in google_word2vec:
                available = True
        if available:
            available_synset.append(synset)
            count+=1
    return count


# In[11]:


#Not using
def txtToDict(labelFile):
    idToNames = {}
    with open(labelFile) as f:
        for line in f:
            i, namesString = line.split(':')
            namesList = namesString[:-1].split(', ')
            namesList = ['_'.join(name.split()) for name in namesList]
            if i != '0':
                idToNames[int(i)] = namesList
    return idToNames


# In[14]:


with open("1k_synsets.txt","r") as file:
    all_ids = file.read().split()
def index_to_1k_id(index):
    return all_ids[index]


# In[20]:


# index_to_1k_id(221)


# In[20]:


# dict_id_to_name_list
# dict_id_to_images
# dict_id_to_prob_dist_from_CNN
# CNN model


# In[25]:


def get_synthetic_vec(probability_distribution, T):
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
        synset_id = index_to_1k_id(training_id)
        one_word_rep = get_one_word(synset_id)[0]
        word_embedding_vectors.append(get_vec(one_word_rep))
    word_embedding_vectors = np.array(word_embedding_vectors)
    normalize_factor = np.sum(highest_T_probability)
    synthetic_word_embedding_vector = np.dot(highest_T_probability/normalize_factor, word_embedding_vectors)
    return synthetic_word_embedding_vector
    


# In[33]:


# probability_distribution = np.array([0.001]*1000)
# get_synthetic_vec(probability_distribution, 10)


# In[43]:


def nearest_neighbor(label_pool, synthetic_vector, k=1):
    """
    Args:
        label pool - a list of synset id from which we want to predict; each synset id must have valid-one-word
        synthetic_vector - a numpy array representing the word embedding of an image
        k - we will return the synset ids that have k highest similarity -> will implement this later for performance reason
    """
    nearest_label = None
    highest_similarity = -10000
    for label_id in label_pool:
        names = id_labels[label_id]
        for name in names:
            if valid_one_word(name):
                word_embed_name = get_vec(name)
                similarity = similarity_score(synthetic_vector, word_embed_name)
                if similarity > highest_similarity:
                    nearest_label = label_id
    return [nearest_label]
    


# In[36]:


# word_list = list()
# with open("1k_synsets.txt","r") as training_synsets:
#     synset_ids = training_synsets.read().split()
# for synset_id in synset_ids:
#     if get_one_word(synset_id)[0] == "entity":
#         print "Emergency"
#     word_list.append(get_one_word(synset_id)[0])
# len(check_availibility(word_list))


# In[44]:


# word_list = list()

    
# probability_distribution = np.array([0.001]*1000)
# synthetic_vec = get_synthetic_vec(probability_distribution, 10)

# id_labels[nearest_neighbor(synset_ids, synthetic_vec,1)[0]] # wait what? caterpillar and cat?? 555


# In[12]:


# id_labels["n02102480"]


# In[32]:


# type(get_vec("spaniel"))


# In[46]:


# Modify this one as you like
def get_accuracy(prob_dist_list, synset_id_list, label_pool, k_hit = 1):
    """
    Args:
        prob_dist_list - a list of prop dist of testing images
        synset_id_list - a list of true synset id corresponded to the prop dist in 'prop_dist_list' one by one
        label_pool - a list of synset id from which we want to predict; each synset id must have valid-one-word
        k_hit - we will consider correct if the first 'k_hit' predicted labels contain the true synset id
    """
    num_testing = len(prob_dist_list)
    num_correct = 0.0
    for index in range(num_testing):
        prob_dist = prob_dist_list[index]
        true_label = synset_id_list[index]
        first_k_hit = nearest_neighbor(label_pool, prob_dist, k_hit) #TODO wrong input?
        if true_label in first_k_hit:
            num_correct += 1
    return (num_correct/num_testing)*100.0


def nearest_neighbor_with_threshold(probability_distribution, top_k, label_pool, threshold):

    synthetic_vector = get_synthetic_vec(probability_distribution, top_k)
    nearest_label_first_guess = nearest_neighbor(label_pool, synthetic_vector, 1)


    new_prob_dist = [probability_distribution[i] * (max_similarity_score_vec_id(synthetic_vector, index_to_1k_id(i)) > threshold)\
        for i in range(len(probability_distribution))]

    new_synthetic_vector = get_synthetic_vec(new_prob_dist, top_k)
    nearest_label_final_guess = nearest_neighbor(label_pool, new_synthetic_vector, 1)
    print similarity_score(new_synthetic_vector, nearest_label_final_guess)
    return nearest_label_final_guess

def max_similarity_score_vec_id(vec, label_id):
    max_similarity = -1000
    names = id_labels[label_id]
    for name in names:
        if valid_one_word(name):
            word_embed_name = get_vec(name)
            similarity = similarity_score(vec, word_embed_name)
            max_similarity = max(similarity, max_similarity)
    return max_similarity


if __name__ == '__main__':
    with open("available_hop2.txt","r") as testing_synsets:
        hop2_synset_ids = testing_synsets.read().split()
    label_pool = hop2_synset_ids

    probs_result_dir = "/Volumes/Kritkorn/results"
    words_result_dir = '/Volumes/Kritkorn/words'
    for probs_file in os.listdir(probs_result_dir):
        if not probs_file.startswith('n02925666'): # buskin
            continue

        probability_distribution = np.loadtxt(os.path.join(probs_result_dir, probs_file))

        threshold = 0
        nn_id = nearest_neighbor_with_threshold(probability_distribution[1:], 10, label_pool, threshold)
        print "%d %s" % (nn_id, id_labels[nn_id])

        # print synthetic_vector

    # get_accuracy()
