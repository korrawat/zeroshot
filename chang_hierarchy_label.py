import numpy as np 
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
print "Kritkorn is loading Glove"
main_word2vec = load_pre_trained_vector("glove.6B.50d.txt")
print "Kritkorn finished loading Glove"

# i = 0
# with open('map_wordnet.txt') as f:
#     while i < 5:
#         line = f.readline()
#         print line
#         i += 1

id_labels = {}
with open('map_wordnet.txt') as f:
    for line in f:
        id_labels[line[:9]] = line[10:-1].split(', ')

# i = 0
# with open('parent_child.txt') as f:
#     while i < 5:
#         line = f.readline()
#         print line.split()
#         i += 1

dic_child = {}
dic_parent = {}


def add_child(parent, child):
    temp_child = dic_child.get(parent, [])
    temp_child.append(child)
    dic_child[parent] = temp_child
    temp_parent = dic_parent.get(child, [])
    temp_parent.append(parent)
    dic_parent[child] = temp_parent

with open('parent_child.txt') as f:
    for line in f:
        parent = line.split()[0]
        child = line.split()[1]
        add_child(parent, child)

# synsets_1k = []
# with open('1k_synsets.txt') as f:
#     for line in f:
#         synsets_1k.append(line[:9])

def find_ancestor(node, level):
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

def find_descendant(node, level):
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
        

def hop(node, up, down):
    ancestors = find_ancestor(node, up)
    set_hop = set([])
    for ancestor in ancestors:
        for level in range(down+1):
            set_hop = set_hop.union(find_descendant(ancestor, level))
    if node in set_hop:
        set_hop.remove(node)
    return set_hop

def is_leaf(node):
    return node not in dic_child

def hop_dist(node, dist):
    set_hop = set([])
    for d in range(dist+1):
        for up in range(1,d+1):
            down = d - up
            ancestors = find_ancestor(node, up)
            for ancestor in ancestors:
                set_hop = set_hop.union(find_descendant(ancestor, down))
    return set_hop


# hop2 = set([])
# for synset in synsets_1k:
#     hop2 = hop2.union(hop_dist(synset, 2))
# hop2 -= set(synsets_1k)


# not_leaf2 = 0
# for synset in synsets_1k:
#     if not is_leaf(synset):
#         if find_descendant(synset, 3) != set([]):
#             not_leaf2 += 1
#             print id_labels[synset]
#             print [id_labels[x] for x in find_descendant(synset, 2)]


# num_leaf_full = 0
# set_dict_keys = set([])
# with open('map_wordnet.txt') as f:
#     for line in f:
#         synset = line[:9]
#         set_dict_keys.add(synset)
#         num_leaf_full += is_leaf(synset)


# set_all_synset = set(dic_child.keys()).union(set(dic_parent.keys()))


def hop_dist_simp(node, dist):
    set_hop = set([])
    for d in range(dist+1):
        for up in range(0,d+1):
            down = d - up
            ancestors = find_ancestor(node, up)
            for ancestor in ancestors:
                set_hop = set_hop.union(find_descendant(ancestor, down))
    return set((filter(lambda x: is_leaf(x), set_hop)))  


# hop2_simp = set([])
# for synset in synsets_1k:
#     hop2_simp = hop2_simp.union(hop_dist_simp(synset, 2))
# hop2_simp -= set(synsets_1k)

# set((filter(lambda x: is_leaf(x), hop2))).issubset(hop2_simp)


# hop_down_2 = set([])
# for synset in synsets_1k:
#     hop_down_2 = hop_down_2.union(find_descendant(synset, 2))
# len(set((filter(lambda x: is_leaf(x), hop_down_2))))

def valid_one_word(word):
    return (' ' not in word) and ('-' not in word) and (word in main_word2vec)


def is_one_word(synset):
    labels = id_labels[synset]
    res = 1
    for label in labels:
        if valid_one_word(label):
            res *= 0
        else:
            res *= 1
    return not res

def get_recent_oneword_ancestor(synset):
    level = 0
    while level >= 0:
        for ancestor in find_ancestor(synset, level):
            if is_one_word(ancestor):
                return ancestor
            level += 1


def get_one_word(synset):
    curr = get_recent_oneword_ancestor(synset)
    labels = id_labels[curr]
    for label in labels:
        if valid_one_word(label):
            return label, is_one_word(synset)


# get_recent_oneword_ancestor('n02343058')


# # In[327]:


# set((filter(lambda x: is_one_word(x), hop2_simp)))


# # In[340]:


# output = 'hop2.txt'
# f = open(output, "w+")
# for synset in set((filter(lambda x: is_one_word(x), hop2_simp))):
#     print >> f, synset
# f.close()


# # In[336]:


# for i in set((filter(lambda x: is_one_word(x), hop2_simp))):
#     if len(i) != 9:
#         print i

