import geoopt
import numpy as np
from enum import Enum
import geoopt.manifolds.stereographic.math as mobmath
import torch
# class Embedding(Enum):
#     EUCLEDIAN = 1
#     SPHERICAL = 2
#     HYPERBOLIC = 3

def find_distance(u, v, k):
    return mobmath.dist(u, v, k=k)

def load_analogy_words():
    word_analogies = []
    with open("data/analogies.txt", 'r') as f:
        for line in f:
            analogy = line.split(" ")
            if len(analogy) == 4:
                word_analogies.append(analogy)

    return word_analogies

def find_cosine_similarity(u, v):
    distance = 0.0
    dot = np.dot(u,v)
    norm_u = np.sqrt(np.sum(u**2))
    norm_v = np.sqrt(np.sum(v**2))
    distance = dot/(norm_u)/norm_v
    return distance

def load_vectors(glove_file):
    pass

    # with open(glove_file, 'r', encoding="utf-8") as file:
    #     words = set()
    #     word_to_vec = {}
    #     for line in file:
    #         line = line.strip().split()
    #         curr_word = line[0]
    #         words.add(curr_word)
    #         word_to_vec[curr_word] = np.array(line[1:], dtype=np.float64)
    # return words, word_to_vec

def find_analogy_glove(word_a, word_b, word_c, embeddings, k):
    word_a = word_a.lower()
    word_b = word_b.lower()
    word_c = word_c.lower()
    
    e_a, e_b, e_c = embeddings[word_a], embeddings[word_b], embeddings[word_c]
    
    words = embeddings.keys()
    max_cosine_sim = -999
    best_word = None
    
    for w in words:
        if w in [word_a, word_b, word_c]:
            continue
        cosine_sim = find_cosine_similarity(e_b - e_a, embeddings[w] - e_c)
        
        if cosine_sim > max_cosine_sim:
            max_cosine_sim = cosine_sim
            best_word = w
            
    return best_word


def find_analogy_mix(word_a, word_b, word_c, embeddings_list, curvatures, weights):
    word_a = word_a.lower()
    word_b = word_b.lower()
    word_c = word_c.lower()
    
    final = {}
    words = embeddings_list[0].keys()
    word_dist = []
    parallen_transes = []
    for i in range(len(curvatures)):

        e_a, e_b, e_c = embeddings_list[i][word_a], embeddings_list[i][word_b], embeddings_list[i][word_c]

        parallen_trans = mobmath.mobius_add(e_c, mobmath.gyration(e_c, -e_a, mobmath.mobius_add(-e_a, e_b, k=curvatures[i]), k=curvatures[i]), k=curvatures[i])
        parallen_transes.append(parallen_trans)

    for w in words:
        
        if w in [word_a, word_b, word_c]:
                continue

        dist = 0.0
    
        for i, embeddings in enumerate(embeddings_list):
            e_a, e_b, e_c = embeddings[word_a], embeddings[word_b], embeddings[word_c]

            c = curvatures[i]
            wt = weights[i]
            parallen_trans = parallen_transes[i]

            # cosine_sim = find_distance(embeddings[w], mobmath.mobius_add(e_c, )   ()find_distance(e_b, e_a), find_distance(, e_c))
            # parallen_trans = self.moebius_add_mat(pos_emb[1], self.gyr_mat(pos_emb[1], -neg_emb, self.moebius_add_mat(-neg_emb, pos_emb[0])))

            distance = find_distance(embeddings[w], parallen_trans, c)
            dist += wt * distance
            # if distance > max_cosine_sim:
                # max_cosine_sim = distance
                # best_word = w
            
        word_dist.append((dist, w))

    word_dist = sorted(word_dist)
    final = word_dist[:10]
    return final

def find_analogy(path ,type="glove"):
    # _, embeddings = load_vectors(path)
    # words_list = load_analogy_words()
    

    # for words in words_list:
    #     if type == "glove":
    #         res = find_analogy_glove(words[0], words[1], words[2])
    #     else:
    #         res = find_analogy_mix(words[0], words[1], words[2])
        
    pass

if __name__ == "__main__":
    # find_cosine_similarity(1, 1)
    embedding_list = []
    for i in range(3):
        embeddings = {"king": torch.rand(20), "queen": torch.rand(20), "man": torch.rand(20), "women": torch.rand(20), "a": torch.rand(20), "women2": torch.rand(20), "women3": torch.rand(20)}
        embedding_list.append(embeddings)

    c = [torch.tensor(0.5), torch.tensor(-0.5), torch.tensor(0)]

    print(find_analogy("king", "queen", "man", embedding_list, c, [0.33, 0.33, 0.33]))



