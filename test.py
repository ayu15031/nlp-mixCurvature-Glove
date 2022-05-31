import torch
from glove import GloVeModel
# import run
import pickle
from tools import Dictionary
import argparse
import numpy as np


def test_glove_model(MODEL_PATH, DICT_PICKLE, CONFIG_FILE):

    config = pickle.load(open(CONFIG_FILE, 'rb'))

    model = GloVeModel(config['EMBEDDING_SIZE'], config['CONTEXT_SIZE'], config['VS'])
    model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')))
    model

    with open(DICT_PICKLE, mode='rb') as fp:
        dictionary = pickle.load(fp)

    emb = model._focal_embeddings.weight + model._context_embeddings.weight
    # emb = model['_focal_embeddings.weight'] + model['_context_embeddings.weight']

    emb = [e.detach().numpy() for e in emb]
    # print(emb[0].sum())
    # print(emb[1].sum())
    # print(emb[2].sum())

    GloVe = {}
    for w in dictionary.word2idx:
        GloVe[w] = emb[dictionary.word2idx[w]]
        
    # print(GloVe['hogwarts'].sum())
    # print(GloVe['dumbledore'].sum())
    # print(GloVe['voldemort'].sum())

    def cosine(a, b):
        return a.dot(b)/(np.linalg.norm(a)*np.linalg.norm(b))

    w1, w2 = 'harry', 'ron'
    print("cosine similarity between", w1, "and", w2, cosine(GloVe[w1], GloVe[w2]))
    # cosine(GloVe['hogwarts'], GloVe['harry'])

    #### Analogies ####



if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--type", type=str, default="vanilla")
    args = parser.parse_args()
    
    SEED = 6969
    # random.seed(SEED)
    # np.random.set_seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    np.random.seed(SEED)


    TYPE = args.type
    DICT_PICKLE = "data/dict_test_" + TYPE + ".pkl"

    print("Testing...")
    MODEL_PATH = "checkpoints/vanilla"
    CONFIG_PATH = "config_vanilla.pkl"

    test_glove_model(MODEL_PATH, DICT_PICKLE, CONFIG_PATH)


