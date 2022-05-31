import os.path
import zipfile
import logging
import pickle
import torch
from glove import GloVeModel, GloVeMixedCurvature
from tools import SpacyTokenizer, Dictionary
import numpy as np

# import debugpy
# debugpy.listen(5951)
# print("Waiting for debugger")
# debugpy.wait_for_client()
# print("Attached! :)")


logging.basicConfig(
    format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

FILE_PATH = 'data/hp.txt'
# FILE_PATH = 'data/book_clean.txt'
COMATRIX_PATH = './data/comat.pickle'
LANG = 'en_core_web_sm'
EMBEDDING_SIZE = 128
CONTEXT_SIZE = 3 #TODO CHANGE
NUM_EPOCH = 1
BATHC_SIZE = 128
LEARNING_RATE = 2e-4


def read_data(file_path, type='file'):
    """ Read data into a string

    Args:
        file_path (str): path for the data file
    """
    text = None
    if type == 'file':
        with open(file_path, mode='r', encoding='utf-8') as fp:
            text = fp.read()
    elif type == 'zip':
        with zipfile.ZipFile(file_path) as fp:
            text = fp.read(fp.namelist()[0]).decode()
    return text




def preprocess(FILE_PATH, CORPUS_PICKLE=None):
    """ Get corpus and vocab_size from raw text

    Args:
        file_path (str): raw file path

    Returns:
        corpus (list): list of idx words
        vocab_size (int): vocabulary size
    """

    # preprocess read raw text
    
    dictionary = Dictionary()

    # if(not os.path.isfile(CORPUS_PICKLE)):
    if(True):
        print("PREPROCESSING AND CREATING DATA DICT...")
        text = read_data(FILE_PATH, type='file')
        logging.info("read raw data")

        # init base model
        tokenizer = SpacyTokenizer(LANG)
        logging.info("loaded tokenizers")

        # build corpus
        doc = tokenizer.tokenize(text)
        logging.info("tokens generated")

        # save doc
        with open(CORPUS_PICKLE, mode='wb') as fp:
            pickle.dump(doc, fp)
        logging.info("tokenized documents saved!")

    else:
        print(f"FOUND THE DICT: {CORPUS_PICKLE}")

    # load doc
    # with open(CORPUS_PICKLE, 'rb') as fp:
    #     doc = pickle.load(fp)

    print(len(doc), list(doc)[1][0:10])

    dictionary.update(doc)
    logging.info("after generate dictionary")
    with open(DICT_PICKLE, mode='wb') as fp:
        pickle.dump(dictionary, fp)

    print("="*20, dictionary.word2idx["hogwarts"])
    corpus = dictionary.corpus(doc)
    vocab_size = dictionary.vocab_size

    return corpus, vocab_size


def train_glove_model(TYPE, FILE_PATH, CORPUS_PICKLE, CONFIG_FILE, resume=False, expt_name="default"):
    
    if(TYPE == "mixed"):
        MODEL_PATH = "model/gloveMixed.pt"
    else:
        MODEL_PATH = "model/glove.pt"

    # preprocess
    corpus, vocab_size = preprocess(FILE_PATH, CORPUS_PICKLE)

    # specify device type
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # init vector model
    logging.info("init model hyperparameter")
    logging.info("Saving config file...")

    config = {
        "EMBEDDING_SIZE": EMBEDDING_SIZE,
        "CONTEXT_SIZE": CONTEXT_SIZE,
        "VS": vocab_size 
    }

    with open(CONFIG_FILE, "wb") as fp:
        pickle.dump(config, fp)


    if TYPE=="vanilla":
        model = GloVeModel(EMBEDDING_SIZE, CONTEXT_SIZE, vocab_size, resume=resume, expt_name=expt_name)
    elif TYPE=="mixed":
        model = GloVeMixedCurvature(EMBEDDING_SIZE, CONTEXT_SIZE, vocab_size, resume=resume, expt_name=expt_name)
    else:
        print(f"wtf is {TYPE}")
    
    model.to(device)

    # fit corpus to count cooccurance matrix
    model.fit(corpus)

    cooccurance_matrix = model.get_coocurrance_matrix()
    # saving cooccurance_matrix
    with open(COMATRIX_PATH, mode='wb') as fp:
        pickle.dump(cooccurance_matrix, fp)

    model.train(NUM_EPOCH, device, learning_rate=LEARNING_RATE)
    # print(f"Space Weight: {model.ws}")


    # save model for evaluation
    torch.save(model.state_dict(), MODEL_PATH)

import argparse

if __name__ == '__main__':
    SEED = 6969
    # random.seed(SEED)
    # np.random.set_seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

    parser = argparse.ArgumentParser()

    parser.add_argument("--type", type=str, default="vanilla")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    
    TYPE = args.type

    CORPUS_PICKLE = "data/f_test_" + TYPE + ".pkl"

    # TYPE = "vanilla"
    # TYPE = "mixed"
    DICT_PICKLE = "data/dict_test_" + TYPE + ".pkl"

    CONFIG_FILE = "config_" + TYPE + ".pkl"

    model = train_glove_model(TYPE, FILE_PATH, CORPUS_PICKLE, CONFIG_FILE, args.resume, expt_name=TYPE)
    
    ###testing 
    # test_glove_model(model, DICT_PICKLE)


