import os.path
import zipfile
import logging
import pickle
import torch
from glove import GloVeModel, GloVeMixedCurvature
from tools import SpacyTokenizer, Dictionary

logging.basicConfig(
    format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

FILE_PATH = 'data/hp.txt'
# FILE_PATH = 'data/book_clean.txt'
COMATRIX_PATH = './data/comat.pickle'
LANG = 'en_core_web_sm'
EMBEDDING_SIZE = 128
CONTEXT_SIZE = 3
NUM_EPOCH = 100
BATHC_SIZE = 512
LEARNING_RATE = 0.01


def read_data(file_path, type='file'):
    """ Read data into a string

    Args:
        file_path (str): path for the data file
    """
    text = None
    if type is 'file':
        with open(file_path, mode='r', encoding='utf-8') as fp:
            text = fp.read()
    elif type is 'zip':
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

    if(not os.path.isfile(CORPUS_PICKLE)):
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
    with open(CORPUS_PICKLE, 'rb') as fp:
        doc = pickle.load(fp)

    dictionary.update(doc)
    logging.info("after generate dictionary")
    corpus = dictionary.corpus(doc)
    vocab_size = dictionary.vocab_size

    return corpus, vocab_size


def train_glove_model(TYPE, FILE_PATH, CORPUS_PICKLE):
    
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

    if TYPE=="vanilla":
        model = GloVeModel(EMBEDDING_SIZE, CONTEXT_SIZE, vocab_size)
    elif TYPE=="mixed":
        model = GloVeMixedCurvature(EMBEDDING_SIZE, CONTEXT_SIZE, vocab_size)
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
    print(f"Space Weight: {model.ws}")


    # save model for evaluation
    torch.save(model.state_dict(), MODEL_PATH)


if __name__ == '__main__':
    CORPUS_PICKLE = "data/f.pkl"
    # TYPE = "vanilla"
    TYPE = "mixed"

    train_glove_model(TYPE, FILE_PATH, CORPUS_PICKLE)
