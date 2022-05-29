from collections import Counter, defaultdict
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import tqdm
import geoopt
import wandb

SEED = 6969
# random.seed(SEED)
# np.random.set_seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)


wandb.init(project="nlp-multimixture", entity="ayushi15")

wandb.config = {
  "learning_rate": 2e-4,
  "epochs": 150,
  "batch_size": 128
}

def normalize(a):
   return (a.T / torch.sqrt(torch.sum(a**2, dim=1))).T


class Euclidean():
    def __init__(self):
        pass

    def dist(self, a, b):

        return torch.norm(b-a)

class ManifoldEmbedding(nn.Module):
    def __init__(self, embedding_size, vocab_size, c):
        super(ManifoldEmbedding, self).__init__()

        self._focal_embeddings = nn.Embedding(
            vocab_size, embedding_size).type(torch.float64)

        self._context_embeddings = nn.Embedding(
            vocab_size, embedding_size).type(torch.float64)

        self._focal_biases = nn.Embedding(
            vocab_size, 1).type(torch.float64)

        self._context_biases = nn.Embedding(
            vocab_size, 1).type(torch.float64)

        if not c:
            self.manifold = Euclidean()  
        else:  
            self.manifold = geoopt.manifolds.Stereographic(k=c, learnable=False)
        
        # self.manifold = geoopt.manifolds.Stereographic(k=c, learnable=(c!=0))
        self.c = c

    def forward(self, focal_input, context_input, log_coocurrence_count):
        focal_embed = normalize(self._focal_embeddings(focal_input))
        context_embed = normalize(self._context_embeddings(context_input))
        focal_bias = self._focal_biases(focal_input)
        context_bias = self._context_biases(context_input)     
        
        #######################
        d = self.manifold.dist(focal_embed, context_embed)

        if self.c < 0:
            d = torch.cosh(d)**2

        else:
            d = d**2/2 #Applying h function

        d = d/torch.norm(d)
        loss = -d + focal_bias + context_bias - log_coocurrence_count

        return loss**2


class GloVeMixedCurvature(nn.Module):
    """Implement GloVe model in Mixed-Curvature space with Pytorch
    """

    def __init__(self, embedding_size, context_size, vocab_size, min_occurrance=1, x_max=100, alpha=3 / 4, expt_name="default", resume=False):
        super(GloVeMixedCurvature, self).__init__()

        self.embedding_size = embedding_size
        if isinstance(context_size, tuple):
            self.left_context, self.right_context = context_size
        if isinstance(context_size, int):
            self.left_context = self.right_context = context_size
        else:
            raise ValueError(
                "'context_size' should be an int or a tuple of two ints")
        self.vocab_size = vocab_size
        self.alpha = alpha
        self.min_occurrance = min_occurrance
        self.x_max = x_max


        self._focal_embeddings = nn.Embedding(
            vocab_size, embedding_size).type(torch.float64)

        self._context_embeddings = nn.Embedding(
            vocab_size, embedding_size).type(torch.float64)

        self._focal_biases = nn.Embedding(vocab_size, 1).type(torch.float64)
        self._context_biases = nn.Embedding(vocab_size, 1).type(torch.float64)


        self._glove_dataset = None

        for params in self.parameters():

            init.uniform_(params, a=-1, b=1)

        # self.w = nn.Parameter(torch.Tensor([0.33, 0.33, 0.33]))
        self.w = torch.Tensor([0.5, 0.25, 0.25])

        self.euc = ManifoldEmbedding(embedding_size, vocab_size, 0)
        self.hyp = ManifoldEmbedding(embedding_size, vocab_size, -0.5)
        self.sph = ManifoldEmbedding(embedding_size, vocab_size, 0.5)
        self.expt_name = expt_name

        if resume:
            self.load_checkpoints(expt_name)


    def fit(self, corpus):
        """get dictionary word list and co-occruence matrix from corpus

        Args:
            corpus (list): contain word id list

        Raises:
            ValueError: when count zero cocurrences will raise the problems
        """

        left_size, right_size = self.left_context, self.right_context
        vocab_size, min_occurrance = self.vocab_size, self.min_occurrance

        # get co-occurence count matrix
        word_counts = Counter()
        cooccurence_counts = defaultdict(float)
        for region in corpus:
            word_counts.update(region)
            for left_context, word, right_context in _context_windows(region, left_size, right_size):
                for i, context_word in enumerate(left_context[::-1]):
                    # add (1 / distance from focal word) for this pair
                    cooccurence_counts[(word, context_word)] += 1 / (i + 1)
                for i, context_word in enumerate(right_context):
                    cooccurence_counts[(word, context_word)] += 1 / (i + 1)
        if len(cooccurence_counts) == 0:
            raise ValueError(
                "No coccurrences in corpus, Did you try to reuse a generator?")

        # get words bag information
        tokens = [word for word, count in
                  word_counts.most_common(vocab_size) if count >= min_occurrance]
        coocurrence_matrix = [(words[0], words[1], count)
                              for words, count in cooccurence_counts.items()
                              if words[0] in tokens and words[1] in tokens]
        self._glove_dataset = GloVeDataSet(coocurrence_matrix)

    def train(self, num_epoch, device, batch_size=512, learning_rate=0.05, loop_interval=10):
        """Training GloVe model

        Args:
            num_epoch (int): number of epoch
            device (str): cpu or gpu
            batch_size (int, optional): Defaults to 512.
            learning_rate (float, optional): Defaults to 0.05. learning rate for Adam optimizer
            batch_interval (int, optional): Defaults to 100. interval time to show average loss

        Raises:
            NotFitToCorpusError: if the model is not fit by corpus, the error will be raise
        """

        if self._glove_dataset is None:
            raise NotFitToCorpusError(
                "Please fit model with corpus before training")

        # basic training setting
        optimizer = geoopt.optim.RiemannianAdam(self.parameters(), lr=learning_rate)
        glove_dataloader = DataLoader(self._glove_dataset, batch_size)
        total_loss = 0
        best_loss = 10000000

        # Optional
        wandb.watch(self)


        for epoch in tqdm.tqdm(range(num_epoch)):
            count = 0
            total_loss = 0
            # print(f"Weights are: {self.w}")
            for idx, batch in enumerate(glove_dataloader):
                count += 1
                optimizer.zero_grad()

                i_s, j_s, counts = batch
                i_s = i_s.to(device)
                j_s = j_s.to(device)
                counts = counts.to(device)
                loss = self._loss(i_s, j_s, counts)

                total_loss += loss.item()

                loss.backward()
                optimizer.step()

            print("epoch: {}, average loss: {}".format(
                        epoch, total_loss/count))

            if total_loss/count < best_loss:
                best_loss = total_loss/count
                print("Saving checkpoints.....")
                self.save_checkpoints(self.expt_name)

            # wandb.log({"loss": total_loss/count, "Euc Weight": self.ws[0], "Hyp Weight": self.ws[1], "Sph Weight": self.ws[2]})

            wandb.log({"loss": total_loss/count})
            

        print("finish glove vector training")

    def get_coocurrance_matrix(self):
        """ Return co-occurance matrix for saving

        Returns:
            list: list itam (word_idx1, word_idx2, cooccurances)
        """

        return self._glove_dataset._coocurrence_matrix

    def embedding_for_tensor(self, tokens):
        if not torch.is_tensor(tokens):
            raise ValueError("the tokens must be pytorch tensor object")

        return self._focal_embeddings(tokens) + self._context_embeddings(tokens)

    def _loss(self, focal_input, context_input, coocurrence_count):
        x_max, alpha = self.x_max, self.alpha

        # count weight factor
        weight_factor = torch.pow(coocurrence_count / x_max, alpha)
        weight_factor[weight_factor > 1] = 1

        log_coocurrence_count = torch.log(coocurrence_count)

        # self.ws = torch.nn.functional.softmax(self.w)

        l1 = self.euc(focal_input, context_input, log_coocurrence_count)
        l2 = self.hyp(focal_input, context_input, log_coocurrence_count)
        l3 = self.sph(focal_input, context_input, log_coocurrence_count)

        # wandb.log({"l_euc": l1.mean(), "l_hyp": l2.mean(), "l_sph": l3.mean()})

        loss = l1*self.w[0] + l2*self.w[1] + l3*self.w[2]
        single_losses = weight_factor * loss
        # single_losses = weight_factor * distance_expr

        mean_loss = torch.mean(single_losses)
        return mean_loss

    def save_checkpoints(self, expt_name):
        raw_model = self.module if hasattr(self, "module") else self
        save_path = "checkpoints/" + expt_name
        # if suffix is not None:
        #     save_path += "_{}".format(suffix)
        torch.save(raw_model.state_dict(), save_path)

    def load_checkpoints(self, expt_name):
        save_path = "checkpoints/" + expt_name
        self.load_state_dict(torch.load(save_path))





        

class GloVeModel(nn.Module):
    """Implement GloVe model with Pytorch
    """

    def __init__(self, embedding_size, context_size, vocab_size, min_occurrance=1, x_max=100, alpha=3 / 4, expt_name="default", resume=False):
        super(GloVeModel, self).__init__()

        self.embedding_size = embedding_size
        if isinstance(context_size, tuple):
            self.left_context, self.right_context = context_size
        if isinstance(context_size, int):
            self.left_context = self.right_context = context_size
        else:
            raise ValueError(
                "'context_size' should be an int or a tuple of two ints")
        self.vocab_size = vocab_size
        self.alpha = alpha
        self.min_occurrance = min_occurrance
        self.x_max = x_max

        self._focal_embeddings = nn.Embedding(
            vocab_size, embedding_size).type(torch.float64)

        self._context_embeddings = nn.Embedding(
            vocab_size, embedding_size).type(torch.float64)

        self._focal_biases = nn.Embedding(vocab_size, 1).type(torch.float64)
        self._context_biases = nn.Embedding(vocab_size, 1).type(torch.float64)


        self._glove_dataset = None

        for params in self.parameters():
            init.uniform_(params, a=-1, b=1)

        self.expt_name = expt_name
        if resume:
            self.load_checkpoints(expt_name)

    def fit(self, corpus):
        """get dictionary word list and co-occruence matrix from corpus

        Args:
            corpus (list): contain word id list

        Raises:
            ValueError: when count zero cocurrences will raise the problems
        """

        left_size, right_size = self.left_context, self.right_context
        vocab_size, min_occurrance = self.vocab_size, self.min_occurrance

        # get co-occurence count matrix
        word_counts = Counter()
        cooccurence_counts = defaultdict(float)
        for region in corpus:
            word_counts.update(region)
            for left_context, word, right_context in _context_windows(region, left_size, right_size):
                for i, context_word in enumerate(left_context[::-1]):
                    # add (1 / distance from focal word) for this pair
                    cooccurence_counts[(word, context_word)] += 1 / (i + 1)
                for i, context_word in enumerate(right_context):
                    cooccurence_counts[(word, context_word)] += 1 / (i + 1)
        if len(cooccurence_counts) == 0:
            raise ValueError(
                "No coccurrences in corpus, Did you try to reuse a generator?")

        # get words bag information
        tokens = [word for word, count in
                  word_counts.most_common(vocab_size) if count >= min_occurrance]
        coocurrence_matrix = [(words[0], words[1], count)
                              for words, count in cooccurence_counts.items()
                              if words[0] in tokens and words[1] in tokens]
        self._glove_dataset = GloVeDataSet(coocurrence_matrix)

    def train(self, num_epoch, device, batch_size=512, learning_rate=0.05, loop_interval=10):
        """Training GloVe model

        Args:
            num_epoch (int): number of epoch
            device (str): cpu or gpu
            batch_size (int, optional): Defaults to 512.
            learning_rate (float, optional): Defaults to 0.05. learning rate for Adam optimizer
            batch_interval (int, optional): Defaults to 100. interval time to show average loss

        Raises:
            NotFitToCorpusError: if the model is not fit by corpus, the error will be raise
        """

        if self._glove_dataset is None:
            raise NotFitToCorpusError(
                "Please fit model with corpus before training")

        # basic training setting
        optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        glove_dataloader = DataLoader(self._glove_dataset, batch_size)
        total_loss = 0
        best_loss = 100000

        for epoch in range(num_epoch):
            count = 0
            total_loss = 0
            for idx, batch in enumerate(glove_dataloader):
                count += 1
                optimizer.zero_grad()
                i_s, j_s, counts = batch
                i_s = i_s.to(device)
                j_s = j_s.to(device)
                counts = counts.to(device)
                loss = self._loss(i_s, j_s, counts)

                total_loss += loss.item()

                loss.backward()
                optimizer.step()
            print("epoch: {}, average loss: {}".format(epoch, total_loss/count))
            if total_loss/count < best_loss:
                print("Saving best model.......")
                best_loss = total_loss/count
                self.save_checkpoints(self.expt_name)
        
            wandb.log({"loss": total_loss/count})

        print("finish glove vector training")

    def get_coocurrance_matrix(self):
        """ Return co-occurance matrix for saving

        Returns:
            list: list itam (word_idx1, word_idx2, cooccurances)
        """

        return self._glove_dataset._coocurrence_matrix

    def embedding_for_tensor(self, tokens):
        if not torch.is_tensor(tokens):
            raise ValueError("the tokens must be pytorch tensor object")

        return self._focal_embeddings(tokens) + self._context_embeddings(tokens)

    def _loss(self, focal_input, context_input, coocurrence_count):
        x_max, alpha = self.x_max, self.alpha

        focal_embed = self._focal_embeddings(focal_input)
        context_embed = self._context_embeddings(context_input)
        focal_bias = self._focal_biases(focal_input)
        context_bias = self._context_biases(context_input)

        # count weight factor
        weight_factor = torch.pow(coocurrence_count / x_max, alpha)
        weight_factor[weight_factor > 1] = 1

        embedding_products = torch.sum(focal_embed * context_embed, dim=1)
        log_cooccurrences = torch.log(coocurrence_count)

        distance_expr = (embedding_products + focal_bias +
                         context_bias + log_cooccurrences) ** 2

        single_losses = weight_factor * distance_expr
        mean_loss = torch.mean(single_losses)
        return mean_loss

    def save_checkpoints(self, expt_name):
        raw_model = self.module if hasattr(self, "module") else self
        save_path = "checkpoints/" + expt_name
        # if suffix is not None:
        #     save_path += "_{}".format(suffix)
        torch.save(raw_model.state_dict(), save_path)

    def load_checkpoints(self, expt_name):
        save_path = "checkpoints/" + expt_name
        self.load_state_dict(torch.load(save_path))


class GloVeDataSet(Dataset):

    def __init__(self, coocurrence_matrix):
        self._coocurrence_matrix = coocurrence_matrix

    def __getitem__(self, index):
        return self._coocurrence_matrix[index]

    def __len__(self):
        return len(self._coocurrence_matrix)


class NotTrainedError(Exception):
    pass


class NotFitToCorpusError(Exception):
    pass


def _context_windows(region, left_size, right_size):
    """generate left_context, word, right_context tuples for each region

    Args:
        region (str): a sentence
        left_size (int): left windows size
        right_size (int): right windows size
    """

    for i, word in enumerate(region):
        start_index = i - left_size
        end_index = i + right_size
        left_context = _window(region, start_index, i - 1)
        right_context = _window(region, i + 1, end_index)
        yield (left_context, word, right_context)


def _window(region, start_index, end_index):
    """Returns the list of words starting from `start_index`, going to `end_index`
    taken from region. If `start_index` is a negative number, or if `end_index`
    is greater than the index of the last word in region, this function will pad
    its return value with `NULL_WORD`.

    Args:
        region (str): the sentence for extracting the token base on the context
        start_index (int): index for start step of window
        end_index (int): index for the end step of window
    """
    last_index = len(region) + 1
    selected_tokens = region[max(start_index, 0):
                             min(end_index, last_index) + 1]
    return selected_tokens





############# kachraa ##########

     # embedding_products = torch.sum(focal_embed * context_embed, dim=1)
        # # log_cooccurrences = torch.log(coocurrence_count)

        # return (embedding_products + focal_bias +
        #                  context_bias + log_coocurrence_count) ** 2

        # # # single_losses = weight_factor * distance_expr
        # # mean_loss = torch.mean(single_losses)
        # # return mean_loss

        #         # embedding_products = torch.sum(focal_embed * context_embed, dim=1)
        # # log_cooccurrences = torch.log(coocurrence_count)


        # x_max, alpha = self.x_max, self.alpha

        # focal_embed = self._focal_embeddings(focal_input)
        # context_embed = self._context_embeddings(context_input)
        # focal_bias = self._focal_biases(focal_input)
        # context_bias = self._context_biases(context_input)

        # # count weight factor
        # weight_factor = torch.pow(coocurrence_count / x_max, alpha)
        # weight_factor[weight_factor > 1] = 1

        # embedding_products = torch.sum(focal_embed * context_embed, dim=1)
        # log_cooccurrences = torch.log(coocurrence_count)

        # distance_expr = (embedding_products + focal_bias +
        #                  context_bias + log_cooccurrences) ** 2

        # single_losses = weight_factor * distance_expr
        # mean_loss = torch.mean(single_losses)
        # return mean_loss


        # loss = self.euc(focal_input, context_input, log_coocurrence_count)
        ############################
        # focal_embed = normalize(self._focal_embeddings(focal_input))
        # context_embed = normalize(self._context_embeddings(context_input))
        # focal_bias = self._focal_biases(focal_input)
        # context_bias = self._context_biases(context_input)

        # count weight factor
        # weight_factor = torch.pow(coocurrence_count / x_max, alpha)
        # weight_factor[weight_factor > 1] = 1

        # embedding_products = torch.sum(focal_embed * context_embed, dim=1)
        # log_cooccurrences = torch.log(coocurrence_count)

        # distance_expr = (embedding_products + focal_bias +
        #                  context_bias + log_cooccurrences) ** 2


                    # wandb.log({"Euc Weight": self.ws[0]})
            # wandb.log({"Hyp Weight": self.ws[1]})
            # wandb.log({"Sph Weight": self.ws[2]})


                    # print(f"Losses euc {torch.mean(l1)}, hyp {torch.mean(l2)}, sph {torch.mean(l3)}")
        # print(f"C euc {self.euc.manifold.k.item()}, hyp {self.hyp.manifold.k.item()}, sph {self.sph.manifold.k.item()}")
        # print(f"weights are {self.ws}")


                # if type_curv == "euc":
        #     self.manifold = geoopt.manifolds.Euclidean()
        # elif type_curv == 
        # #     self.manifold = geoopt.manifolds.PoincareBall()
        # self.manifold = geoopt.manifolds.StereographicExact(k=c, learnable=False)

                # self.w1 = nn.Parameter(torch.Tensor([0.33]))
        # self.w2 = nn.Parameter(torch.Tensor([0.33]))
