import os, sys, argparse, gzip, re, logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from transformers import AutoTokenizer, AutoModel
import tqdm
import random
import string

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def _is_noise_candidate(w: str) -> bool:
    # don't corrupt special tokens, numbers, punctuation-only tokens
    if not w:
        return False
    if any(ch.isdigit() for ch in w):
        return False
    if all(ch in string.punctuation for ch in w):
        return False
    if len(w) <= 2:
        return False
    return True

def _typo_word(w: str) -> str:
    """Create a small typo: swap, delete, insert, or replace a character."""
    if not _is_noise_candidate(w):
        return w

    ops = ["swap", "delete", "insert", "replace"]
    op = random.choice(ops)

    # work with list for easy edits
    chars = list(w)
    L = len(chars)

    if op == "swap" and L >= 3:
        i = random.randint(0, L - 2)
        chars[i], chars[i+1] = chars[i+1], chars[i]
        return "".join(chars)

    if op == "delete" and L >= 3:
        i = random.randint(0, L - 1)
        del chars[i]
        return "".join(chars)

    if op == "insert":
        i = random.randint(0, L)
        c = random.choice(string.ascii_lowercase)
        chars.insert(i, c)
        return "".join(chars)

    if op == "replace":
        i = random.randint(0, L - 1)
        c = random.choice(string.ascii_lowercase)
        chars[i] = c
        return "".join(chars)

    return w

def add_noise_to_sentence(tokens, word_noise_prob=0.15):
    """Randomly corrupt some words in a token list."""
    out = []
    for w in tokens:
        if random.random() < word_noise_prob:
            out.append(_typo_word(w))
        else:
            out.append(w)
    return out

def read_conll(handle, input_idx=0, label_idx=2):
    conll_data = []
    contents = re.sub(r'\n\s*\n', r'\n\n', handle.read())
    contents = contents.rstrip()
    for sent_string in contents.split('\n\n'):
        annotations = list(zip(*[ word_string.split() for word_string in sent_string.split('\n') ]))
        assert(input_idx < len(annotations))
        if label_idx < 0:
            conll_data.append( annotations[input_idx] )
            logging.info("CoNLL: {}".format( " ".join(annotations[input_idx])))
        else:
            assert(label_idx < len(annotations))
            conll_data.append(( annotations[input_idx], annotations[label_idx] ))
            logging.info("CoNLL: {} ||| {}".format( " ".join(annotations[input_idx]), " ".join(annotations[label_idx])))
    return conll_data

class MiniTransformerHead(nn.Module):
    def __init__(self, hidden_dim, tagset_size, nhead=4, ff_dim=256, dropout=0.1):
        super().__init__()
        enc_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=nhead,
            dim_feedforward=ff_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=1)
        self.dropout = nn.Dropout(dropout)
        self.proj = nn.Linear(hidden_dim, tagset_size)

    def forward(self, x, attention_mask=None):
        # x: [B, T, H]
        # Transformer expects src_key_padding_mask=True for PAD positions
        if attention_mask is not None:
            pad_mask = (attention_mask == 0)  # [B, T] bool
        else:
            pad_mask = None

        x = self.encoder(x, src_key_padding_mask=pad_mask)
        x = self.dropout(x)
        return self.proj(x)  # [B, T, C]

class TransformerModel(nn.Module):

    def __init__(
            self,
            basemodel,
            tagset_size,
            lr=5e-5
        ):
        torch.manual_seed(1)
        super(TransformerModel, self).__init__()
        self.basemodel = basemodel
        # the encoder will be a BERT-like model that receives an input text in subwords and maps each subword into
        # contextual representations
        self.encoder = None
        # the hidden dimension of the BERT-like model will be automatically set in the init function!
        self.encoder_hidden_dim = 0
        # The linear layer that maps the subword contextual representation space to tag space
        self.classification_head = None
        # The CRF layer on top of the classification head to make sure the model learns to move from/to relevant tags
        # self.crf_layer = None
        # optimizers will be initialized in the init_model_from_scratch function
        self.optimizers = None
        self.init_model_from_scratch(basemodel, tagset_size, lr)

    def init_model_from_scratch(self, basemodel, tagset_size, lr):
        self.encoder = AutoModel.from_pretrained(basemodel)
        self.encoder_hidden_dim = self.encoder.config.hidden_size
        self.classification_head = MiniTransformerHead(
            hidden_dim=self.encoder_hidden_dim,
            tagset_size=tagset_size,
            nhead=4,        # 4 is safe for DistilBERT hidden=768
            ff_dim=256,
            dropout=0.1
        )
        # TODO initialize self.crf_layer in here as well.
        # TODO modify the optimizers in a way that each model part is optimized with a proper learning rate!
        self.optimizers = [
            optim.Adam(
                list(self.encoder.parameters()) + list(self.classification_head.parameters()),
                lr=lr
            )
        ]

    def forward(self, input_ids, attention_mask=None):
        encoded = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        emissions = self.classification_head(encoded, attention_mask=attention_mask)  # [B, T, C]
        return F.log_softmax(emissions, dim=-1)  # [B, T, C]

class FinetuneTagger:

    def __init__(
            self,
            modelfile,
            modelsuffix='.pt',
            basemodel='distilbert-base-uncased',
            trainfile=os.path.join('data', 'train.txt.gz'),
            epochs=5,
            batchsize=4,
            lr=5e-5
        ):
        # the input sentences will be handled using this object, you do not need to manually encode input sentence words
        self.tokenizer = AutoTokenizer.from_pretrained(basemodel)
        self.trainfile = trainfile
        self.modelfile = modelfile
        self.modelsuffix = modelsuffix
        self.basemodel = basemodel
        self.epochs = epochs
        self.batchsize = batchsize
        self.lr = lr
        self.training_data = []
        self.tag_to_ix = {}  # replace output labels / tags with an index
        self.ix_to_tag = []  # during inference we produce tag indices so we have to map it back to a tag
        self.model = None # setup the model in self.decode() or self.train()

        # --- noise augmentation (robustness to misspellings) ---
        self.use_noise_aug = True
        self.word_noise_prob = 0.15
        self.noise_sent_prob = 0.70   # probability a training sentence is noised

    def load_training_data(self, trainfile):
        if trainfile[-3:] == '.gz':
            with gzip.open(trainfile, 'rt') as f:
                self.training_data = read_conll(f)
        else:
            with open(trainfile, 'r') as f:
                self.training_data = read_conll(f)

        for sent, tags in self.training_data:
            for tag in tags:
                if tag not in self.tag_to_ix:
                    self.tag_to_ix[tag] = len(self.tag_to_ix)
                    self.ix_to_tag.append(tag)

        logging.info("tag_to_ix:", self.tag_to_ix)
        logging.info("ix_to_tag:", self.ix_to_tag)

    def prepare_sequence(self, input_tokens_list, target_sequence=None):
        """
        The function that creates single example (input, target) training tensors or (input) inference tensors.
        """
        sentence_in = self.tokenizer(
            input_tokens_list,
            is_split_into_words=True,
            add_special_tokens=False,
            return_tensors="pt"
        )
        if target_sequence:
            subword_positions = sentence_in.encodings[0].word_ids
            idxs = [self.tag_to_ix[w] for w in target_sequence]
            target = [idxs[x] for x in subword_positions]
            return sentence_in, torch.tensor(target, dtype=torch.long)
        return sentence_in

    def argmax(self, model, seq):
        # collect per-word subword log-prob vectors, then average them
        per_word = [[] for _ in seq]
        with torch.no_grad():
            inputs = self.prepare_sequence(seq).to(device)
            tag_logp = model(
                inputs.input_ids,
                attention_mask=inputs.attention_mask
            ).squeeze(0)  # [Tsub, C]

            for i, word_id in enumerate(inputs.encodings[0].word_ids):
                if word_id is None:
                    continue
                per_word[word_id].append(tag_logp[i])  # store vector [C]

        output = []
        for vecs in per_word:
            avg = torch.stack(vecs, dim=0).mean(dim=0)          # [C]
            output.append(self.ix_to_tag[int(avg.argmax())])    # choose best tag

        assert len(seq) == len(output)
        return output

    def train(self):
        self.load_training_data(self.trainfile)
        self.model = TransformerModel(self.basemodel, len(self.tag_to_ix), lr=self.lr).to(device)
        # TODO You may want to set the weights in the following line to increase the effect of
        #   gradients for infrequent labels and reduce the dominance of the frequent labels
        ignore_idx = -100
        loss_function = nn.NLLLoss(ignore_index=ignore_idx)
        self.model.train()
        loss = float("inf")
        total_loss = 0
        loss_count = 0
        for epoch in range(self.epochs):
            train_iterator = tqdm.tqdm(self.training_data)
            batch = []
            for tokenized_sentence, tags in train_iterator:
                # On-the-fly typo augmentation: keep tags, only corrupt input words
                if self.use_noise_aug and random.random() < self.noise_sent_prob:
                    tokenized_sentence = add_noise_to_sentence(
                        tokenized_sentence,
                        word_noise_prob=self.word_noise_prob
                    )
                # Step 1. Get our inputs ready for the network, that is, turn them into
                # Tensors of subword indices. Pre-trained transformer based models come with their fixed
                # input tokenizer which in our case will receive the words in a sentence and will convert the words list
                # into a list of subwords (e.g. you can look at https://aclanthology.org/P16-1162.pdf to get a better
                # understanding about BPE subword vocabulary creation technique).
                # The expected labels will be copied as many times as the size of the subwords list for each word and
                # returned in targets label.
                batch.append(self.prepare_sequence(tokenized_sentence, tags))
                if len(batch) < self.batchsize:
                    continue
                pad_id = self.tokenizer.pad_token_id
                o_id = self.tag_to_ix['O']
                max_len = max([x[1].size(0) for x in batch])
                # in the next two lines we pad the batch items so that each sequence comes to the same size before
                #  feeding the input batch to the model and calculating the loss over the target values.
                ignore_idx = -100
                input_lens = [x[0].input_ids[0].size(0) for x in batch]
                max_len = max([x[1].size(0) for x in batch])

                input_batch = [
                    x[0].input_ids[0].tolist() + [pad_id] * (max_len - x[0].input_ids[0].size(0))
                    for x in batch
                ]

                attention_batch = [
                    [1] * L + [0] * (max_len - L)
                    for L in input_lens
                ]

                target_batch = [
                    x[1].tolist() + [ignore_idx] * (max_len - x[0].input_ids[0].size(0))
                    for x in batch
                ]
                sentence_in = torch.LongTensor(input_batch).to(device)
                attention_mask = torch.LongTensor(attention_batch).to(device)
                targets = torch.LongTensor(target_batch).to(device)
                # Step 2. Remember that Pytorch accumulates gradients.
                # We need to clear them out before each instance
                self.model.zero_grad()
                # Step 3. Run our forward pass.
                tag_scores = self.model(sentence_in, attention_mask=attention_mask)
                # Step 4. Compute the loss, gradients, and update the parameters by
                #  calling optimizer.step()
                loss = loss_function(tag_scores.view(-1, len(self.tag_to_ix)), targets.view(-1))
                total_loss += loss.item()
                loss_count += 1
                loss.backward()
                # TODO you may want to freeze the BERT encoder for a couple of epochs
                #   and then start performing full fine-tuning.
                for optimizer in self.model.optimizers:
                    optimizer.step()
                # HINT: getting the value of loss below 2.0 might mean your model is moving in the right direction!
                train_iterator.set_description(f"loss: {total_loss/loss_count:.3f}")
                del batch[:]

            if epoch == self.epochs - 1:
                epoch_str = '' # last epoch so do not use epoch number in model filename
            else:
                epoch_str = str(epoch)
            savefile = self.modelfile + epoch_str + self.modelsuffix
            print(f"Saving model file: {savefile}", file=sys.stderr)
            torch.save({
                        'epoch': epoch,
                        'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': self.model.optimizers[0].state_dict(),
                        'loss': loss,
                        'tag_to_ix': self.tag_to_ix,
                        'ix_to_tag': self.ix_to_tag,
                    }, savefile)

    def model_str(self):
        if not os.path.isfile(self.modelfile + self.modelsuffix):
            raise IOError(f"Error: missing model file {self.modelfile + self.modelsuffix}")

        saved_model = torch.load(self.modelfile + self.modelsuffix)
        tag_to_ix = saved_model['tag_to_ix']
        ix_to_tag = saved_model['ix_to_tag']
        model = TransformerModel(self.basemodel, len(tag_to_ix), lr=self.lr).to(device)
        model.load_state_dict(saved_model['model_state_dict'])
        return str(model)

    def decode(self, inputfile):
        if inputfile[-3:] == '.gz':
            with gzip.open(inputfile, 'rt') as f:
                input_data = read_conll(f, input_idx=0, label_idx=-1)
        else:
            with open(inputfile, 'r') as f:
                input_data = read_conll(f, input_idx=0, label_idx=-1)

        if not os.path.isfile(self.modelfile + self.modelsuffix):
            raise IOError(f"Error: missing model file {self.modelfile + self.modelsuffix}")

        saved_model = torch.load(self.modelfile + self.modelsuffix)
        self.tag_to_ix = saved_model['tag_to_ix']
        self.ix_to_tag = saved_model['ix_to_tag']
        model = TransformerModel(self.basemodel, len(self.tag_to_ix), lr=self.lr).to(device)
        model.load_state_dict(saved_model['model_state_dict'])
        # use the model for evaluation not training
        model.eval()
        decoder_output = []
        for sent in tqdm.tqdm(input_data):
            decoder_output.append(self.argmax(model, sent))
        return decoder_output

if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument("-i", "--inputfile", dest="inputfile",
                            default=os.path.join('data', 'input', 'dev.txt'),
                             help="produce chunking output for this input file")
    argparser.add_argument("-t", "--trainfile", dest="trainfile",
                            default=os.path.join('data', 'train.txt.gz'),
                            help="training data for chunker")
    argparser.add_argument("-m", "--modelfile", dest="modelfile",
                            default=os.path.join('data', 'chunker'),
                            help="filename without suffix for model files")
    argparser.add_argument("-s", "--modelsuffix", dest="modelsuffix", default='.pt',
                            help="filename suffix for model files")
    argparser.add_argument("-M", "--basemodel", dest="basemodel",
                            default='distilbert-base-uncased',
                            help="The base huggingface pretrained model to be used as the encoder.")
    argparser.add_argument("-e", "--epochs", dest="epochs", type=int, default=5,
                            help="number of epochs [default: 5]")
    argparser.add_argument("-b", "--batchsize", dest="batchsize", type=int, default=16,
                            help="batch size [default: 16]")
    argparser.add_argument("-r", "--lr", dest="lr", type=float, default=5e-5,
                            help="the learning rate used to finetune the BERT-like encoder module.")
    argparser.add_argument("-f", "--force", dest="force", action="store_true", default=False,
                            help="force training phase (warning: can be slow)")
    argparser.add_argument("-l", "--logfile", dest="logfile", default=None,
                            help="log file for debugging")
    opts = argparser.parse_args()
    if opts.logfile is not None:
        logging.basicConfig(filename=opts.logfile, filemode='w', level=logging.DEBUG)
    modelfile = opts.modelfile
    if modelfile.endswith('.pt'):
        modelfile = modelfile.removesuffix('.pt')
    chunker = FinetuneTagger(
                    modelfile,
                    modelsuffix=opts.modelsuffix,
                    basemodel=opts.basemodel,
                    trainfile=opts.trainfile,
                    epochs=opts.epochs,
                    batchsize=opts.batchsize,
                    lr=opts.lr
                )
    if not os.path.isfile(modelfile + opts.modelsuffix) or opts.force:
        print(f"Could not find modelfile {modelfile + opts.modelsuffix} or -f used. Starting training.", file=sys.stderr)
        chunker.train()
        print("Training done.", file=sys.stderr)
    # use the model file if available and opts.force is False
    assert(os.path.isfile(modelfile + opts.modelsuffix))
    print(f"Found modelfile {modelfile + opts.modelsuffix}. Starting decoding.", file=sys.stderr)
    decoder_output = chunker.decode(opts.inputfile)
    print("\n\n".join([ "\n".join(output) for output in decoder_output ]))