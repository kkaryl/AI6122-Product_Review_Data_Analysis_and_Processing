import argparse
from BiLSTM import bilstm_attn

from torchtext.data import TabularDataset
from torchtext import data
from torchtext.vocab import GloVe
import numpy as np
from torch.autograd import Variable
from torch.nn import functional as F

import torch

def get_parser():
    parser = argparse.ArgumentParser(description="classify the given sample")
    parser.add_argument('-n','--name',default='best_model_hidden_128_eval_weighted.pt',help='sentiment classification model name')
    parser.add_argument('-s','--sentence',help='input sentence do classification')
    return parser


def load_data():
    tokenize = lambda x: x.split()
    TEXT = data.Field(sequential=True, tokenize=tokenize, lower=True, include_lengths=True, batch_first=True,
                      fix_length=200)
    LABEL = data.LabelField()
    train_data, test_data = TabularDataset.splits(path='', train='train.csv', test='test.csv', format='csv',
                                                  fields=[('text', TEXT), ('label', LABEL)])
    TEXT.build_vocab(train_data, vectors=GloVe(name='6B', dim=300))
    LABEL.build_vocab(train_data)
    word_embeddings = TEXT.vocab.vectors
    print("Length of Text Vocabulary: " + str(len(TEXT.vocab)))
    print("Vector size of Text Vocabulary: ", TEXT.vocab.vectors.size())
    print("Label Length: " + str(len(LABEL.vocab)))
    vocab_size = len(TEXT.vocab)
    return TEXT,word_embeddings,vocab_size

def initialize(model_file,vocab_size,word_embeddings):
    lr = 1e-3
    batch_size = 32
    output_size = 3
    hidden_size = 128
    embedding_length = 300
    num_layers = 1
    seed = 12
    dropout = 0.5
    torch.manual_seed(seed)
    use_cuda = torch.cuda.is_available()
    sequence_length = 200
    attention_size = 200
    model = bilstm_attn(batch_size, output_size, hidden_size, vocab_size, embedding_length, word_embeddings, True,
                        dropout,
                        use_cuda, sequence_length, attention_size)
    model.load_state_dict(torch.load(model_file))
    model.cuda()
    model.eval()
    return model

def pad(TEXT, x):
    max_len = TEXT.fix_length + (
    TEXT.init_token, TEXT.eos_token).count(None) - 2
    padded= []
    padded.append(([] if TEXT.init_token is None else [TEXT.init_token])+ list(x[-max_len:] if TEXT.truncate_first else x[:max_len])+ ([] if TEXT.eos_token is None else [TEXT.eos_token])+ [TEXT.pad_token] * max(0, max_len - len(x)))
    return padded

def eval(model,test_sen,TEXT):
    test_sen1 = TEXT.preprocess(test_sen)
    test_sen1 = pad(TEXT, test_sen1)[0]
    test_sen1 = [[TEXT.vocab.stoi[x] for x in test_sen1]]
    test_sen = np.asarray(test_sen1)
    test_sen = torch.LongTensor(test_sen)
    test_tensor = Variable(test_sen, requires_grad=False)
    test_tensor = test_tensor.cuda()
    output, att = model(test_tensor, 1)
    output = F.softmax(output, 1)
    return output

def main():
    parser = get_parser()
    args = parser.parse_args()
    print(args)
    TEXT,word_embeddings,vocab_size=load_data()
    model = initialize(args.name,vocab_size,word_embeddings)
    output=eval(model,args.sentence,TEXT)
    print(args.sentence)
    if torch.argmax(output[0]) == 0:
        print("Sentiment: Positive")
    elif torch.argmax(output[0]) == 2:
        print("Sentiment: Neutral")
    else:
        print("Sentiment: Negative")
    print(output.squeeze(0).data.cpu().numpy().tolist())


if __name__ == '__main__':
    main()