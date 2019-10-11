# models.py

from sentiment_data import *
from typing import List
import numpy as np
import torch
import torch.nn as nn
from torch import optim


class Net(nn.Module):
    def __init__(self, input_size=300):
        super(Net, self).__init__()
        self.input_size= input_size
        self.net = nn.Sequential(
            nn.Linear(self.input_size, 100),
            nn.ReLU(True),
            nn.Linear(100, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)


class RNN(nn.Module):
    def __init__(self, device, input_size=300, hidden_size=100, num_layers=2, num_classes=1):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.device = device
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.output = nn.Sequential(
            nn.Linear(hidden_size, num_classes),
            nn.Sigmoid()
        )

    def forward(self, x, last_seq):
        # Set initial hidden and cell states
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(self.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(self.device)

        out, _ = self.lstm(x, (h0, c0))  # out: tensor of shape (batch_size, seq_length, hidden_size)
        adapt_output = torch.zeros(x.size(0), self.hidden_size).to(self.device)

        for i in range(x.size(0)):
            adapt_output[i, :] = out[i, int(last_seq[i]), :]

        return self.output(adapt_output)


def acc_evaluate(output, target):
    predict = (output >= 0.5).squeeze()
    return (predict == target).sum().float()

def dev_eval(Net1, word_vectors, device, dev_mat, dev_labels_arr):
    input_vec = []
    target = []
    for j in range(dev_mat.shape[0]):
        idx_sen = dev_mat[j]
        target.append(dev_labels_arr[j])
        vec_sen = 0

        for j in idx_sen:
            vec_sen += word_vectors.get_embedding(int(j)) / len(idx_sen)
        input_vec.append(vec_sen)

    input_vec = torch.from_numpy(np.asarray(input_vec)).to(device).float()
    target = torch.from_numpy(np.asarray(target)).to(device).float()
    output = Net1(input_vec)
    Accuracy = acc_evaluate(output, target)/dev_mat.shape[0]
    return Accuracy


def dev_eval_rnn(Net1, word_vectors, device, dev_mat, dev_labels_arr, dev_seq_lens):

    input_vec = []
    target = []
    last_seq = []

    for j in range(dev_mat.shape[0]):
        idx_sen = dev_mat[j]
        target.append(dev_labels_arr[j])
        last_seq.append(dev_seq_lens[j])

        vec_sen = []
        for j in idx_sen:
            vec_sen.append(word_vectors.get_embedding(int(j)) / len(idx_sen))

        input_vec.append(np.asarray(vec_sen))

    input_vec = torch.from_numpy(np.asarray(input_vec)).to(device).float()
    target = torch.from_numpy(np.asarray(target)).to(device).float()
    last_seq = torch.from_numpy(np.asarray(last_seq)).to(device).float()

    output = Net1(input_vec, last_seq)
    Accuracy = acc_evaluate(output, target)/dev_mat.shape[0]

    return Accuracy


def pad_to_length(np_arr, length):
    """
    Forces np_arr to length by either truncation (if longer) or zero-padding (if shorter)
    :param np_arr:
    :param length: Length to pad to
    :return: a new numpy array with the data from np_arr padded to be of length length. If length is less than the
    length of the base array, truncates instead.
    """
    result = np.zeros(length)
    result[0:np_arr.shape[0]] = np_arr
    return result


# , using dev_exs for development and returning
# predictions on the *blind* test_exs (all test_exs have label 0 as a dummy placeholder value).
def train_evaluate_ffnn(train_exs: List[SentimentExample], dev_exs: List[SentimentExample], test_exs: List[SentimentExample], word_vectors: WordEmbeddings) -> List[SentimentExample]:
    """
    Train a feedforward neural network on the given training examples, using dev_exs for development, and returns
    predictions on the *blind* test examples passed in. Returned predictions should be SentimentExample objects with
    predicted labels and the same sentences as input (but these won't be read by the external code). The code is set
    up to go all the way to test predictions so you have more freedom to handle example processing as you see fit.
    :param train_exs:
    :param dev_exs:
    :param test_exs:
    :param word_vectors:
    :return:
    """
    # 59 is the max sentence length in the corpus, so let's set this to 60
    seq_max_len = 60
    # To get you started off, we'll pad the training input to 60 words to make it a square matrix.
    train_mat = np.asarray([pad_to_length(np.array(ex.indexed_words), seq_max_len) for ex in train_exs])
    train_labels_arr = np.array([ex.label for ex in train_exs])

    dev_mat = np.asarray([pad_to_length(np.array(ex.indexed_words), seq_max_len) for ex in dev_exs])
    dev_labels_arr = np.array([ex.label for ex in dev_exs])

    test_mat = np.asarray([pad_to_length(np.array(ex.indexed_words), seq_max_len) for ex in test_exs])
    test_labels_arr = np.array([ex.label for ex in test_exs])

    num_epoch = 4
    batch_size = 50
    use_cuda = torch.cuda.is_available()
    device = 'cuda' if use_cuda else 'cpu'
    print('device is', device)
    Net1 = Net().to(device).float()
    optim_Net1 = optim.Adam(Net1.parameters(), lr=0.01, betas=(0.9, 0.999))
    criterion = nn.BCELoss()

    # Also store the sequence lengths -- this could be useful for training LSTMs
    train_seq_lens = np.array([len(ex.indexed_words) for ex in train_exs])
    # Labels

    for epoch in range(num_epoch):
        input_vec = []
        target = []

        cnt = 0
        train_loss = 0
        train_acc = 0

        for j in range(train_mat.shape[0]):
            idx_sen = train_mat[j]
            idx_sen = idx_sen[np.nonzero(idx_sen)]
            target.append(train_labels_arr[j])
            vec_sen = 0

            for j in idx_sen:
                vec_sen += word_vectors.get_embedding(int(j))/len(idx_sen)

            input_vec.append(vec_sen)
            cnt += 1

            if cnt >= batch_size:

                input_vec = torch.from_numpy(np.asarray(input_vec)).to(device).float()
                target = torch.from_numpy(np.asarray(target)).to(device).float()

                output = Net1(input_vec)

                Accuracy = acc_evaluate(output, target)
                loss = criterion(output, target)

                train_loss += loss.item()
                train_acc += Accuracy

                optim_Net1.zero_grad()
                loss.backward()
                optim_Net1.step()

                cnt = 0
                input_vec = []
                target = []


        print('Epoch:{:}, loss:{:6f}, train_acc:{:.4f}, dev_acc:{:.6f}'.format(epoch, train_loss, train_acc.item()/train_mat.shape[0], dev_eval(Net1, word_vectors, device, dev_mat, dev_labels_arr)))

    input_vec = []
    for j in range(test_mat.shape[0]):
        idx_sen = test_mat[j]
        vec_sen = 0

        for j in idx_sen:
            vec_sen += word_vectors.get_embedding(int(j)) / len(idx_sen)
        input_vec.append(vec_sen)

    input_vec = torch.from_numpy(np.asarray(input_vec)).to(device).float()
    output = Net1(input_vec)
    test_predict = 1 * (output >= 0.5)

    cnt = 0
    for ex in test_exs:
        ex.label = test_predict[cnt]
        cnt += 1

    return test_exs


# Analogous to train_ffnn, but trains your fancier model.
def train_evaluate_fancy(train_exs: List[SentimentExample], dev_exs: List[SentimentExample], test_exs: List[SentimentExample], word_vectors: WordEmbeddings) -> List[SentimentExample]:
    seq_max_len = 60
    # To get you started off, we'll pad the training input to 60 words to make it a square matrix.
    train_mat = np.asarray([pad_to_length(np.array(ex.indexed_words), seq_max_len) for ex in train_exs])
    train_labels_arr = np.array([ex.label for ex in train_exs])
    train_seq_lens = np.array([len(ex.indexed_words) for ex in train_exs])

    dev_mat = np.asarray([pad_to_length(np.array(ex.indexed_words), seq_max_len) for ex in dev_exs])
    dev_labels_arr = np.array([ex.label for ex in dev_exs])
    dev_seq_lens = np.array([len(ex.indexed_words) for ex in dev_exs])

    test_mat = np.asarray([pad_to_length(np.array(ex.indexed_words), seq_max_len) for ex in test_exs])
    #dev_labels_arr = np.array([ex.label for ex in dev_exs])
    test_seq_lens = np.array([len(ex.indexed_words) for ex in test_exs])

    num_epoch = 27
    batch_size = 50
    use_cuda = torch.cuda.is_available()
    device = 'cuda' if use_cuda else 'cpu'
    print('device is', device)
    Net1 = RNN(device).to(device).float()
    optim_Net1 = optim.SGD(Net1.parameters(), lr=0.01)
    criterion = nn.BCELoss()

    for epoch in range(num_epoch):

        input_vec = []
        target = []
        last_seq = []

        cnt = 0
        train_loss = 0
        train_acc = 0

        for j in range(train_mat.shape[0]):
            idx_sen = train_mat[j]
            target.append(train_labels_arr[j])
            last_seq.append(train_seq_lens[j])

            vec_sen = []
            for j in idx_sen:
                vec_sen.append(word_vectors.get_embedding(int(j))/len(idx_sen))

            input_vec.append(np.asarray(vec_sen))
            cnt += 1

            if cnt >= batch_size:
                input_vec = torch.from_numpy(np.asarray(input_vec)).to(device).float()
                target = torch.from_numpy(np.asarray(target)).to(device).float()
                last_seq = torch.from_numpy(np.asarray(last_seq)).to(device).float()

                output = Net1(input_vec, last_seq)

                Accuracy = acc_evaluate(output, target)

                loss = criterion(output, target)

                train_loss += loss.item()
                train_acc += Accuracy

                optim_Net1.zero_grad()
                loss.backward()
                optim_Net1.step()

                cnt = 0
                input_vec = []
                target = []
                last_seq = []

        print('Epoch:{:}, loss:{:6f}, train_acc:{:.4f}, dev_acc:{:.6f}'.format(epoch, train_loss,  train_acc.item() / train_mat.shape[0], dev_eval_rnn(
            Net1, word_vectors, device, dev_mat, dev_labels_arr, dev_seq_lens
        )))

    input_vec = []
    last_seq = []

    for j in range(test_mat.shape[0]):
        idx_sen = test_mat[j]
        last_seq.append(test_seq_lens[j])

        vec_sen = []
        for j in idx_sen:
            vec_sen.append(word_vectors.get_embedding(int(j)) / len(idx_sen))

        input_vec.append(np.asarray(vec_sen))

    input_vec = torch.from_numpy(np.asarray(input_vec)).to(device).float()
    last_seq = torch.from_numpy(np.asarray(last_seq)).to(device).float()

    output = Net1(input_vec, last_seq)
    test_predict = 1*(output >=0.5)


    cnt=0
    for ex in test_exs:
        ex.label = test_predict[cnt]
        cnt += 1

    return test_exs
