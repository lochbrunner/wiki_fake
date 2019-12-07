#!/usr/bin/env python

import argparse
import pickle

import numpy as np

import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.rnn as rnn
from torch.nn.parameter import Parameter
from torch.utils.data import Dataset, DataLoader


def load_data(filename):
    with open(filename, 'rb') as file:
        return pickle.load(file)


def memoize(function):
    memo = {}

    def wrapper(*args):
        if args in memo:
            return memo[args]
        else:
            rv = function(*args)
            memo[args] = rv
            return rv
    return wrapper


class Padder:
    def __init__(self, max_length, pad_token, device):
        self.max_length = max_length
        self.pad_token = pad_token
        self.device = device

    def __call__(self, x):
        l = len(x)
        padded_x = np.ones((self.max_length))*self.pad_token
        padded_x[0:len(x)] = x

        x = torch.as_tensor(padded_x, dtype=torch.long).to(self.device)
        l = torch.as_tensor(l).to(self.device)
        return x, l


class WikiDataset(Dataset):
    def __init__(self, data, transform):
        self.sentences = data['sentences']
        self.transform = transform

    def __len__(self):
        return len(self.sentences)

    # @memoize
    def __getitem__(self, index):
        return self.transform(self.sentences[index])


def reconstruct_sentence(mapping, sentence: torch.Tensor, display_repeatings=True):
    mapping = {v: k for k, v in mapping.items()}
    result = ''
    last_word_i = -1
    last_word_count = 0
    for word in torch.split(sentence, 1, dim=0):
        word = torch.squeeze(word)

        _, i = word.max(dim=0)
        if i == last_word_i:
            last_word_count += 1
            continue
        if last_word_count > 1 and display_repeatings:
            result += f'({last_word_count})'
        last_word_count = 0
        last_word_i = i
        result += ' ' + mapping[i.item()]
    return result


class Discriminator(nn.Module):
    def __init__(self, vocab_size):
        super(Discriminator, self).__init__()
        embedding_size = 128
        hidden_size = 128
        self.pad_token = 0
        self.embedding = Parameter(torch.Tensor(
            vocab_size, embedding_size))

        self.lstm = nn.LSTM(embedding_size, hidden_size, batch_first=True)
        self.out = nn.Linear(hidden_size, 1)

    def _forward_impl(self, x, l):
        x = rnn.pack_padded_sequence(
            x, l, batch_first=True, enforce_sorted=False)
        x, _ = self.lstm(x)
        x, _ = rnn.pad_packed_sequence(x, batch_first=True)
        x = F.leaky_relu(x, 0.2)
        x = self.out(x)
        x = torch.sigmoid(x)
        return x[:, -1].squeeze()

    def forward(self, x, l):
        x = F.embedding(x, self.embedding, self.pad_token)
        return self._forward_impl(x, l)

    def forward_digit(self, x, l):
        x = torch.matmul(x, self.embedding)
        return self._forward_impl(x, l)


class Generator(nn.Module):
    def __init__(self, vocab_size, max_length, latent_size):
        super(Generator, self).__init__()
        embedding_size = 256
        # pad_token = 0
        self.max_length = max_length

        self.lstm = nn.LSTM(latent_size, embedding_size,
                            num_layers=3, batch_first=True, bidirectional=True)
        self.to_out = nn.Linear(embedding_size*2, vocab_size)

    def forward(self, z, l):
        z = z[:, None, :]

        z = z.expand(-1, self.max_length, -1)
        # seq_len x batch x input_size
        z = rnn.pack_padded_sequence(
            z, l, batch_first=True, enforce_sorted=False)
        y, _ = self.lstm(z)
        y, _ = rnn.pad_packed_sequence(y, batch_first=True)
        y = F.leaky_relu(y, 0.2)
        y = self.to_out(y)

        y = torch.squeeze(y)
        return F.softmax(y, dim=1)


def main(database, model_filename):
    use_cuda = torch.cuda.is_available() and True
    device = torch.device('cuda:0' if use_cuda else 'cpu')
    pad_token = 0
    data = load_data(database)
    mapping = data['mapping']
    max_length = data['max_length']
    batch_size = 32
    dataset = WikiDataset(data, Padder(max_length, pad_token, device))
    dataloader = DataLoader(dataset, batch_size=batch_size,
                            shuffle=True, num_workers=0)

    latent_size = 10
    generator = Generator(len(mapping), max_length, latent_size)
    generator.to(device)
    discriminator = Discriminator(len(mapping))
    discriminator.to(device)

    # Training setup
    real_label = torch.as_tensor([1], dtype=torch.float).to(device)
    fake_label = torch.as_tensor([0], dtype=torch.float).to(device)

    criterion = nn.BCELoss()

    discriminator_optim = optim.SGD(discriminator.parameters(), lr=0.002)
    generator_optim = optim.Adam(generator.parameters(), lr=0.002)
    fake_l = torch.tensor([max_length], dtype=torch.long).to(device)

    def print_fake():
        z = torch.randn(1, latent_size).to(device)
        fake = generator(z, fake_l)
        print(f'Fake: "{reconstruct_sentence(mapping, fake.squeeze())}"')

    for epoch in range(1):
        epoch_reals_count = 0.0
        epoch_fakes_count = 0.0
        epoche_dis_loss = 0.0
        epoche_gen_dis_loss = 0.0
        epoch_i = 0.0
        for real, l in dataloader:
            # Discriminator with real
            discriminator_optim.zero_grad()
            judge = discriminator(real, l).view(-1)
            err_dis_real = criterion(judge, real_label.expand(batch_size))
            err_dis_real.backward()
            reals_count = judge.mean().item()

            # Discriminator with fakes
            z = torch.randn(batch_size, latent_size).to(device)
            fake = generator(z, fake_l.expand(batch_size))
            judge = discriminator.forward_digit(fake.detach(), fake_l).view(-1)
            fakes_count = judge.mean().item()
            dis_loss = criterion(judge, fake_label)
            dis_loss.backward()
            discriminator_optim.step()

            # Update creator
            generator_optim.zero_grad()
            judge = discriminator.forward_digit(fake, fake_l).view(-1)
            gen_dis_loss = criterion(judge, real_label)
            gen_dis_loss.backward()
            generator_optim.step()
            # stats
            epoche_dis_loss += dis_loss.item()
            epoche_gen_dis_loss += gen_dis_loss.item()
            epoch_fakes_count += fakes_count
            epoch_reals_count += reals_count
            epoch_i += 1.0

        reals_count = epoch_reals_count / epoch_i
        fakes_count = epoch_fakes_count / epoch_i
        dis_loss = epoche_dis_loss / epoch_i
        gen_dis_loss = epoche_gen_dis_loss / epoch_i
        print(
            f'real: {reals_count*100:.3f}% - fake: {fakes_count*100:.3f}%  loss D: {dis_loss:.3f} G: {gen_dis_loss:.3f}')
        print_fake()

    print_fake()
    print_fake()
    print_fake()

    torch.save({
        'epoch': 30,
        'latent_size': latent_size,
        'generator_model_state_dict': generator.state_dict(),
        'generator_optim': generator_optim.state_dict(),
        'discriminator_model_state_dict': discriminator.state_dict(),
        'discriminator_optim': discriminator_optim.state_dict()
    }, model_filename)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='wiki gan')
    parser.add_argument('-i', '--input-filename', type=str, required=True)
    parser.add_argument('-m', '--model-filename', type=str,
                        default='data/snapshot.pickle')
    args = parser.parse_args()
    main(args.input_filename)
