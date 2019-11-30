#!/usr/bin/env python

import argparse
import pickle

import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.utils.data import Dataset, DataLoader


def load_data(filename):
    with open(filename, 'rb') as file:
        return pickle.load(file)


class WikiDataset(Dataset):
    def __init__(self, sentences):
        self.sentences = sentences

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, index):
        return torch.as_tensor(self.sentences[index])


def to_words(mapping, sentence: torch.Tensor):
    mapping = {v: k for k, v in mapping.items()}
    result = ''
    for word in torch.split(sentence, 1, dim=0):
        word = torch.squeeze(word)

        _, i = word.max(dim=0)
        result += ' ' + mapping[i.item()]
    return result


class Discriminator(nn.Module):
    def __init__(self, vocab_size):
        super(Discriminator, self).__init__()
        embedding_size = 64
        hidden_size = 64
        self.pad_token = 0
        self.embedding = Parameter(torch.Tensor(
            vocab_size, embedding_size))

        self.lstm = nn.LSTM(embedding_size, hidden_size)
        self.out = nn.Linear(hidden_size, 1)

    def _forward_impl(self, x):
        x, _ = self.lstm(x)
        x = F.relu(x)
        x = self.out(x)
        x = torch.sigmoid(x)
        return x[-1]

    def forward(self, x):
        x = F.embedding(x, self.embedding, self.pad_token)
        x = x[:, None]
        return self._forward_impl(x)

    def forward_digit(self, x):
        x = torch.matmul(x, self.embedding)
        x = x[:, None]
        return self._forward_impl(x)


class Generator(nn.Module):
    def __init__(self, vocab_size, max_length, latent_size):
        super(Generator, self).__init__()
        embedding_size = 64
        # pad_token = 0
        self.max_length = max_length

        self.lstm = nn.LSTM(latent_size, embedding_size)
        self.to_out = nn.Linear(embedding_size, vocab_size)

    def forward(self, z):
        z = z[None, :].expand(self.max_length, -1)[:, None]
        # seq_len x batch x input_size
        y, _ = self.lstm(z)
        y = self.to_out(y)
        y = torch.squeeze(y)
        return F.softmax(y, dim=1)


def main(database):
    data = load_data(database)
    mapping = data['mapping']
    max_length = data['max_length']
    sentences = data['sentences']
    dataset = WikiDataset(sentences)
    # dataloader = DataLoader(dataset, batch_size=32,
    #                         shuffle=True, num_workers=0)

    latent_size = 10
    generator = Generator(len(mapping), max_length, latent_size)
    discriminator = Discriminator(len(mapping))

    # Training setup
    real_label = torch.as_tensor([1], dtype=float).float()
    fake_label = torch.as_tensor([0], dtype=float).float()

    criterion = nn.BCELoss()

    discriminator_optim = optim.SGD(discriminator.parameters(), lr=0.01)
    generator_optim = optim.SGD(generator.parameters(), lr=0.01)

    for epoch in range(50):
        # Discriminator with real
        discriminator_optim.zero_grad()
        real = dataset[epoch]
        judge = discriminator(real).view(-1)
        err_dis_real = criterion(judge, real_label)
        err_dis_real.backward()
        reals_count = judge.mean().item()
        print(f'{reals_count*100:.3f}% of real images are judged as real')

        # Discriminator with fakes
        z = torch.randn(latent_size)
        fake = generator(z)
        judge = discriminator.forward_digit(fake.detach()).view(-1)
        fakes_count = judge.mean().item()
        print(f'{fakes_count*100:.3f}% of fake images are judged as real')
        err_dis_fake = criterion(judge, fake_label)
        err_dis_fake.backward()
        discriminator_optim.step()

        # Update creator
        generator_optim.zero_grad()
        judge = discriminator.forward_digit(fake).view(-1)
        err_dis_fake = criterion(judge, real_label)
        err_dis_fake.backward()
        print(f'loss G&D: {err_dis_fake.item()}')
        generator_optim.step()

    z = torch.randn(latent_size)
    fake = generator(z)
    judge = discriminator.forward_digit(fake.detach()).view(-1)
    fakes_count = judge.mean().item()
    print(f'{fakes_count*100:.3f}% of fake images are judged as real')
    err_dis_fake = criterion(judge, fake_label)
    print(f'Fake: "{to_words(mapping, fake)}"')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='wiki gan')
    parser.add_argument('-i', '--input-filename', type=str, required=True)
    args = parser.parse_args()
    main(args.input_filename)
