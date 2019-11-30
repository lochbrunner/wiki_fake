#!/usr/bin/env python

import argparse
import pickle


def read_sentences(filename):
    with open(filename, 'rb') as file:
        while True:
            try:
                yield pickle.load(file)['sentence']
            except EOFError:
                return


def clean(words):
    for word in words:
        if '_' in word:
            word = word.replace('_', '')
        if ',' in word:
            word = word.replace(',', '')
        if '.' in word:
            word = word.replace('.', '')
        if word == '':
            continue
        try:
            int(word)
            return '2'
        except Exception:
            pass
        yield word.lower()


def main(input_filename, output_filename):
    mapping = {'<PAD>': 0, ',': 1}
    embedded_sentences = []
    max_length = 0
    longest_sentence = ''

    for sentence in read_sentences(input_filename):
        words = sentence.split(' ')
        if len(words) > max_length:
            max_length = len(words)
            longest_sentence = ' '.join(words)
        words = list(clean(words))
        for word in words:
            if word not in mapping:
                mapping[word] = len(mapping)
        embedded = [mapping[word] for word in words]
        embedded_sentences.append(embedded)

    with open('words.txt', 'w') as file:
        words = [word for word in mapping]
        words.sort()
        for word in words:
            file.write(word + '\n')

    with open(output_filename, 'wb') as file:
        pickle.dump({
            'mapping': mapping,
            'sentences': embedded_sentences,
            'max_length': max_length}, file),

    print(f'Sentences: {len(embedded_sentences)}')
    print(f'Unique Words: {len(mapping)-1}')
    print(f'Total Words: {sum([len(s) for s in embedded_sentences])}')
    print(f'Longest sentence: {max_length}')
    print(f'Longest sentence: {longest_sentence}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Postprocess real data.')
    parser.add_argument('-i', '--input-filename', type=str, required=True)
    parser.add_argument('-o', '--output-filename', type=str, required=True)
    args = parser.parse_args()
    main(input_filename=args.input_filename,
         output_filename=args.output_filename)
