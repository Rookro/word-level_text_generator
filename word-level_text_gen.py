#!/usr/bin/env python
# -*- coding: utf-8 -*-


import numpy as np
import os
import sys
import numpy as np
import random
from time import localtime, strftime
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Bidirectional, GRU, Embedding
from keras.callbacks import LambdaCallback, ModelCheckpoint, EarlyStopping, TensorBoard
from keras.utils import plot_model



# Parameters
SEQLEN = 10  # Length of input sequence
STEP = 1
EPOCHS = 50  # Max epoch
BATCH_SIZE = 100
EMBEDDING_DIM = 300  # Dimension of the dense embedding
NUM_GRU = 2 # the number of GRU layer
HIDDEN_SIZE = 800  # Size of hidden layers in GRUs




def split_training_set(sentences_original, next_original, percentage_val=2):
    print('Creating training and validation set...')
    cut_index = int(len(sentences_original) * (1.-(percentage_val/100.)))
    x_train, x_val = sentences_original[:cut_index], sentences_original[cut_index:]
    y_train, y_val = next_original[:cut_index], next_original[cut_index:]

    print('Size of training set = %d' % len(x_train))
    print('Size of validation set = %d' % len(y_val))
    return (x_train, y_train), (x_val, y_val)


def get_model(dropout=0.2):
    print('Building model...')
    model = Sequential()
    model.add(Embedding(input_dim=len(words), output_dim=EMBEDDING_DIM))
    for _ in range(NUM_GRU - 1):
        model.add(Bidirectional(GRU(HIDDEN_SIZE, dropout=dropout, recurrent_dropout=dropout, return_sequences = True)))
    model.add(Bidirectional(GRU(HIDDEN_SIZE)))
    model.add(Dropout(dropout))
    model.add(Dense(len(words)))
    model.add(Activation('softmax'))
    return model


# Data generator for model
def generator(sentence_list, next_word_list, batch_size):
    index = 0
    while True:
        x = np.zeros((batch_size, SEQLEN), dtype=np.int32)
        y = np.zeros((batch_size), dtype=np.int32)
        for i in range(batch_size):
            for t, w in enumerate(sentence_list[index % len(sentence_list)]):
                x[i, t] = word2id[w]
            y[i] = word2id[next_word_list[index % len(sentence_list)]]
            index = index + 1
        yield x, y


# Functions from keras-team/keras/blob/master/examples/lstm_text_generation.py
def sample(preds, temperature=1.0):
    # helper function to sample an index from a probability array
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)


def on_epoch_end(epoch, logs):
    # Function invoked at end of each epoch. Prints generated text.
    examples_file.write('\n----- Generating text after Epoch: %d\n' % epoch)

    # Randomly pick a seed sequence
    seed_index = np.random.randint(len(input_sentences+input_sentences_val))
    seed = (input_sentences+input_sentences_val)[seed_index]

    for diversity in [0.3, 0.4, 0.5, 0.6, 0.7]:
        sentence = seed
        examples_file.write('----- Diversity:' + str(diversity) + '\n')
        examples_file.write('----- Generating with seed:\n' + ' '.join(sentence) + '\n')
        examples_file.write(' '.join(sentence))

        for i in range(60):
            x_pred = np.zeros((1, SEQLEN))
            for t, word in enumerate(sentence):
                x_pred[0, t] = word2id[word]

            preds = model.predict(x_pred, verbose=0)[0]
            next_index = sample(preds, diversity)
            next_word = id2word[next_index]

            sentence = sentence[1:]
            sentence.append(next_word)

            examples_file.write(' '+next_word)
        examples_file.write('\n')
    examples_file.write('='*80 + '\n')
    examples_file.flush()


def make_tensorboard(set_dir_name=''):
    tictoc = strftime('%a_%d_%b_%Y_%H_%M_%S', localtime())
    directory_name = tictoc
    log_dir = set_dir_name + '_' + directory_name
    os.mkdir(log_dir)
    tensorboard = TensorBoard(log_dir=log_dir)
    return tensorboard





if __name__ == '__main__':
    # Argument check
    if not len(sys.argv) > 1:
        print('\033[91m', 'No input file is given.', '\033[0m')
        sys.exit()
    elif not os.path.isfile(sys.argv[1]):
        print('\033[91m', sys.argv[1], 'not found.', '\033[0m')
        sys.exit()

    if not os.path.isdir('./checkpoints/text_generator_model/'):
        os.makedirs('./checkpoints/text_generator_model/')

    data = sys.argv[1]
    examples = './examples_text_generator_' + strftime('%a_%d_%b_%Y_%H_%M_%S', localtime()) + '.txt'

    # load the text file
    print('Loading the text file...')
    with open(data, encoding='utf-8') as f:
        corpus = f.read().replace('\n', ' \n ')
    print('Corpus length in characters: ', len(corpus))

    # split text into words
    corpus_in_words = [w for w in corpus.split(' ') if w.strip() != '' or w == '\n']
    print('Corpus length in words:', len(corpus_in_words))

    # create lookup tables
    words = set(corpus_in_words)
    word2id = dict((c, i) for i, c in enumerate(words))
    id2word = dict((i, c) for i, c in enumerate(words))


    print('Creating input and next words...')
    input_sentences = []
    next_words = []
    for i in range(0, len(corpus_in_words) - SEQLEN, STEP):
        input_sentences.append(corpus_in_words[i:i + SEQLEN])
        next_words.append(corpus_in_words[i + SEQLEN])
    print('The number of input: ', len(input_sentences))

    # split (x, y), (x_val, y_val)
    (input_sentences, next_words), (input_sentences_val, next_words_val) = split_training_set(input_sentences, next_words)


    model = get_model()
    model.summary()
    plot_model(model, to_file='model.pdf', show_shapes=True)
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    file_path = './checkpoints/text_generator_model/GRU-epoch{epoch:03d}-words%d-seqlen%d-loss{loss:.4f}-acc{acc:.4f}-val_loss{val_loss:.4f}-val_acc{val_acc:.4f}' % (
         len(words), SEQLEN,
    )
    checkpoint = ModelCheckpoint(file_path, monitor='val_acc', save_best_only=True)
    print_callback = LambdaCallback(on_epoch_end=on_epoch_end)
    early_stopping = EarlyStopping(monitor='val_acc', patience=5)
    callbacks_list = [checkpoint, print_callback, make_tensorboard(set_dir_name = 'tensorboard'), early_stopping]

    examples_file = open(examples, 'w')
    model.fit_generator(generator(input_sentences, next_words, BATCH_SIZE),
                        steps_per_epoch=int(len(input_sentences)/BATCH_SIZE) + 1,
                        epochs=EPOCHS,
                        callbacks=callbacks_list,
                        validation_data=generator(input_sentences_val, next_words_val, BATCH_SIZE),
                        validation_steps=int(len(input_sentences_val)/BATCH_SIZE) + 1)
    model.save('text_generator_model_' + strftime('%a_%d_%b_%Y_%H_%M_%S', localtime()))



















