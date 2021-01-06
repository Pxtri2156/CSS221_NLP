import argparse
from pickle import load
from pickle import dump
from numpy.random import rand
from numpy.random import shuffle
import numpy  as np

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical


import sys 

sys.path.append("./")
from util.load_and_save import load_data, save_data


# fit a tokenizer
def create_tokenizer(lines):
	tokenizer = Tokenizer()
	tokenizer.fit_on_texts(lines)
	return tokenizer

# max sentence length
def max_length(lines):
	return max(len(line.split()) for line in lines)

# encode and pad sequences
def encode_sequences(tokenizer, length, lines):
	# integer encode sequences
	X = tokenizer.texts_to_sequences(lines)
	# pad sequences with 0 values
	X = pad_sequences(X, maxlen=length, padding='post')
	return X


# one hot encode target sequence
def encode_output(sequences, vocab_size):
	ylist = list()
	for sequence in sequences:
		encoded = to_categorical(sequence, num_classes=vocab_size)
		ylist.append(encoded)
	y = np.array(ylist)
	y = y.reshape(sequences.shape[0], sequences.shape[1], vocab_size)
	return y

def main(args):

    # load datasets
    dataset = load_data(args.data_path)
    train = load_data(args.train_path)
    test = load_data(args.test_path)
    # prepare english tokenizer
    eng_tokenizer = create_tokenizer(dataset[:, 0])
    eng_vocab_size = len(eng_tokenizer.word_index) + 1
    eng_length = max_length(dataset[:, 0])
    print('English Vocabulary Size: %d' % eng_vocab_size)
    print('English Max Length: %d' % (eng_length))
    print("English tokenize: ", eng_tokenizer )
    # prepare vietnameses tokenizer
    vi_tokenizer = create_tokenizer(dataset[:, 1])
    vi_vocab_size = len(vi_tokenizer.word_index) + 1
    vi_length = max_length(dataset[:, 1])
    print('Vietnamese Vocabulary Size: %d' % vi_vocab_size)
    print('Vietnamese Max Length: %d' % (vi_length))
    print('Vietnamese tokenize: ', vi_tokenizer)

    # prepare training data
    trainX = encode_sequences(eng_tokenizer, eng_length, train[:, 0])
    print("train X: ", trainX.shape)
    trainY = encode_sequences(vi_tokenizer, vi_length, train[:, 1])
    print('train Y: ', trainY.shape)
    trainY = encode_output(trainY, vi_vocab_size)
    print('train Y: ', trainY.shape)
    # prepare validation data
    testX = encode_sequences(eng_tokenizer, eng_length, test[:, 0])
    print("testX: ", testX.shape)
    testY = encode_sequences(vi_tokenizer, vi_length, test[:, 1])
    print('testY: ', testY.shape)
    testY = encode_output(testY, vi_vocab_size)
    print('testY: ', testY.shape)

def args_parse():

    parser = argparse.ArgumentParser(description = "This is argument of encode and decode" )
    parser.add_argument('-d', '--data_path', default="./data.pkl",
    help="this is the path of data that need split")
    parser.add_argument('-tr', '--train_path', default="./train.pkl",
    help="The path of train data was splited")
    parser.add_argument('-ts', '--test_path', default="./test.pkl",
    help="The path of test data was splited")
    
    return parser.parse_args()

if __name__ == "__main__":
    args = args_parse()
    main(args)
