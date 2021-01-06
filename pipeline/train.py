import argparse

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.utils.vis_utils import plot_model
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import Embedding
from keras.layers import RepeatVector
from keras.layers import TimeDistributed
from keras.callbacks import ModelCheckpoint

import sys 
sys.path.append("./")

from util.load_and_save import load_data, save_data
from model import define_model


from vectorization.encode_decode import create_tokenizer, max_length, encode_sequences, \
                                  encode_output

def main(args):
    # Load data -- sentence pairs 

    # Tokenization 
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

    # Train 
    ## Load model
    model = define_model(eng_vocab_size, vi_vocab_size, eng_length, vi_length, 256)
    model.compile(optimizer='adam', loss='categorical_crossentropy')
    # summarize defined model
    print(model.summary())
    plot_model(model, to_file='model.png', show_shapes=True)
    ## Training 
    # fit model
    filename = args.model_path
    checkpoint = ModelCheckpoint(filename, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
    model.fit(trainX, trainY, epochs=30, batch_size=64, validation_data=(testX, testY), callbacks=[checkpoint], verbose=2)
    # Save model
def args_parse():
    parser = argparse.ArgumentParser(description="Nature language processing ")
    parser.add_argument('-m', '--model_path',  default="./model.h5",
                        help="The path to save model ")
    parser.add_argument('-d', '--data_path', default="./data.pkl",
    help="this is the path of data that need split")
    parser.add_argument('-tr', '--train_path', default="./train.pkl",
    help="The path of train data")
    parser.add_argument('-ts', '--test_path', default="./test.pkl",
    help="The path of test data")
    return parser.parse_args()
    
if __name__ == "__main__":
    args = args_parse()
    main(args)