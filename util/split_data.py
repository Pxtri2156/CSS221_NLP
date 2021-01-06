from pickle import load
from pickle import dump
from numpy.random import rand
from numpy.random import shuffle

from util.load_and_save import save_data, load_data



def main(args):
    # load dataset
    clean_dataset_path = args.input_path
    raw_dataset = load_data(clean_dataset_path)

    # reduce dataset size
    dataset = raw_dataset
    # random shuffle
    shuffle(dataset)
    # split into train/test
    len_split = args.ratio*dataset.shape[0]
    train, test = dataset[:len_split], dataset[len_split: ]
    print("Shape train: ", train.shape)
    print("Shape test: ", test.shape)
    # save
    save_data(train, args.train_path)
    save_data(test, args.test_path)

def args_parse():
    parser = argparse.ArgumentParser(description = "This is argument of split data" )
    parser.add_argument('-i', '--input_path', default="./data.pkl",
    help="this is the path of data that need split")
    parser.add_argument('-r', '--ratio', default=0.7,
    help="Ratio for split train data and test data, default: train/test -> 7/3")
    parser.add_argument('-tr', '--train_path', default="./train.pkl",
    help="The path of train data was splited")
    parser.add_argument('-ts', '--test_path', default="./test.pkl",
    help="The path of test data was splited")

    return parser.parse_args()

if __name__ == "__main__":
    args = args_parse()
    main(args)