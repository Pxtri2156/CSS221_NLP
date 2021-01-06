import string
import re
import argparse
from pickle import dump
from unicodedata import normalize
from numpy import array


from util.load_and_save import save_data, load_data, load_doc 


# split a loaded document into sentences
def to_pairs(doc):
	lines = doc.strip().split('\n')
	pairs = [line.split('\t') for line in  lines]
	return pairs

# clean a list of lines
def clean_pairs(lines):
  cleaned = list()
  # prepare regex for char filtering
  re_print = re.compile('[^%s]' % re.escape(string.printable))
  # prepare translation table for removing punctuation
  table = str.maketrans('', '', string.punctuation)
  for pair in lines:
    clean_pair = list()
    for line in pair:
        print("line 1: ", line)
        # tokenize on white space
        line = line.split()
        print("line 2: ", line)
        # convert to lowercase
        line = [word.lower() for word in line]
        print('line 3: ', line)
        # remove punctuation from each token
        line = [word.translate(table) for word in line]
        print("line 4: ", line)
        # remove tokens with numbers in them
        line = [word for word in line if word.isalpha()]
        print("line 5: ", line)
        # store as string
        clean_pair.append(' '.join(line))
    print('clean pair: ', clean_pair)
    cleaned.append(clean_pair)
  return array(cleaned)


def main(args):
    dataset_path = args.input_path
    # load dataset
    doc = load_doc(dataset_path)
    print("doc: ", type(doc))
    # split into english-german pairs
    pairs = to_pairs(doc)
    print('pairs: ', pairs[:10])
    # clean sentences
    clean_pairs = clean_pairs(pairs)
    print('clean data: ', clean_pairs[:10])
    clean_pairs = clean_pairs[:,:2]
    # save clean pairs to file
    save_clean_data(clean_pairs , args.output_path)
    # spot check
    for i in range(1000):
        print('[%s] => [%s]' % (clean_pairs[i,0], clean_pairs[i,1]))

def args_parse():
    parser = argparse.ArgumentParser(description="Nature language processing ")
    parser.add_argument('-i', '--input_path',  default="./data.txt",
                        help="The path of en-vi data")
    parser.add_argument('-o', '--output_path',  default="./english-vietnamese.pkl",
                        help="The path of clean data ")
    return parser.parse_args()

if __name__ == '__main__':

    args = args_parse()
    main(args)