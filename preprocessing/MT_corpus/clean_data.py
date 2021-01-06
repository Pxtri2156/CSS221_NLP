import os
import string
import re
import glob2
import pickle
import argparse
import collections
import numpy as np
from xml.dom import minidom


def to_clean_pairs(lines):
  cleaned = list()
  # prepare regex for char filtering
  re_print = re.compile('[^%s]' % re.escape(string.printable))
  # prepare translation table for removing punctuation
  table = str.maketrans('', '', string.punctuation)
  for pair in lines:
    clean_pair = list()
    for line in pair:
        #print("line 1: ", line)
        # tokenize on white space
        line = line.split()
        #print("line 2: ", line)
        # convert to lowercase
        line = [word.lower() for word in line]
        #print('line 3: ', line)
        # remove punctuation from each token
        line = [word.translate(table) for word in line]
        #print("line 4: ", line)
        # remove tokens with numbers in them
        line = [word for word in line if word.isalpha()]
        #print("line 5: ", line)
        # store as string
        clean_pair.append(' '.join(line))
    #print('clean pair: ', clean_pair)
    cleaned.append(clean_pair)
    #print(cleaned)
    #input()
  return np.array(cleaned)


def load_data(path):
    
    english_data_path = os.path.join(path, 'OutTA')
    vietnam_data_path = os.path.join(path, 'OutTV')
    E_xml_files = sorted(glob2.glob(os.path.join(english_data_path, '*.xml')))
    V_xml_files = sorted(glob2.glob(os.path.join(vietnam_data_path, '*.xml')))

    assert len(E_xml_files) == len(V_xml_files), "The number of english files is different with the number of vietname files !!"

    vietname_sentences = dict()
    english_sentences = dict()
    
    for vietname_file_path, english_file_path in zip(V_xml_files, E_xml_files):
        vietname_content = minidom.parse(vietname_file_path)
        sentences = vietname_content.getElementsByTagName('SEG')
        for sentence in sentences:
            id_sentence = sentence.attributes['id'].value
            content = sentence.firstChild.nodeValue
            vietname_sentences[os.path.split(vietname_file_path)[-1].split(".")[0] + '_' + str(id_sentence)] = content

        english_content = minidom.parse(english_file_path)
        sentences = english_content.getElementsByTagName('SEG')
        for sentence in sentences:
            id_sentence = sentence.attributes['id'].value
            content = sentence.firstChild.nodeValue
            english_sentences[os.path.split(english_file_path)[-1].split(".")[0] + '_' + str(id_sentence)] = content

    assert len(vietname_sentences) == len(english_sentences), "The number of english sentences is different with the number of vietname sentences !!"

    vietname_sentences = collections.OrderedDict(sorted(vietname_sentences.items()))
    english_sentences = collections.OrderedDict(sorted(english_sentences.items()))

    pair_sentences = []
    for english_sentence, vietname_sentence in zip(english_sentences.values(), vietname_sentences.values()):
        pair_sentences.append([english_sentence, vietname_sentence])

    return pair_sentences


def main(args):
    print(args)

    # Read data
    print("[INFO] Loading data to pairs of sentences ...")
    pair_sentences = load_data(path=args['input_path'])
    
    # Clean data
    print("[INFO] Cleaning data ...")
    clean_pair_sentences = to_clean_pairs(pair_sentences)

    # Save clean data
    print("[INFO] Saving clean data to {} ...".format(args['output_path']))
    with open(args['output_path'], 'wb') as f:
        pickle.dump(clean_pair_sentences, f)



def args_parse():
    parser = argparse.ArgumentParser(description="MT corpus data preprocessing")
    parser.add_argument('-i', '--input_path',  default="../../data/MT corpus/Computer/",
                        help="The path of raw data")
    parser.add_argument('-o', '--output_path',  default="../../clean_data/MT_corpus_computer.pkl",
                        help="The path to store clean data ")
    return vars(parser.parse_args())

if __name__ == '__main__':

    args = args_parse()
    main(args)