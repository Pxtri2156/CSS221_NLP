from pickle import load
from pickle import dump

# load doc into memory
def load_doc(filename):
    print("Loading: ", filename)
    # open the file as read only
    file = open(filename, mode='rt', encoding='utf-8')
    # read all text
    text = file.read()
    # close the file
    file.close()
    return text

# save a list of data to pickle file
def save_data(sentences, filename):
    dump(sentences, open(filename, 'wb'))
    print('Saved: %s' % filename)

# load a  dataset from pickle file
def load_data(filename):
    print("Loading: ", filename)
    return load(open(filename, 'rb'))


