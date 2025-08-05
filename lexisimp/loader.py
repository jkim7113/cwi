import os
import re
import pandas as pd
from utils import create_mapping, zero_digits
import model
from collections import Counter
import nltk
from nltk.tokenize import word_tokenize

# nltk.download('all')

def load_sentences(path, lower, zeros):
    df = pd.read_csv(path)
    df.groupby("Sentence").apply(lambda x: x.sort_values(by="End"))

    sentences = []
    sentence = "" # e.g. #12-9 An air ambulance also responded.
    index = 0
    word_list = [] # ['#12-9', 'An', 'air', ... , 'responded', '.']
    label_list = [] # ['N', 'N', 'N', ... , 'C', 'N']
    num_error = 0

    for _, row in df.iterrows():
        if sentence != row["Sentence"]:
            word_list = word_tokenize(sentence) # Split string into words and punctuation. Remove all 's
            label_list = [label.replace(" ","N") for label in label_list] # Replace empty labels with N
            concat_list = [[word_list[i], label_list[i]] for i in range(len(word_list))]
            if len(concat_list) > 0:
                sentences.append(concat_list)

            # Update sentence, word_list, and label_list for the new set of annotations
            sentence = zero_digits(row["Sentence"]) if zeros else row["Sentence"]
            index = 0
            word_list = word_tokenize(sentence)
            label_list = [' ' for word in word_list]
            print(word_list)

        # If the current row contains a multi-word expression, break it into two or more words
        current_word = str(row["Word"]).split() # ['security', 'reasons']
        if len(current_word) > 1: # multi-word expression
            for word in current_word:
                word = word.replace(",","")
                try: 
                    index =  word_list.index(word)
                    if label_list[index] == '': # the current word is left unannotated by single-word rows e.g. the row containing "security" 
                        label_list[index] = "C" if row["Complexity_Binary"] == 1 else "N"
                except:
                    num_error += 1

        else: 
            try:
                index =  word_list.index(current_word[0])
                label_list[index] = "C" if row["Complexity_Binary"] == 1 else "N"
            except:
                num_error += 1

        index += 1

    print(f'Percentage of rows dropped: {num_error / len(df) * 100}')

    return sentences

def word_mapping(sentences, lower):
    """
    Create a dictionary and a mapping of words, sorted by frequency.
    """
    words = [x[0].lower() if lower else x[0] for s in sentences for x in s]
    dico = dict(Counter(words))
    
    dico['<PAD>'] = 10000001
    dico['<UNK>'] = 10000000
    dico = {k:v for k,v in dico.items() if v>=3}
    word_to_id, id_to_word = create_mapping(dico)

    print("Found %i unique words (%i in total)" % (
        len(dico), len(words)
    ))
    return dico, word_to_id, id_to_word


def char_mapping(sentences):
    """
    Create a dictionary and mapping of characters, sorted by frequency.
    """
    chars = ''.join([w[0] for s in sentences for w in s])
    dico = dict(Counter(chars))
    dico['<PAD>'] = 10000001
    dico['<UNK>'] = 10000000

    char_to_id, id_to_char = create_mapping(dico)
    print("Found %i unique characters" % len(dico))
    return dico, char_to_id, id_to_char


def tag_mapping(sentences):
    """
    Create a dictionary and a mapping of tags, sorted by frequency.
    """
    tags = [word[-1] for s in sentences for word in s]
    dico = dict(Counter(tags))
    dico[model.START_TAG] = -1
    dico[model.STOP_TAG] = -2
    tag_to_id, id_to_tag = create_mapping(dico)
    print("Found %i unique named entity tags" % len(dico))
    return dico, tag_to_id, id_to_tag


def cap_feature(s):
    """
    Capitalization feature:
    0 = low caps
    1 = all caps
    2 = first letter caps
    3 = one capital (not first letter)
    """
    if s.lower() == s:
        return 0
    elif s.upper() == s:
        return 1
    elif s[0].upper() == s[0]:
        return 2
    else:
        return 3


def prepare_sentence(str_words, word_to_id, char_to_id, lower=False):
    """
    Prepare a sentence for evaluation.
    """
    def f(x): return x.lower() if lower else x
    words = [word_to_id[f(w) if f(w) in word_to_id else '<UNK>']
             for w in str_words]
    chars = [[char_to_id[c] for c in w if c in char_to_id]
             for w in str_words]
    caps = [cap_feature(w) for w in str_words]
    return {
        'str_words': str_words,
        'words': words,
        'chars': chars,
        'caps': caps
    }


def prepare_dataset(sentences, word_to_id, char_to_id, tag_to_id, lower=True):
    """
    Prepare the dataset. Return a list of lists of dictionaries containing:
        - word indexes
        - word char indexes
        - tag indexes
    """
    def f(x): return x.lower() if lower else x
    data = []
    for sentence in sentences:
        str_words = [w[0] for w in sentence]
        words = [word_to_id[f(w) if f(w) in word_to_id else '<UNK>']
                 for w in str_words]
        # Skip characters that are not in the training set
        chars = [[char_to_id[c if c in char_to_id else '<UNK>'] for c in w]
                 for w in str_words]
        caps = [cap_feature(w) for w in str_words]
        tags = [tag_to_id[w[-1]] for w in sentence]
        data.append({
            'str_words': str_words,
            'words': words,
            'chars': chars,
            'caps': caps,
            'tags': tags,
        })
    return data


def augment_with_pretrained(dictionary, ext_emb_path, words):
    """
    Augment the dictionary with words that have a pretrained embedding.
    If `words` is None, we add every word that has a pretrained embedding
    to the dictionary, otherwise, we only add the words that are given by
    `words` (typically the words in the development and test sets.)
    """
    print('Loading pretrained embeddings from %s...' % ext_emb_path)
    assert os.path.isfile(ext_emb_path)

    # Load pretrained embeddings from file
    pretrained = set([
        line.rstrip().split()[0].strip()
        for line in open(ext_emb_path, 'r', encoding='utf-8')
    ])

    # We either add every word in the pretrained file,
    # or only words given in the `words` list to which
    # we can assign a pretrained embedding
    if words is None:
        for word in pretrained:
            if word not in dictionary:
                dictionary[word] = 0
    else:
        for word in words:
            if any(x in pretrained for x in [
                word,
                word.lower(),
                re.sub('\d', '0', word.lower())
            ]) and word not in dictionary:
                dictionary[word] = 0

    word_to_id, id_to_word = create_mapping(dictionary)
    return dictionary, word_to_id, id_to_word


def pad_seq(seq, max_length, PAD_token=0):
    # add pads
    seq += [PAD_token for i in range(max_length - len(seq))]
    return seq















