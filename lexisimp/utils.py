import os
import re
import numpy as np
from torch.nn import init
import tensorflow as tf
import tensorflow_hub as hub

elmo = hub.load("https://tfhub.dev/google/elmo/3")

models_path = "./models"
eval_path = "./evaluation"
eval_temp = os.path.join(eval_path, "temp")
eval_script = os.path.join(eval_path, "conlleval")


def get_name(parameters):
    """
    Generate a model name from its parameters.
    """
    l = []
    for k, v in parameters.items():
        if type(v) is str and "/" in v:
            l.append((k, v[::-1][: v[::-1].index("/")][::-1]))
        else:
            l.append((k, v))
    name = ",".join(["%s=%s" % (k, str(v).replace(",", "")) for k, v in l])
    return "".join(i for i in name if i not in "\/:*?<>|")


def set_values(name, param, pretrained):
    """
    Initialize a network parameter with pretrained values.
    We check that sizes are compatible.
    """
    param_value = param.get_value()
    if pretrained.size != param_value.size:
        raise Exception(
            "Size mismatch for parameter %s. Expected %i, found %i."
            % (name, param_value.size, pretrained.size)
        )
    param.set_value(np.reshape(pretrained, param_value.shape).astype(np.float32))


def create_mapping(dico):
    """
    Create a mapping (item to ID / ID to item) from a dictionary.
    Items are ordered by decreasing frequency.
    """
    sorted_items = sorted(dico.items(), key=lambda x: (-x[1], x[0]))
    id_to_item = {i: v[0] for i, v in enumerate(sorted_items)}
    item_to_id = {v: k for k, v in id_to_item.items()}
    return item_to_id, id_to_item


def zero_digits(s):
    """
    Replace every digit in a string by a zero.
    """
    return re.sub("\d", "0", s)

def insert_singletons(words, singletons, p=0.5):
    """
    Replace singletons by the unknown word with a probability p.
    """
    new_words = []
    for word in words:
        if word in singletons and np.random.uniform() < p:
            new_words.append(0)
        else:
            new_words.append(word)
    return new_words


def pad_word_chars(words):
    """
    Pad the characters of the words in a sentence.
    Input:
        - list of lists of ints (list of words, a word being a list of char indexes)
    Output:
        - padded list of lists of ints
        - padded list of lists of ints (where chars are reversed)
        - list of ints corresponding to the index of the last character of each word
    """
    max_length = max([len(word) for word in words])
    char_for = []
    char_rev = []
    char_pos = []
    for word in words:
        padding = [0] * (max_length - len(word))
        char_for.append(word + padding)
        char_rev.append(word[::-1] + padding)
        char_pos.append(len(word) - 1)
    return char_for, char_rev, char_pos


def create_input(data, parameters, add_label, singletons=None):
    """
    Take sentence data and return an input for
    the training or the evaluation function.
    """
    words = data["words"]
    chars = data["chars"]
    if singletons is not None:
        words = insert_singletons(words, singletons)
    if parameters["cap_dim"]:
        caps = data["caps"]
    char_for, char_rev, char_pos = pad_word_chars(chars)
    input = []
    if parameters["word_dim"]:
        input.append(words)
    if parameters["char_dim"]:
        input.append(char_for)
        if parameters["char_bidirect"]:
            input.append(char_rev)
        input.append(char_pos)
    if parameters["cap_dim"]:
        input.append(caps)
    if add_label:
        input.append(data["tags"])
    return input


def init_embedding(input_embedding):
    """
    Initialize embedding
    """
    bias = np.sqrt(3.0 / input_embedding.size(1))
    init.uniform_(input_embedding, -bias, bias)


def init_linear(input_linear):
    """
    Initialize linear transformation
    """
    init.xavier_normal_(input_linear.weight.data)
    init.normal_(input_linear.bias.data)


def adjust_learning_rate(optimizer, lr):
    """
    shrink learning rate for pytorch
    """
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


def init_lstm(input_lstm):
    """
    Initialize lstm
    """
    for param in input_lstm.parameters():
        if len(param.shape) >= 2:
            init.orthogonal_(param.data)
        else:
            init.normal_(param.data)

def get_elmo_embedding(sentences):
    embeddings = elmo.signatures["default"](tf.constant(sentences))["elmo"]
    return embeddings

def get_elmo_score(synonyms, tok_sentence, index):
    import nltk
    import numpy
    import scipy 
    
    vectors = get_elmo_embedding(tok_sentence)
    original_vector = vectors[index][0]

    distances = []

    for synonym in synonyms:
        new_sentence = tok_sentence.copy()
        del new_sentence[index]
        for i,word in enumerate(synonym):
            new_sentence.insert((index + i), word)

        new_vectors = get_elmo_embedding(new_sentence)
        if len(synonym) == 1:
            new_vector = new_vectors[index][0]
        else:
            phrase_vectors = []
            for i,word in enumerate(synonym):
                phrase_vectors.append(numpy.array(new_vectors[(index + i)][0]))
            new_vector = numpy.mean(phrase_vectors,axis=0)

        distances.append(scipy.spatial.distance.cosine(original_vector.numpy().tolist(), new_vector.numpy().tolist()))

    return distances

def get_sim_score(word, synonym):
    word_vector = word_embeddings.get(word)
    synonym_vector = word_embeddings.get(synonym)


def get_ngram(left, right):
    import pandas as pd
    two_grams = pd.read_csv('./ngram/w2_.txt', sep='\t', header=None, encoding='latin1')
    two_grams.columns = ["freq", "w1", "w2"]

    freq = two_grams[(two_grams["w1"]==left) & (two_grams["w2"]==right)].freq.values

    return freq


def get_gram_score(synonyms, tokenized, pos_tags, index):

    return_list = []

    left_bi = True
    right_bi = True

    original_word = tokenized[index]

    try:
        left_word = tokenized[index - 1]
        left_pos = pos_tags[index - 1]
    except:
        left_bi = False

    try:
        right_word = tokenized[index + 1]
        right_pos = pos_tags[index + 1]
    except:
        right_bi = False

    if right_word in [',','.',';',':'] or ('P' in right_pos[1]):
        right_bi = False

    if left_word in [',','.',';',':'] or ('P' in left_pos[1]):
        left_bi = False

    original_left = get_ngram(left_word,original_word)
    

    if  original_left.size == 0:
        left_bi = False
    if get_ngram(original_word,right_word).size == 0:
        right_bi = False



    for synonym in synonyms:
        if len(synonym) == 1:
            if left_bi:
                left_prob = get_ngram(left_word, synonym[0])
            if right_bi:
                right_prob = get_ngram(synonym[0], right_word)
        else:
            synonym_left = synonym[len(synonym)-1]
            synonym_right = synonym[0]

            if left_bi:
                left_prob = get_ngram(left_word, synonym_right)
            if right_bi:
                right_prob = get_ngram(synonym_left, right_word)

        if left_bi and right_bi:
            
            if (np.sum(left_prob) > 0) and (np.sum(right_prob) > 0):
                return 1
            else:
                return 0
        elif not left_bi:
            if right_bi:
                if right_prob > 0:
                    return 1
                else:
                    return 0

        elif not right_bi:
            if left_bi:
                if left_prob > 0:
                    return 1
                else:
                    return 0
