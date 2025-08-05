# coding=utf-8
import optparse
import torch
import pickle
import string
from nltk.tokenize import word_tokenize, sent_tokenize
from loader import *
from utils import *

optparser = optparse.OptionParser()
optparser.add_option(
    "-g", '--use_gpu', default='1',
    type='int', help='whether or not to ues gpu'
)

optparser.add_option(
    '--model_path', default='models/v2/test',
    help='model path'
)

optparser.add_option(
    '--map_path', default='models/v2/mapping.pkl',
    help='model path'
)
optparser.add_option(
    '--char_mode', choices=['CNN', 'LSTM'], default='CNN',
    help='char_CNN or char_LSTM'
)

opts = optparser.parse_args()[0]

mapping_file = opts.map_path

with open(mapping_file, 'rb') as f:
    mappings = pickle.load(f)

word_to_id = mappings['word_to_id']
tag_to_id = mappings['tag_to_id']
id_to_tag = {k[1]: k[0] for k in tag_to_id.items()}
char_to_id = mappings['char_to_id']
parameters = mappings['parameters']
word_embeds = mappings['word_embeds']

use_gpu = opts.use_gpu == 1 and torch.cuda.is_available()
device = torch.device("cuda" if use_gpu else "cpu")

lower = parameters['lower']
zeros = parameters['zeros']

model = torch.load(opts.model_path, weights_only=False, map_location=torch.device("cpu"))
model_name = opts.model_path.split('/')[-1].split('.')[0]

model.to(device)
model.eval()

def make_inference(data):
    dwords = torch.LongTensor(data['words'])
    dcaps = torch.LongTensor(data['caps'])
    chars2 = data['chars']

    if parameters['char_mode'] == 'LSTM':
        chars2_sorted = sorted(chars2, key=lambda p: len(p), reverse=True)
        d = {}
        for i, ci in enumerate(chars2):
            for j, cj in enumerate(chars2_sorted):
                if ci == cj and not j in d and not i in d.values():
                    d[j] = i
                    continue
        chars2_length = [len(c) for c in chars2_sorted]
        char_maxl = max(chars2_length)
        chars2_mask = np.zeros((len(chars2_sorted), char_maxl), dtype='int')
        for i, c in enumerate(chars2_sorted):
            chars2_mask[i, :chars2_length[i]] = c
        chars2_mask = torch.LongTensor(chars2_mask)

    if parameters['char_mode'] == 'CNN':
        d = {}
        chars2_length = [len(c) for c in chars2]
        char_maxl = max(chars2_length)
        chars2_mask = np.zeros((len(chars2_length), char_maxl), dtype='int')
        for i, c in enumerate(chars2):
            chars2_mask[i, :chars2_length[i]] = c
        chars2_mask = torch.LongTensor(chars2_mask)

    prob, out = model(dwords.to(device), chars2_mask.to(device), dcaps.to(device), chars2_length, d)
    return prob, out

def calculate_complexity(tokenized): # wrapper function for inference
    data = prepare_sentence(tokenized, word_to_id, char_to_id, lower)
    prob, out = make_inference(data)
    prob = [row[1].item() for row in prob]

    return tokenized, out, prob

def calculate_complexities(indices, tokenized):
    _, _, probs = calculate_complexity(tokenized)
    word_probs = [probs[each_index] for each_index in indices]

    return float(sum(word_probs)/len(word_probs))

def calculate_synonym_complexities(synonyms, tokenized, index):
    word_complexities = []

    for entry in synonyms:
        indices = []
        tokenized_copy = tokenized.copy()

        del tokenized_copy[index]
        # if synonym contains multiple words we calculate average complexity of words
        for i, word in enumerate(entry):
            tokenized_copy.insert(index + i, word)
            indices.append(index + i)

        prob = calculate_complexities(indices, tokenized_copy)
        word_complexities.append(prob)

    return word_complexities

# # Inference and print results 
# input_sentences = input("Input sentence: ")
# input_sentences = sent_tokenize(input_sentences)
# tokens = []
# labels = []
# probs = []

# for sentence in input_sentences:
#     sentence = word_tokenize(sentence)
#     _, out, prob = calculate_complexity(sentence)
#     tokens.extend(sentence)
#     labels.extend(out)
#     probs.extend(prob)

# print()

# GREEN = '\033[32m'
# GRAY = '\033[90m'
# RESET = '\033[0m'
# space_punc = string.punctuation.replace("(", "")
# print(tag_to_id)

# for i in range(len(tokens)):
#     word = tokens[i]
#     label = labels[i]
#     prob = probs[i]
#     space = " "

#     if i+1 < len(tokens) and tokens[i+1] in space_punc:
#         space = ""

#     if label == 0: # Not difficult
#         print(f"{word} {GRAY}{prob}{RESET}", end=space)
#     else:
#         print(f"{GREEN}{word}{RESET} {GRAY}{prob}{RESET}", end=space)