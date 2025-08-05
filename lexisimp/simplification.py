import nltk
import spacy
import string
import inflect
import plural
import verb 
import pandas as pd
from utils import get_elmo_score, get_gram_score
from nltk.corpus import wordnet as wn
from cwi import calculate_complexity, calculate_synonym_complexities

# nltk.download('wordnet')
# input_sentences = input("Input sentence: ")
# tokens, labels, probs = calculate_complexity(input_sentences)

class Sentence:
    def __init__(self, tokenized, threshold, ignore_list):
        self.threshold = threshold
        self.tokenized = tokenized
        self.indices = list(enumerate(self.tokenized))
        self.pos_tags = nltk.pos_tag(self.tokenized)
        
        if ignore_list == []:
            self.ignore_index = [c for (a,b),(c,d) in zip(self.pos_tags, self.indices) if 'P' in b]
        else:
            self.ignore_index = ignore_list
        
        _, _, probs = calculate_complexity(self.tokenized)
 
        self.complex_words = [(a,b) for a,b in list(zip([a for a,b in self.indices], probs)) if b > self.threshold]
        self.complex_words = [(a,b) for a,b in self.complex_words if a not in self.ignore_index]
        self.complex_words = sorted(self.complex_words, key = lambda x: x[1], reverse=True)

    def add_ignore(self, item):
        self.ignore_index.append(item)
    
    def make_simplification(self, synonym, index):
        tokens = self.tokenized
    
        del tokens[index]

        for i,word in enumerate(synonym):
            tokens.insert((index + i), word)
            self.add_ignore(index)
            

        self.tokenized = tokens
    
        self.indices = list(enumerate(self.tokenized))
        self.pos_tags = nltk.pos_tag(self.tokenized)

        _, _, probs = calculate_complexity(self.tokenized)

        self.complex_words = [(a,b) for a,b in list(zip([a for a,b in self.indices], probs)) if b > self.threshold]
        self.complex_words = [(a,b) for a,b in self.complex_words if a not in self.ignore_index]
        self.complex_words = sorted(self.complex_words, key = lambda x: x[1], reverse=True)
        
class Word:
    def __init__(self, sentence_object, index):
        pos_tags = sentence_object.pos_tags
        self.token_sent = sentence_object.tokenized
        self.pos_sent = nltk.pos_tag(self.token_sent)
        self.word = pos_tags[index][0]
        self.pos = pos_tags[index][1]
        self.index = index
        self.synonyms = None
        self.tense = None
        self.lemma = None
        self.is_plural = True if (self.pos == 'NNS') else False

    def get_synonyms(self):
        spacy_module = spacy.load('en_core_web_sm')
        # nltk.download('wordnet)

        doc = spacy_module("".join([" "+i if not i.startswith("'") and i not in string.punctuation else i for i in self.token_sent]).strip())

        for token in doc:
            if str(token) == self.word:
                self.lemma = token.lemma_
        
        pos = wn.NOUN
        
        if 'V' in self.pos:
            pos=wn.VERB
            self.tense = verb.verb_tense(self.word)
        elif 'N' in self.pos:
            pos=wn.NOUN
        elif 'J' in self.pos:
            pos=wn.ADJ
        elif 'RB' in self.pos:
            pos=wn.ADV

        synsets = wn.synsets(self.lemma, pos=pos)
        self.synonyms = [lemma.replace("_", " ") for syn in synsets for lemma in syn.lemma_names()]
        self.synonyms.extend([lemma.name().split(".")[0].replace("_", " ") for syn in synsets for lemma in syn.similar_tos()]) # Incorporate similar-tos

        temp_set = []

        for word in self.synonyms:
            temp_set.append(word)

        temp_set = set(temp_set)

        temp_set = [[x] for x in temp_set]

        self.synonyms = temp_set

        if self.is_plural == True:
            p = inflect.engine()
            all_synonyms = []
            for synonym in self.synonyms:
                new_synonyms = []
                for word in synonym:
                    if p.singular_noun(word) is False:
                        new_synonyms.append(plural.noun_plural(word))
                    else:
                        new_synonyms.append(word)
                all_synonyms.append(new_synonyms)
                    

            self.synonyms = all_synonyms

        if self.tense != None:
            tense_synonyms = []
            for x in self.synonyms:
                multi_word = []
                for element in x:
                    try:
                        multi_word.append((verb.verb_conjugate(element, tense=self.tense, negate=False)))
                    except:
                        multi_word.append(element)
                tense_synonyms.append(multi_word)
                
            self.synonyms = tense_synonyms	

    def get_ranked_synonyms(self):
        if len(self.synonyms) > 1:
            synonym_scores = pd.DataFrame()
            synonym_scores['synonyms'] = self.synonyms
            synonym_scores['sem_sim'] = get_elmo_score(self.synonyms, self.token_sent, self.index)
            synonym_scores['complexity'] = calculate_synonym_complexities(self.synonyms, self.token_sent, self.index)
            synonym_scores['grammaticality'] = get_gram_score(self.synonyms, self.token_sent, self.pos_sent, self.index)
            #filtering process
            print(synonym_scores)
            synonym_scores = synonym_scores[synonym_scores['sem_sim']<0.30]
            synonym_scores = synonym_scores.sort_values(by=['complexity'])
            
            try:
                top_synomym = synonym_scores.synonyms.values[0]
                
            except:
                return [self.word]

            return top_synomym
        else:
            return [self.word]