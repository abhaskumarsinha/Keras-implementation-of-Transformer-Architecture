import numpy as np
import re
from tqdm import tqdm
import tensorflow as tf



class Dataset:
    def __init__(self, primary_language, secondary_language, max_limit = 63, total_lines = 10000):
        
        self.max_limit = max_limit
        
        lines = open(primary_language, encoding='utf-8').read().splitlines()[:total_lines]
        self.primary_tokens = []
        for line in lines:
            for word in line.split():
                self.primary_tokens += [word]
        self.primary_tokens = list(set(self.primary_tokens))
        print("Primary Language Tokens: " + str(len(self.primary_tokens)))
        self.primary_lines = lines
                
        lines = open(secondary_language, encoding='utf-8').read().splitlines()[:total_lines]
        self.secondary_tokens = []
        for line in lines:
            for word in line.split():
                self.secondary_tokens += [word]
        self.secondary_tokens = list(set(self.secondary_tokens))
        print("Secondary Language Tokens: " + str(len(self.secondary_tokens)))
        self.secondary_lines = lines
    
    def create_lists(self, string):
        words = string.split()
        pyramid = []
        next_words = []
        
        pyramid += [""]
        next_words += [words[0]]
        
        for i in range(len(words) - 1):
            pyramid.append(" ".join(words[:i+1]))
            if i+1 < len(words):
                next_words.append(words[i+1])
        return pyramid, next_words
    
    def create_dataset(self):
        X = []
        Y = []
        output = []
        
        for i in tqdm(range(len(self.secondary_lines))):
            
            if len(self.primary_lines[i].split()) < self.max_limit and len(self.secondary_lines[i].split()) < self.max_limit:
                
                line = self.secondary_lines[i]
                
                pyramid, next_words = self.create_lists(line)
                if not len(pyramid) == len(next_words):
                    print("ERROR AT LINE " + str(i))
                X += pyramid
                output += next_words
                Y += [self.primary_lines[i]]*len(pyramid)
            
        
        self.X = X
        self.Y = Y
        self.output = output
        
        print("Total Lines added: " + str(len(X)))
        
        self.primary_vectorizer = tf.keras.layers.TextVectorization(standardize = None, vocabulary = self.primary_tokens, output_sequence_length = self.max_limit + 1)
        self.secondary_vectorizer = tf.keras.layers.TextVectorization(standardize = None, vocabulary = self.secondary_tokens, output_sequence_length = self.max_limit + 1)
        self.output_vectorizer = tf.keras.layers.TextVectorization(standardize = None, vocabulary = self.secondary_tokens, output_sequence_length = 1)
        
        self.encoder_inputs = self.secondary_vectorizer(X)
        self.decoder_inputs = self.primary_vectorizer(Y)
        self.output_vectors = self.secondary_vectorizer(output)[:, 0]
        
    def convert_to_primary(self, vectorized_text):
        # Convert vectorized text back to string
        vocab_index_list = self.primary_vectorizer.get_vocabulary() # to get the vocab index
        vocab_index = { int(i) : vocab_index_list[i] for i in range(len(vocab_index_list))}
        inv_vocab_index = {v: k for v, k in vocab_index.items()} # to invert the index
        original_text = " ".join([inv_vocab_index[int(i)] for i in vectorized_text.numpy()[0]])
        return original_text
    
    
    def convert_to_secondary(self, vectorized_text):
        vocab_index_list = self.secondary_vectorizer.get_vocabulary() # to get the vocab index
        vocab_index = { int(i): vocab_index_list[i] for i in range(len(vocab_index_list))}
        inv_vocab_index = {v: k for v, k in vocab_index.items()} # to invert the index
        original_text = " ".join([inv_vocab_index[int(i)] for i in vectorized_text.numpy()[0]])
        return original_text
    
    