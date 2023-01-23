import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math

# Section 3.2.1 Scaled Dot-product Attention
def scaled_dot_product_attention(query, key, value):
    
    key_query_product = tf.nn.softmax(tf.einsum("bij, bij -> bi", query, key)/tf.math.sqrt(tf.cast(query.shape[-2], dtype=tf.float32)))
    output = tf.einsum("bi, bij -> bij", key_query_product, value)
    
    return output

class MultiheadAttention(tf.keras.layers.Layer):
    def __init__(self, d_k = 8, model_embedding = 512):
        super(MultiheadAttention, self).__init__()
        self.d_k = d_k
        self.model_embedding = model_embedding
        self.query_projection = [None]*self.d_k
        self.key_projection = [None]*self.d_k
        self.value_projection = [None]*self.d_k
        self.attention_result = [None]*self.d_k
        self.final_dense = tf.keras.layers.Dense(self.model_embedding, activation = 'linear')
        
    def build(self, input_sizes):
        print(input_sizes)
        self.query_matrix = []
        self.key_matrix = []
        self.value_matrix = []
        
        for i in range(self.d_k):
            self.query_matrix.append(tf.keras.layers.Dense(self.model_embedding))
            self.key_matrix.append(tf.keras.layers.Dense(self.model_embedding))
            self.value_matrix.append(tf.keras.layers.Dense(self.model_embedding))
        
    def call(self, inputs):
        query, key, value = inputs
        
        for i in range(self.d_k):
            self.query_projection[i] = self.query_matrix[i](query)
            self.key_projection[i] = self.key_matrix[i](key)
            self.value_projection[i] = self.value_matrix[i](value)
        
        for i in range(self.d_k):
            self.attention_result[i] = scaled_dot_product_attention(self.query_projection[i], self.key_projection[i], self.value_projection[i])
        
        result_tensor = self.attention_result[0]
        
        for i in range(1, self.d_k):
            result_tensor = tf.concat([result_tensor, self.attention_result[i]], -1)
        
        result_tensor = self.final_dense(result_tensor)
        
        return result_tensor
    

class Encoder(tf.keras.layers.Layer):
    def __init__(self, d_k = 8, attention_function = MultiheadAttention, model_embedding = 512):
        super(Encoder, self).__init__()
        self.d_k = d_k
        self.normalize_layer_1 = tf.keras.layers.Normalization(axis = -1)
        self.normalize_layer_2 = tf.keras.layers.Normalization(axis = -1)
        self.embedding_size = model_embedding
        self.relu_layer = tf.keras.layers.Dense(self.embedding_size, activation = 'relu')
        self.linear_layer = tf.keras.layers.Dense(self.embedding_size, activation = 'linear')
        self.attention = attention_function(d_k = self.d_k, model_embedding = self.embedding_size)
    
    def build(self, input_shape):
        print(input_shape)
    
    def call(self, inputs):
        
        attention_layer_output = self.attention((inputs, inputs, inputs))
        attention_layer_output = self.normalize_layer_1(inputs + attention_layer_output)
        
        feedforward_net_output = self.linear_layer(self.relu_layer(attention_layer_output))
        feedforward_net_output = self.normalize_layer_2(attention_layer_output + feedforward_net_output)
        
        return attention_layer_output, feedforward_net_output
        

class Decoder(tf.keras.layers.Layer):
    def __init__(self, input_words = 64, attention_function = MultiheadAttention, model_embedding = 512, d_k = 8):
        super(Decoder, self).__init__()
        self.d_k = d_k
        self.input_words = 64
        self.model_embedding = model_embedding
        self.attention1 = attention_function(d_k = self.d_k, model_embedding = self.model_embedding)
        self.attention2 = attention_function(d_k = self.d_k, model_embedding = self.model_embedding)
        
        self.first_sublayer_norm = tf.keras.layers.Normalization(axis = -1)
        self.concatenate1 = tf.keras.layers.Concatenate(axis = -2)
        self.concatenate2 = tf.keras.layers.Concatenate(axis = -2)
        self.second_sublayer_norm = tf.keras.layers.Normalization(axis = -1)
        self.feedforward_relu = tf.keras.layers.Dense(self.model_embedding, activation = 'relu')
        self.feedforward_linear = tf.keras.layers.Dense(self.model_embedding, activation = 'linear')
        self.third_sublayer_norm = tf.keras.layers.Normalization(axis = -1)
        
        
    def build(self, input_shape):
        print(input_shape)
    def call(self, inputs):
        
        decoder_input, encoder_layer1_input, encoder_layer2_input = inputs
        
        first_sublayer_out = self.attention1((decoder_input, decoder_input, decoder_input))
        first_sublayer_out = self.first_sublayer_norm(decoder_input + first_sublayer_out)
        
        second_sublayer_output = self.attention2((first_sublayer_out,  encoder_layer1_input, encoder_layer2_input))
        second_sublayer_output = self.second_sublayer_norm(second_sublayer_output + first_sublayer_out)
        
        third_sublayer_out = self.feedforward_relu(self.feedforward_linear(second_sublayer_output))
        third_sublayer_out = self.third_sublayer_norm(second_sublayer_output + third_sublayer_out)
        
        return third_sublayer_out

def positional_function(words, embedding):
    pos = np.zeros((words, embedding))
    
    for i in range(words):
        for j in range(embedding):
            if j%2 == 0:
                pos[i, j] = math.sin(i/pow(10000, 2*j/(512)))
            else:
                pos[i, j] = math.cos(i/pow(10000, 2*j/(512)))
    
    return pos

class PositionalEmbedding(tf.keras.layers.Layer):
    def __init__(self, positional_function = positional_function, embedding_size = 512, words = 64):
        super(PositionalEmbedding, self).__init__()
        self.embedding_size = embedding_size
        self.words = words
        self.pos_mat = tf.cast(tf.convert_to_tensor(positional_function(self.words, self.embedding_size)), tf.float32)
        
    def build(self, input_sizes):
        print(input_sizes)
        
    def call(self, inputs):
        embed = tf.einsum("bij, ij -> bij", inputs, self.pos_mat)            
        return embed