# Keras-implementation-of-Transformer-Architecture
This repository presents a Python-based implementation of the Transformer architecture, as proposed by Vaswani et al. in their 2017 paper "Attention is all you need." The implementation is a variant of the original model, featuring a bi-directional design similar to BERT and the ability to predict the right-most token in sequence-to-sequence tasks, similar to GPT. The model was built from scratch using the TensorFlow Keras library. This implementation can be utilized for various natural language processing tasks. The Neural Machine Translation task - from English to Hindi has been demonstrated.

*The code presented in the repository is unofficial and has been implemented for educational purposes to aid beginners in basics of NLP Transformer Architecture. Feel free to play around and design new models with it.*

## Introduction

The code has a modular and functional-style implementation of the Transformer architecture that can be utilized for various Natural Language Processing (NLP) or Computer Vision tasks. The model is built on top of the Keras TF Python library, which allows for easy customization and training. The modular design allows for easy adaptation to any NLP Transformer-based model, such as BERT, PaLM, GPT, T5, and others. The flexibility of the implementation allows for the utilization of custom optimizers, regularizers, training steps, and learning rate schedules, as well as the option to use Keras' default model.fit() training loop. This implementation is designed to be flexible and adaptable to a wide range of NLP and Computer Vision tasks and can be easily customized to meet specific requirements.

### Implementation details

In this study, a sample Machine Translation task from English to Hindi is used to demonstrate the functionality of the proposed Transformer implementation. The model utilizes a vocabulary of 5,374 word-tokens, along with 2 special use tokens [PAD] and [UNK] for padding and unknown words, respectively, for both Hindi and English. The embedding vector size of the tokens in both languages is 64 dimensions( <img src="https://latex.codecogs.com/svg.latex?\&space;d_{\text{model}}" title=" d_{\text{model}}" />). The multi-head attention ( <img src="https://latex.codecogs.com/svg.latex?\&space;d_k" title=" d_k" />) is set to 8, and the model is capable of processing a token count of 64 at a time. The Transformer architecture is composed of stacks of seven encoders and decoders, resulting in a total of 25 million parameters. This implementation demonstrates the ability to adapt the Transformer architecture to a Machine Translation task and the flexibility to customize the model for specific requirements.

It is important to note that in this implementation, masking has been removed from the dot-product attention mechanism in order to preserve bi-directionality in the model. This allows the model to read the input sentence from both directions at each instance, enabling it to generate the next word in the sequence. Additionally, the model is designed to generate one token at a time, utilizing the information from both directions to make its predictions. This approach allows the model to take into account the context of the entire sentence when generating each word, resulting in a more accurate translation.

### Dataset

the HindiEnCorp dataset [1] was used for the Machine Translation task, which originally contains 278,000 sentences with their corresponding English counterparts and alignment scores. For demonstration purposes, the model was trained on only the first 10,000 sentences of the dataset. It should be noted that this is not a sufficient amount of data for benchmarking and it is recommended to use 200,000 dataset instances for stable translation results. No pre-training was employed to the model. The Hindi sentences of the dataset were used to create the input sequences, which were padded to 64 tokens, and the n+1-th word was used as the output in one-hot encoding format. This approach allows the model to learn the relationship between the input sequence and the corresponding translation.

#### Tokenization

We employed Byte-Pair Encoding (BPE) tokenization [2] from the subword-nmt library with the number of symbols (-s switch) set to 5000. This caused a significant imbalance between the English and corresponding Hindi translation sentences, particularly in terms of the number of tokens. However, for demonstration purposes, this issue can be ignored. BPE tokenization is a popular method for NLP tasks as it can help to reduce the number of unique tokens in the vocabulary and handle out-of-vocabulary words. It's good to note that this tokenization method is not perfect, and in practice, it may be necessary to adjust the number of symbols or use other tokenization methods to achieve better results. Next, we use `tf.keras.layers.TextVectorization()` in Keras to convert the strings to text to integer vectors, which when passed through the layer of embedding matrices, results the required tensors for the model.

### Usage

## Results
### Graphs
### Convergence Tables
### Sample Outputs

## For commercial-use/deployment

## Conclusion

