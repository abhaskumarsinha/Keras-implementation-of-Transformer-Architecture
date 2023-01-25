# Keras-implementation-of-Transformer-Architecture

[[`Jump to TPU Colab demo Notebook`]](https://github.com/abhaskumarsinha/Keras-implementation-of-Transformer-Architecture/blob/main/StackedTransformer_TPU%20output.ipynb) [[`Original Paper`]](https://arxiv.org/abs/1706.03762) [[`Transformer Huggingface`]](https://huggingface.co/docs/transformers/index)


<p align="justify">
This repository presents a Python-based implementation of the Transformer architecture, as proposed by Vaswani et al. in their 2017 paper "Attention is all you need." The implementation is a variant of the original model, featuring a bi-directional design similar to BERT and the ability to predict the right-most token in sequence-to-sequence tasks, similar to GPT. The model was built from scratch using the TensorFlow Keras library. This implementation can be utilized for various natural language processing tasks. The Neural Machine Translation task - from English to Hindi has been demonstrated.

*The code presented in the repository is unofficial and has been implemented for educational purposes to aid beginners in basics of NLP Transformer Architecture. Feel free to play around and design new models with it.*
</p><br>

## 1. Introduction

<p align="justify">
The code has a modular and functional-style implementation of the Transformer architecture that can be utilized for various Natural Language Processing (NLP) or Computer Vision tasks. The model is built on top of the Keras TF Python library, which allows for easy customization and training. The modular design allows for easy adaptation to any NLP Transformer-based model, such as BERT, PaLM, GPT, T5, and others. The flexibility of the implementation allows for the utilization of custom optimizers, regularizers, training steps, and learning rate schedules, as well as the option to use Keras' default model.fit() training loop. This implementation is designed to be flexible and adaptable to a wide range of NLP and Computer Vision tasks and can be easily customized to meet specific requirements.
</p>


### 1.1. Implementation details

<p align="justify">
In this study, a sample Machine Translation task from English to Hindi is used to demonstrate the functionality of the proposed Transformer implementation. The model utilizes a vocabulary of 5,374 word-tokens, along with 2 special use tokens [PAD] and [UNK] for padding and unknown words, respectively, for both Hindi and English. The embedding vector size of the tokens in both languages is 64 dimensions( <img src="https://latex.codecogs.com/svg.latex?\&space;d_{\text{model}}" title=" d_{\text{model}}" />). The multi-head attention ( <img src="https://latex.codecogs.com/svg.latex?\&space;d_k" title=" d_k" />) is set to 8, and the model is capable of processing a token count of 64 at a time. The Transformer architecture is composed of stacks of seven encoders and decoders, resulting in a total of 25 million parameters. This implementation demonstrates the ability to adapt the Transformer architecture to a Machine Translation task and the flexibility to customize the model for specific requirements.</p>

<p align="justify">
It is important to note that in this implementation, masking has been removed from the dot-product attention mechanism in order to preserve bi-directionality in the model. This allows the model to read the input sentence from both directions at each instance, enabling it to generate the next word in the sequence. Additionally, the model is designed to generate one token at a time, utilizing the information from both directions to make its predictions. This approach allows the model to take into account the context of the entire sentence when generating each word, resulting in a more accurate translation.
</p>

### 1.2 Dataset

<p align="justify">
the HindiEnCorp dataset [1] was used for the Machine Translation task, which originally contains 278,000 sentences with their corresponding English counterparts and alignment scores. For demonstration purposes, the model was trained on only the first 10,000 sentences of the dataset. It should be noted that this is not a sufficient amount of data for benchmarking and it is recommended to use 200,000 dataset instances for stable translation results. No pre-training was employed to the model. The Hindi sentences of the dataset were used to create the input sequences, which were padded to 64 tokens, and the n+1-th word was used as the output in one-hot encoding format. This approach allows the model to learn the relationship between the input sequence and the corresponding translation.
</p>


#### 1.2.1 Tokenization

<p align="justify">
We employed Byte-Pair Encoding (BPE) tokenization [2] from the subword-nmt library with the number of symbols (-s switch) set to 5000. This caused a significant imbalance between the English and corresponding Hindi translation sentences, particularly in terms of the number of tokens. However, for demonstration purposes, this issue can be ignored. BPE tokenization is a popular method for NLP tasks as it can help to reduce the number of unique tokens in the vocabulary and handle out-of-vocabulary words. It's good to note that this tokenization method is not perfect, and in practice, it may be necessary to adjust the number of symbols or use other tokenization methods to achieve better results. Next, we use `tf.keras.layers.TextVectorization()` in Keras to convert the strings to text to integer vectors, which when passed through the layer of embedding matrices, results the required tensors for the model.
</p>
  
### 1.3 Usage

<p align="justify">
In this implementation, the `\dataset\` path requires two separate files for Hindi and English sentences, where the i-th line of the English file corresponds to the i-th line of the Hindi translation. The target language, Hindi, is referred to as the "Secondary" language and the source language, English, is referred to as the "Primary" language. The `dataset.create_dataset()` function automatically compiles the sentences of the primary and secondary languages as required. The compiled data can be accessed through the following variables: `dataset.encoder_inputs`, `dataset.decoder_inputs`, and `dataset.output_vectors` for primary language input to the encoder (English), secondary language input to the decoder (Hindi), and secondary language word one-hot encoding output (Hindi), respectively.</P>

<p align="justify">
The `Encoder()`, `Decoder()` are used to create the transformer encoders and decoders respectively. Note that we are using special Multi-head attention here in the implementation using loops in contrast to the original multi-head attention implementation by native TensorFlow library.
</p>

## 2. Results
### 2.1. Graphs
The convergence training graph:
![Epoch vs Training Loss Graph](https://github.com/abhaskumarsinha/Keras-implementation-of-Transformer-Architecture/raw/main/plot.svg)



<!-- ### Convergence Tables -->
### 2.2. Sample Outputs

> politicians do not have permission to do what needs to be done. <br>
> राजनीतिज्ञों के पास जो कार्य करना चाहिए, वह कि कि रहा कि न है करना , प्रकार वह होते हैं | हैं सकती है चाहिए । चाहिए । चाहिए चाहिए . . . . है । । । . . . . है । . . . । । . । . । । । । । । । ।

> This percentage is even greater than the percentage in India.<br>
> यह प्रतिशत भारत में हिन्दुओं प्रतिशत से अधिक है। है। है। है। है। है। है। है। है। है। है। है। है। है। है। है। है। है। की है। हैं। हैं। हैं। हैं। हैं। हैं। हैं। हैं। हैं। हैं। हैं। हैं। हैं। हैं। हैं। हैं। हैं। हैं। हैं। हैं। हैं। हैं। हैं। हैं। हैं। हैं। हैं। हैं। हैं। हैं। हैं। हैं। हैं। हैं। हैं।

> humans destroyed the commons that they depended on. <br>
> मानवों ने उन ही साझे संसाधनों को नष्ट किया जिन पर वो आधारित थे। थे। थे। थे। थे। थे। थे। थे। थे। थे। थे। थे। थे। थे। थे। थे। थे। थे। थे। थे। थे। थे। थे। थे। थे। थे। थे। थे। थे। थे। थे। थे। थे। थे। थे। थे। थे। थे। थे। थे। थे। थे। थे। थे। थे। थे। थे।

> Sanskrit is world's oldest language of vedas. <br>
> संस्कृत वका सेसेसेसेसेसेलिए में करें ही... ... ... ... ... ... ... ... ... हीहै। | है। है। है। नाम है। शामिल हैं है। है। है। है। शामिल है. है। है। है। है। है। है। है। है। है। है। है। है। है। है। है। है। है। है। है। है। है। | है। | रहे रहे

> In Indian culture, Holy texts have a special importance, and the Purans are among the most important of all the holy texts. <br>
> भारतीय जीवन-धारा में जिन ग्रन्थों का महत्वपूर्ण स्थान है उनमें पुराण भक्ति-ग्रंथों के रूप में बहुत महत्वपूर्ण माने जाते हैं। हैं। हैं। हैं। हैं। हैं। हैं। हैं। हैं। हैं। हैं। हैं। हैं। हैं। हैं। हैं। हैं। हैं। हैं। हैं। हैं। हैं। हैं। हैं। हैं। हैं। हैं। हैं। हैं। हैं। हैं। हैं। हैं। हैं। हैं। हैं। हैं। हैं।

> In Indian culture, Holy texts are written in Sanskrit language in greater percentage <br>
> भारतीय आयु संस्कृत बिप्राप्त की के तततहो ही | होना है के दो तथा बी बी दो दो दो दो है, | है . और . . . और . और पर . और . . भी पर . पर करने . ) है . . . . . . . . . . . . . . . .

> India and China are countries in Asia <br>
> और और याही ों एक एक मयामममजुभर भर है, है, है हैं हूँ, हूँ, । । कोहै। सहेकोहै। है। हैं। हैं। हैं। हैं। हैं। हैं। हैं। हैं। हैं। हैं। हैं। हैं। हैं। हैं। हैं। हैं। हैं। हैं। हैं। हैं। हैं। हैं। हैं। हैं। हैं। हैं। हैं। । । । । । । । ।

> Ancient Sanskrit literature is extremely old, vast and diverse <br>
> संस्कृत का प्राचीन साहित्य अत्यन्त प्राचीन विशाल और विविधतापूर्ण है। है। है। है। है। है। है। है। में है। है। है। है। है। है। है. है. की है. दीहै. है. है. टटहै. है. में है. है. हुए ते है. से है. है. है. है. है. है. है. है. है. है. है. है. है. है. है. है. है. है.

> Ancient Sanskrit literature is extremely old religious creations in Hindi. <br>
> स्कृत एक दूसरा पूर्ण सहायता सहायता सम्का बोउपयोग उपयोग का ों स्ट है। का है। है। है। है। है। है। है। है। है। है। की होती है। ध्यान का है| है| है| है| है| है| है| है| है| है| है| है| है| है| है| है| है| है| है| है| है। है। है। है। है। है। है। है। है। है। है। है। है।

## 3. For commercial-use/deployment

<p align="justify">
The demonstration code **IS BY NO WAY** suitable for any commercial-use/deployment. Kindly, train a bigger version of the transformer with atleast few million examples to get reliable estimate of the translated language. Consider pre-training them in Hindi only texts and then training pre-trained version for translation or fine-tuning for other downstream tasks. Consider less-resource machine translation models & techniques instead in case of less training data
</p>

## 4. Projects

<p>
For CPU-only training, the model is perfectly capable of jobs requiring MC (Machine Comprehension), POS Tagging, NER (Named-Entity Recognition), Text generation etc. But for the case of tasks such as chatbots, translation, summarization, consider pre-training it over large corpus of data before training them with few hundred million example over very high-performance TPUs or distributed learning servers.
</p>


## 5. Conclusion

<p align="justify">
We therefore end our documentation leaving example notebooks for the readers to experiment and learn from our implementation. Feel free to open issues or open new PR for the project.
</p>


# Biblography
1. Bojar, Ondřej, et al. "Hindencorp-hindi-english and hindi-only corpus for machine translation." Proceedings of the Ninth International Conference on Language Resources and Evaluation (LREC'14). 2014.
2. Rico Sennrich, Barry Haddow and Alexandra Birch (2016): Neural Machine Translation of Rare Words with Subword Units Proceedings of the 54th Annual Meeting of the Association for Computational Linguistics (ACL 2016). Berlin, Germany.
3. Vaswani, Ashish, et al. "Attention is all you need." Advances in neural information processing systems 30 (2017).
4. Devlin, Jacob, et al. "Bert: Pre-training of deep bidirectional transformers for language understanding." arXiv preprint arXiv:1810.04805 (2018).
5. Radford, Alec, et al. "Improving language understanding by generative pre-training." (2018).
6. Raffel, Colin, et al. "Exploring the limits of transfer learning with a unified text-to-text transformer." J. Mach. Learn. Res. 21.140 (2020): 1-67.
7. Chowdhery, Aakanksha, et al. "Palm: Scaling language modeling with pathways." arXiv preprint arXiv:2204.02311 (2022).
8. Chorowski, Jan K., et al. "Attention-based models for speech recognition." Advances in neural information processing systems 28 (2015).

