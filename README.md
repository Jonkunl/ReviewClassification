# ReviewClassification

[Pandas](https://pandas.pydata.org/) library is used to read and resize to a balanced dataset.

* To generate the embeddings, following libraries are used.
  - [word2vec](https://radimrehurek.com/gensim/models/word2vec.html) from Gensim
  - [sent2vec](https://github.com/epfml/sent2vec) library
  - [bert-as-service](https://github.com/hanxiao/bert-as-service)
  
  - pre-trained models can be download from
    - word2vec : [GoogleNews-vectors-negative300.bin.gz](https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit)
    - sent2vec: [Pre-trained models](https://github.com/epfml/sent2vec#downloading-sent2vec-pre-trained-models)
    - BERT: [BERT-Base, Uncased](https://storage.googleapis.com/bert_models/2018_10_18/uncased_L-12_H-768_A-12.zip)

Data provided in the 'Dataset' folder are csv files containing reviews with their corresponding embedding vectors and labels, which can be used for training and predicting without further processing.

* Following Jupyter Notebooks are included:
  - s2v and Bert
    - This Notebook is used for classification of s2v and BERT embedding with either Neural Network classifier or SVM
  - word2vec
    - This Notebook is used for classification of w2v with either Neural Network classifier or SVM
  - Detaile instructions are included in the Notebook
 
Two python files are included to help with the embedding processing and classification. Parameters in the [NeuralNetClassifier.py] can be adjusted for further testing.

