import keras
from keras.models import Sequential
from keras.layers import *
from keras.optimizers import SGD
from sklearn.metrics import confusion_matrix, f1_score
import itertools
import numpy as np
from gensim.models.keyedvectors import KeyedVectors
from sklearn.metrics import roc_curve, auc
from numpy import interp
from itertools import cycle
import matplotlib.pyplot as plt


def fullyconnected(dim, x_train, y_train, x_validate, y_validate):
    # create model
    model = Sequential()
    # add model layers
    model.add(Dense(128, activation='relu', input_dim=dim))
    model.add(Dropout(0.5))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(5, activation='softmax'))
    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer=sgd,
                  metrics=['accuracy', keras.metrics.categorical_accuracy])
    model.fit(x_train, y_train,
              epochs=25,
              batch_size=128,
              validation_data=(x_validate, y_validate))

    return model

def print_results(y_test, y_pred):
    y_true = keras.utils.to_categorical(y_test, num_classes=5)
    Y_pred = np.argmax(y_pred, axis=1)
    Y_true = np.argmax(y_true, axis=1)
    print("F1 score: {}".format(f1_score(Y_true, Y_pred, average='micro')))
    print(confusion_matrix(Y_true, Y_pred))


def convo_fc(dim, model_path, x_train, x_validate, x_test, y_train, y_validate):
    w2v_model = KeyedVectors.load_word2vec_format('../shoes_w2v_model.bin', binary=True)

    lists = [x_train, x_validate, x_test]
    combined = list(itertools.chain.from_iterable(lists))

    from keras.preprocessing.text import Tokenizer
    t = Tokenizer()
    t.fit_on_texts(combined)
    vocab_size = len(t.word_index) + 1
    # integer encode the documents
    train_doc = t.texts_to_sequences(x_train)
    validate_doc = t.texts_to_sequences(x_validate)
    test_doc = t.texts_to_sequences(x_test)

    from keras.preprocessing.sequence import pad_sequences
    # pad documents to a max length of 4 words
    max_length = 200
    x_train = pad_sequences(train_doc, maxlen=max_length, padding='post')
    x_validate = pad_sequences(validate_doc, maxlen=max_length, padding='post')
    x_test = pad_sequences(test_doc, maxlen=max_length, padding='post')

    from numpy import zeros
    embedding_matrix = zeros((vocab_size, dim))
    for word, i in t.word_index.items():
        if word in w2v_model.vocab:
            embedding_matrix[i] = w2v_model[word]


    model = Sequential()
    e = Embedding(input_dim=vocab_size, output_dim=dim, weights=[embedding_matrix],
                  input_length=x_train.shape[1])
    model.add(e)
    model.add(Conv1D(filters=32, kernel_size=8, activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(5, activation='softmax'))
    print(model.summary())
    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer=sgd,
                  metrics=['accuracy'])

    model.fit(x_train, y_train, epochs=25, batch_size=128, validation_data=(x_validate, y_validate))
    y_pred = model.predict(x_test, x_test, batch_size=128)

    return y_pred

def roc_auc_plot(y_true, y_pred, title):
    n_classes = 5
    lw = 2

    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true[:, i], y_pred[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(y_true.ravel(), y_pred.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(y_true.ravel(), y_pred.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    # First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += interp(all_fpr, fpr[i], tpr[i])

    # Finally average it and compute AUC
    mean_tpr /= n_classes

    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    # Plot all ROC curves
    plt.figure()
    plt.plot(fpr["micro"], tpr["micro"],
             label='micro-average (area = {0:0.2f})'
                   ''.format(roc_auc["micro"]),
             color='deeppink', linestyle=':', linewidth=2)

    plt.plot(fpr["macro"], tpr["macro"],
             label='macro-average (area = {0:0.2f})'
                   ''.format(roc_auc["macro"]),
             color='navy', linestyle=':', linewidth=2)

    colors = cycle(['aqua', 'darkorange', 'cornflowerblue', 'red', 'yellow'])
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=lw,
                 label='class {0} (area = {1:0.2f})'
                       ''.format(i + 1, roc_auc[i]))

    plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend(loc="lower right")
    plt.show()
