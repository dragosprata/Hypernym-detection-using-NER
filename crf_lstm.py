import pandas as pd
import numpy as np

from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.layers import LSTM, Dense, TimeDistributed, Embedding, Bidirectional
from keras.models import Model, Input
from keras_contrib.layers import CRF
from keras.callbacks import ModelCheckpoint

import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

from sklearn_crfsuite.metrics import flat_classification_report
from seqeval.metrics import precision_score, recall_score, f1_score, classification_report, accuracy_score


# Reading the csv file
df = pd.read_csv('../ner_dataset/ner_dataset.csv', encoding="ISO-8859-1")


# first 10 rows
# print(df.head(10))


# Sentence will be list of tuples with its tag and pos.
class Sentence(object):
    def __init__(self, df):
        self.n_sent = 1
        self.df = df
        self.empty = False
        agg = lambda s: [(term, pos, tag) for term, pos, tag in zip(s['Term'].values.tolist(),
                                                                    s['POS'].values.tolist(),
                                                                    s['Tag'].values.tolist())]
        self.grouped = self.df.groupby("Sentence #").apply(agg)
        self.sentences = [s for s in self.grouped]

    def get_text(self):
        try:
            s = self.grouped['Sentence: {}'.format(self.n_sent)]
            self.n_sent += 1
            return s
        except:
            return None


getter = Sentence(df)

# all sentences from data set
sentences = getter.sentences

# Defining parameters for LSTM network
# number of data points passed in each iteration
batch_size = 64
# passes through entire data set
epochs = 8
# maximum length of review
max_len = 75
# dimension of embedding vector
embedding = 40

# get unique words and labels from data
terms = list(df['Term'].unique())
tags = list(df['Tag'].unique())

# Dictionary term:index pair
# term is key and its value is corresponding index
term_to_index = {term: i + 2 for i, term in enumerate(terms)}
term_to_index["UNK"] = 1
term_to_index["PAD"] = 0

# Dictionary tag:index pair
# tag is key and value is index.
tag_to_index = {tag: i + 1 for i, tag in enumerate(tags)}
tag_to_index["PAD"] = 0

index2term = {i: term for term, i in term_to_index.items()}
index2tag = {i: term for term, i in tag_to_index.items()}

# print("The term obstruction is identified by the index: {}".format(term_to_index["obstruction"]))
# print("The label Risk-Factor is identified by the index: {}".format(tag_to_index["Risk-Factor"]))


# Each sentence will be convert into list of index from list of tokens
X = [[term_to_index[term[0]] for term in sent] for sent in sentences]

# Padding each sequence to have same length  of each word
X = pad_sequences(maxlen=max_len, sequences=X, padding="post", value=term_to_index["PAD"])

# Convert label to index
y = [[tag_to_index[term[2]] for term in sent] for sent in sentences]

# padding
y = pad_sequences(maxlen=max_len, sequences=y, padding="post", value=tag_to_index["PAD"])

num_tag = df['Tag'].nunique()
# One hot encoded labels
y = [to_categorical(i, num_classes=num_tag + 1) for i in y]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15)

print("Size of training input data : ", X_train.shape)
print("Size of training output data : ", np.array(y_train).shape)
print("Size of testing input data : ", X_test.shape)
print("Size of testing output data : ", np.array(y_test).shape)


print('First sentence before processing: \n', ' '.join([term[0] for term in sentences[0]]))
print('First sentence after processing: \n ', X[0])

# First label before and after processing.
print('First label before processing : \n', ' '.join([term[2] for term in sentences[0]]))
print('First label after processing : \n ', y[0])


# Bidirectional LSTM-CRF Network
num_tags = df['Tag'].nunique()

# Architecture
input = Input(shape=(max_len,))
model = Embedding(input_dim=len(terms) + 2, output_dim=embedding, input_length=max_len)(input)
model = Bidirectional(LSTM(units=50, return_sequences=True, recurrent_dropout=0.1))(model)
model = TimeDistributed(Dense(50, activation="relu"))(model)
crf = CRF(num_tags+1)  # CRF layer
out = crf(model)  # output

model = Model(input, out)
model.compile(optimizer="rmsprop", loss=crf.loss_function, metrics=[crf.accuracy])

model.summary()

# check and save the best model performance
checkpointer = ModelCheckpoint(filepath='../eval/Mymodel.h5', verbose=0, mode='auto', save_best_only=True, monitor='val_loss')

history = model.fit(X_train, np.array(y_train), batch_size=batch_size, epochs=epochs, validation_split=0.1, callbacks=[checkpointer])

history.history.keys()

acc = history.history['crf_viterbi_accuracy']
val_acc = history.history['val_crf_viterbi_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
plt.figure(figsize=(8, 8))
epochs = range(1, len(acc) + 1)
plt.plot(epochs, acc, 'bo', label='acuratețe de antrenare')
plt.plot(epochs, val_acc, 'b', label='acuratețe de validare')
# plt.title('Acuratețea la antrenare și validare')
plt.legend()
plt.show()


plt.figure(figsize=(8, 8))
plt.plot(epochs, loss, 'bo', label='pierderea de antrenare')
plt.plot(epochs, val_loss, 'b', label='pierderea de validare')
# plt.title('Pierderea la antrenare și validare')
plt.legend()
plt.show()

# Evaluation
y_pred = model.predict(X_test)
y_pred = np.argmax(y_pred, axis=-1)
y_test_true = np.argmax(y_test, -1)

# Convert the index to tag
y_pred = [[index2tag[i] for i in row] for row in y_pred]
y_test_true = [[index2tag[i] for i in row] for row in y_test_true]

print("F1-score is : {:.1%}".format(f1_score(y_test_true, y_pred)))
print("Accuracy is : {:.1%}".format(accuracy_score(y_test_true, y_pred)))
print("Precision is : {:.1%}".format(precision_score(y_test_true, y_pred)))
print("Recall is : {:.1%}".format(recall_score(y_test_true, y_pred)))

report = flat_classification_report(y_pred=y_pred, y_true=y_test_true)
print(report)
