import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn_crfsuite import CRF
from sklearn_crfsuite.metrics import flat_f1_score, flat_accuracy_score, flat_precision_score, flat_recall_score
from sklearn_crfsuite.metrics import flat_classification_report
from collections import Counter

# Load data frame
df = pd.read_csv('../ner_dataset/ner_dataset.csv', encoding="ISO-8859-1")

df = df.fillna(method='ffill')

# print first 10 rows
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


def word2features(sent, i):
    term = sent[i][0]
    postag = sent[i][1]

    features = {
        'bias': 1.0,
        'term.lower()': term.lower(),
        'term[-3:]': term[-3:],
        'term[-2:]': term[-2:],
        'term.isupper()': term.isupper(),
        'term.istitle()': term.istitle(),
        'term.isdigit()': term.isdigit(),
        'postag': postag,
        'postag[:2]': postag[:2],
    }
    if i > 0:
        term1 = sent[i-1][0]
        postag1 = sent[i-1][1]
        features.update({
            '-1:term.lower()': term1.lower(),
            '-1:term.istitle()': term1.istitle(),
            '-1:term.isupper()': term1.isupper(),
            '-1:postag': postag1,
            '-1:postag[:2]': postag1[:2],
        })
    else:
        features['BOS'] = True

    if i < len(sent)-1:
        term1 = sent[i+1][0]
        postag1 = sent[i+1][1]
        features.update({
            '+1:term.lower()': term1.lower(),
            '+1:term.istitle()': term1.istitle(),
            '+1:term.isupper()': term1.isupper(),
            '+1:postag': postag1,
            '+1:postag[:2]': postag1[:2],
        })
    else:
        features['EOS'] = True

    return features


def sent2features(sent):
    return [word2features(sent, i) for i in range(len(sent))]


def sent2labels(sent):
    return [label for token, postag, label in sent]

"""
def sent2tokens(sent):
    return [token for token, postag, label in sent]
"""

X = [sent2features(s) for s in sentences]
y = [sent2labels(s) for s in sentences]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# load the model
crf = CRF(algorithm='lbfgs', c1=0.1, c2=0.1, max_iterations=100, all_possible_transitions=True)

# train the model
crf.fit(X_train, y_train)

# Predicting on the test set.
y_pred = crf.predict(X_test)


# Performance
f1_score = flat_f1_score(y_test, y_pred, average='weighted')
print("F1 score: ", f1_score)

acc = flat_accuracy_score(y_test, y_pred)
print("Accuracy: ", acc)

rec = flat_recall_score(y_test, y_pred, average='weighted')
print("Recall: ", rec)

prec = flat_precision_score(y_test, y_pred, average='weighted')
print("Precision: ", prec)

report = flat_classification_report(y_test, y_pred)
print(report)


def print_transitions(trans_features):
    for(label_from, label_to), weight in trans_features:
        print("%-6s -> %-7s %0.6f" % (label_from, label_to, weight))


print("Top likely transitions:")
print_transitions(Counter(crf.transition_features_).most_common(20))

print("\nTop unlikely transitions:")
print_transitions(Counter(crf.transition_features_).most_common()[-20:])
