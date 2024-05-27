import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os, codecs, json

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import Pipeline

from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.neural_network import MLPClassifier
from sklearn.multiclass import OneVsRestClassifier

from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc

np.set_printoptions(precision=2, suppress=True)


def split_train_test(df, train_size, i_seed):
    np.random.seed(i_seed)

    codes = np.array(df['code'])
    mask_train = np.zeros(codes.shape, dtype=bool)

    class_names = []
    for _c in df['code'].unique():
        idx_c = np.where(codes == _c)[0]
        if len(idx_c) < 50:
            continue

        N = min(train_size, len(idx_c) >> 1)
        idx_train = np.random.choice(idx_c, N, replace=False)
        mask_train[idx_train] = True

        class_names.append(_c)

    X_train = np.array(df['abst clean'][mask_train])
    Y_train = np.array(df['code'][mask_train])

    X_test = np.array(df['abst clean'][~mask_train])
    Y_test = np.array(df['code'][~mask_train])
    Z_test = {
        'ncit': np.array(df['n cit'][~mask_train]),
        'date': np.array(df['date'][~mask_train]),
        'jID': np.array(df['journal ID'][~mask_train]),
        'eID': np.array(df['eID'][~mask_train]),
        'Ndisc_cit': np.array(df['multidis max'][~mask_train])
    }

    return np.array(class_names), X_train, Y_train, X_test, Y_test, Z_test


algorithms = {
    "MLPC": OneVsRestClassifier(MLPClassifier(hidden_layer_sizes=(100,), activation='relu', solver='lbfgs', alpha=0.001, max_iter=100)),
    "SVC": OneVsRestClassifier(svm.SVC(kernel='rbf', probability=True)),
    "NB": OneVsRestClassifier(MultinomialNB()),
    "LogReg": OneVsRestClassifier(LogisticRegression(solver='lbfgs', multi_class='auto', max_iter=300)),
    "LinSVC": OneVsRestClassifier(svm.LinearSVC())
}

print('Reading data... ', end='', flush=True)

df = pd.read_excel('../data/data_v3.xlsx')
print('success!', flush=True)

print(df)

path_out = 'results/'
if not os.path.exists(path_out):
    os.makedirs(path_out)

t_sizes = [500]
# t_sizes = [100, 250, 500, 1000]
n_seeds = 1
for train_size in t_sizes:
    scores = {}
    print(train_size)
    for i_seed in range(n_seeds):
        print(f' seed={i_seed} split... ', end='', flush=True)
        class_names, X_train, Y_train, X_test, Y_test, Z_test = split_train_test(df, train_size, i_seed)
        print(f'done! {len(class_names)} labels. Set sizes: training={len(X_train)}, test={len(X_test)}', flush=True)

        Y_train = label_binarize(Y_train, classes=class_names)
        print(np.sum(Y_train, axis=0))
        Y_test_b = label_binarize(Y_test, classes=class_names)

        # for key in ['SVC', 'LinSVC', 'NB', 'LogReg', 'MLPC']:
        for key in ['LinSVC']:
            print(f'  {key}; training... ', end='', flush=True)
            classifier = Pipeline(
                [('vect', CountVectorizer()), ('tfidf', TfidfTransformer()), ('clf', algorithms[key])])
            classifier.fit(X_train, Y_train)
            print(f'done!', flush=True)

            print('    testing... ', end='', flush=True)
            if key in ['MLPC', 'NB']:
                Y_scores = classifier.predict_proba(X_test)
            else:
                Y_scores = classifier.decision_function(X_test)
            Y_imax = np.argmax(Y_scores, axis=1)
            Y_classified = class_names[Y_imax]

            standard_score = (Y_scores - np.mean(Y_scores, axis=1, keepdims=True)) / \
                        (np.std(Y_scores, axis=1, keepdims=True) + 1e-8)

            certainty = np.max(standard_score, axis=1)

            multi_thresh = 2.5
            mask = standard_score > multi_thresh
            misclass_diversity = np.sum(mask, axis=1, keepdims=True)

            pd.DataFrame(Y_scores[:10000]).to_excel(path_out + f'yscores-{i_seed:02d}-{key}-{train_size:04d}.xlsx', sheet_name='Sheet0')
            print(f'done! {100 * np.sum(Y_test == Y_classified) / len(Y_classified):.2f}%', flush=True)

            df_out = pd.DataFrame(
                np.column_stack(
                    (Y_test, Z_test['ncit'], Z_test['date'], Z_test['jID'], Z_test['eID'],
                     Z_test['Ndisc_cit'], Y_classified, certainty, misclass_diversity)
                ),
                index=None,
                columns=['subject', 'ncit', 'date', 'jID', 'eID', 'Ndisc_cit', 'classification', 'certainty', 'misclass_diversity'])

            df_out.to_excel(path_out + f'test-{i_seed:02d}-{key}-{train_size:04d}.xlsx',
                            sheet_name='Sheet0', index=False)
