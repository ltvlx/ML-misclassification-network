import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random
import os
import codecs
import json

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


np.set_printoptions(precision=1, suppress=True)





def split_train_test(df, class_names, train_size, i_seed):
    np.random.seed(i_seed)
    
    codes = np.array(df['code'])
    mask_train = np.zeros(codes.shape, dtype=bool)

    for _c in class_names:
        idx_c = np.where(codes == _c)[0]
        N = min(train_size, len(idx_c) >> 1)

        idx_train = np.random.choice(idx_c, N, replace=False)

        mask_train[idx_train] = True

    X_train = np.array(df['abst clean'][mask_train])
    Y_train = np.array(df['code'][mask_train])

    X_test = np.array(df['abst clean'][~mask_train])
    Y_test = np.array(df['code'][~mask_train])
    Z1_test = np.array(df['n cit'][~mask_train])
    Z2_test = np.array(df['date'][~mask_train])
    Z3_test = np.array(df['journal ID'][~mask_train])
    
    return X_train, Y_train, X_test, Y_test, Z1_test, Z2_test, Z3_test





algorithms = {
    "MLPC":   OneVsRestClassifier(MLPClassifier(alpha=0.001, max_iter=1000)),
    "SVC":    OneVsRestClassifier(svm.SVC(kernel='linear', probability=True)),
    "NB":     OneVsRestClassifier(MultinomialNB()),
    "LogReg": OneVsRestClassifier(LogisticRegression(solver='lbfgs', multi_class='auto', max_iter=300)),
    "LinSVC": OneVsRestClassifier(svm.LinearSVC())
}

fname_in = '../data/data_v3.xlsx'
path_out = 'results-all/'


print('Reading data... ', end='', flush=True)
with pd.ExcelFile(fname_in) as xls:
    df = pd.read_excel(xls, 'Sheet0')
print('success!', flush=True)

class_names = [code for code in set(df['code']) if len(df[df['code'] == code].index) >= 100]
df = df.loc[df['code'].isin(class_names)]


# t_sizes = [10, 50, 100, 250, 500, 1000, 2000]
t_sizes = [500]

for train_size in t_sizes:
    scores = {}
    print(train_size)
    for i_seed in range(1):
        print(f' seed={i_seed} split... ', end='', flush=True)
        X_train, Y_train, X_test, Y_test, Z1_test, Z2_test, Z3_test = split_train_test(df, class_names, train_size, i_seed)
        print(f'done! {len(class_names)} labels. Set sizes: training={len(X_train)}, test={len(X_test)}', flush=True)

        Y_train = label_binarize(Y_train, classes=class_names)
        Y_test_b = label_binarize(Y_test, classes=class_names)


        # for key in ['SVC']:
        # for key in ['MLPC', 'NB', 'LogReg']:
        for key in ['LinSVC']:
            print(f'  {key}; training... ', end='', flush=True)
            classifier = Pipeline([('vect', CountVectorizer()), ('tfidf', TfidfTransformer()), ('clf', algorithms[key])])
            classifier.fit(X_train, Y_train)
            print(f'done!', flush=True)

            print('    testing... ', end='', flush=True)
            if key in ['MLPC', 'NB']:
                y_scores = classifier.predict_proba(X_test)
            else:
                y_scores = classifier.decision_function(X_test)
            print(f'done!', flush=True)
            # print("%.2f; %.2f"%(np.min(y_scores), np.max(y_scores)))
            # print(y_scores.shape, y_scores[0])

            Y_classified = [class_names[np.argmax(_s)] for _s in  y_scores]

            df_out = pd.DataFrame(np.column_stack((Y_test, Z1_test, Z2_test, Z3_test, Y_classified)), index=None,
                                  columns=['subject', 'ncit', 'date', 'j_ID', 'classification'])
            with pd.ExcelWriter(f'{path_out}test-{i_seed:02d}-{key}-{train_size:04d}.xlsx') as writer:
                df_out.to_excel(writer, sheet_name='Sheet0', index=False)











