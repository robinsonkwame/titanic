import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn import cross_validation as CV
# use model_selection CV
from sklearn.linear_model import LogisticRegression as LR
from sklearn.ensemble import RandomForestClassifier as RF
from sklearn.metrics import roc_auc_score as AUC
from sklearn.metrics import accuracy_score as accuracy
from sklearn.base import clone
from sklearn.utils import shuffle

# adapted from the always excellent fastml.com
# see: https://github.com/zygmuntz/adversarial-validation/tree/master/numerai

# example train, test file, assume test response column is last
train_file = './data/train.csv'
test_file = './data/test.csv'

label = 'label' # name of test/train indicator column

def mark_instances(train, test, response_column=-1, copy=True):
    """
    Construct combined dataframe with instances marked as from test or training set
    """
    ret = None# return variable, the combined data frame with train/test indicator column

    train.is_copy = False
    test.is_copy = False # to remove indexing view versus copy warning

    if isinstance(response_column, int):
        train.drop(train.columns[response_column], axis=1, inplace=True) #todo: remove inplace?
    else:
        train.drop(response_column, axis=1, inplace=True)

    train[label] = 0
    test[label] = 1

    ret = pd.concat( (train, test), copy=copy )

    return ret

def distinguish(train, clfs, dtypes=['number'], columns=None, fill_func='median'):
    """
    Given the train data frame, dtype/columns subset, learn a classifier to
    predict whether a row is a test row or not.

    Includes convienence logic to impute values in dataframe if any nan are present
    """
    ret_scores = None
    ret_clf = None

    _train = train.copy() # note: doubles memory :-/
    # shuffle(...)? Sincew we use random CV splits I don't think we have to shuffle the data frame

    if columns:
        _train = train[columns]

    _train = _train.select_dtypes(dtypes)

    if _train.isnull().values.any():
        if fill_func == 'median':
            fill_func = _train.median

        _train = _train.fillna(fill_func(), inplace=True)

    assert _train.isnull().values.any() == False, "NaN detected in data frame!"

    _train.reset_index(inplace=True, drop=True)

    X = _train.drop(label, axis=1)# I hope these are pointers, not memory copies
    Y = _train[label]

    # To appropriately generalize (and not overfit by learning and predicting on same set) we
    # predict on stratified CV splits and gradually fill in our probability predictions of an instance
    # being from the test set or not

    # We take the best predictions from the set of trained classifiers so that we can construct overall
    # predictions in place. This incurs a training overhead of |clfs| unfortunately over |n_folds|/|_train|

    cv = CV.StratifiedKFold(Y,
                            n_folds = 5,
                            shuffle = True,
                            random_state = 42)

    for fold, (train_idx, test_idx) in enumerate(cv):
        print('\t Fold: {}'.format(fold))

        x_train = X.iloc[train_idx]
        x_test = X.iloc[test_idx]

        y_train = Y.iloc[train_idx]
        y_test = Y.iloc[test_idx]

        # find best clf for this train/test split and store off predictions
        best_score = -1
        best_clf = clfs[0]
        for clf in clfs:
            # ... assuming that .fit wipes out prior training (otherwise clone(clf)... needed)
            clf.fit(x_train, y_train)

            auc = AUC(y_test, clf.predict_proba(x_test)[:,1])

            if best_score <= auc:
                best_score = auc
                best_clf = clf # was clone(clf)

        # map the test_idx offsets to _train Index indices, assign probability of class 'label' == 1
        _train.loc[_train.index[test_idx], 'predicted label proba'] = best_clf.predict_proba(x_test)[:,1]
        print("AUC: {} ({})".format(best_score, best_clf.__class__.__name__))

    return _train # dataframe with predicted label proba, ~1 -> test example, ~0 -> train example

#test = pd.read_csv(test_file)
#train = pd.read_csv(train_file)

## example usage
## todo: convert to sklearn compatiable feature transformer
#clfs = [RF(n_estimators=100, n_jobs=-1, verbose=False), LR()]
#k = mark_instances(response_column='Survived')
#
## We want all numeric columns except for passenger ID, which leaks test labels since
## all values after 800 or are test labels
#k.drop('PassengerId', axis=1, inplace=True)
#
#b = distinguish(k, clfs)
#
## so now we can sort instances by predicted label and validate against those instances
## that more like the actual test examples
##
## these are the the indices, predicted probabilities
## b[b.label == 0].sort_values(by='predicted label proba', axis='index', ascending=False)['predicted label proba']
