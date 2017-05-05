import pandas as pd
from sklearn import cross_validation as CV
from sklearn.linear_model import LogisticRegression as LR
from sklearn.ensemble import RandomForestClassifier as RF
from sklearn.metrics import roc_auc_score as AUC
from sklearn.metrics import accuracy_score as accuracy

# adapted from the always excellent fastml.com
# see: https://github.com/zygmuntz/adversarial-validation/tree/master/numerai

# example train, test file, assume test response column is last
train_file = './data/train.csv'
test_file = './data/test.csv'

label = 'label' # name of test/train indicator column

def mark_instances(train=train, test=test, response_column=-1, inplace=True, copy=True):
    """
    Construct combined dataframe with instances marked as from test or training set
    """
    ret = None# return variable, the combined data frame with train/test indicator column

    if isinstance(response_column, int):
        train.drop(train.columns[response_column], axis=1, inplace=inplace) #todo: remove inplace?
    else:
        train.drop(response_column, axis=1, inplace=inplace)

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

    if columns:
        _train = train[columns]

    _train = _train.select_dtypes(dtypes)

    if _train.isnull().values.any():
        if fill_func == 'median':
            fill_func = _train.median

        _train = _train.fillna(fill_func(), inplace=True)

    best_score = -1
    for clf in clfs:
        scores = CV.cross_val_score( clf, _train.drop(label, axis=1),
                                     train[label], scoring='roc_auc', cv=5)

        print("\tDistinguish: mean AUC: {:.2%}, std: {:.2%} \n".format( scores.mean(), scores.std()))
        print(clf)
        if best_score < scores.mean(): # could do weighted against .std?
            best_score = scores.mean()
            ret_scores = scores
            ret_clf = clf

    return (ret_scores, ret_clf)

test = pd.read_csv(test_file)
train = pd.read_csv(train_file)

clfs = [RF(n_estimators=100, n_jobs=-1, verbose=True), LR()]

# todo: incorporate this logic, https://github.com/zygmuntz/adversarial-validation/blob/master/numerai/sort_train.py
#
# looks like he basically runs a version of distinguish on stratifiedKFold, where the clf is fit on a subset
# and the remaining subset is used for AUC against clf.predict_proba()
# then the predictions are stored off
#
# This is done so that we can look at a histogram of probab on is a test example or not (would be flat if no data shift)
# additionally, or whats happenning, is that they're using the known test labels, looking at the proba, outputing a histogram
# as well as measn accuracy. This is nice to have btu I dont think we care as much.
#
# what we do want is the sorted training items by proba of being a test example so that we can properly validate. Even then
# we really only just need the proba values, so most of sort_train can be replaced with predict_proba and returna  column with those
