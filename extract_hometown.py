import os
import unicodedata
import numpy as np
import pandas as pd
from nameparser import HumanName # for extract mixed order first, last names
from fuzzyset import FuzzySet # for matching between wiki table names and Titanic roster names
from imputer import Imputer as KNNimputer #see: https://github.com/bwanglzu/Imputer.py
from boruta import BorutaPy # pip install git+https://github.com/scikit-learn-contrib/boruta_py.git
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

# see: http://wikitable2csv.ggor.de/
regex_home_region = "(?:.*,){0,2}(.*)" # says: ignore first 0 to 2 matches if they exist, but capture remaining content
sep = ","
data = ["train.csv", "test.csv"]

titanic = pd.concat( pd.read_csv(filename, sep=sep) for filename in [os.path.join("../", "data", csv) for csv in data] )
titanic.reset_index(inplace=True) # concat'ed index from two data frames
df = pd.read_csv("../data/wiki_table.csv", sep=sep)

# Remove extraneous descriptions from the Name field from wiki table names
extra_descriptions = ["and chauffeur,", "and cook,", "and maid,",
                      "and valet,", "and secretary,", "and governess,",
                      "and dragoman,", "and clerk,", "and nurse,",
                      "and manservant,", ",\[\d+?\]", "\[\d+?\]",
                      "\[Note \d+?\]"] # note: there is [Note \d] in there

for extra in extra_descriptions:
    df['Name'] = df['Name'].str.replace(extra, "", case=False)
    df['Hometown'] = df['Hometown'].str.replace(extra, "", case=False)# could just do last two extra items

# Refactor the wiki table data frame to be indexed by name and construct an indexer for
# determining what provided name best matches the wiki table index. We also transliterate
# the names, hometown to ascii (since the Titanic dataset is in ascii)
def transliterate(x):
    return str(unicodedata.normalize("NFKD", x).encode("ascii", 'ignore'), encoding ='utf8')

df.drop_duplicates('Name', inplace=True) # not sure why there are duplicates?

df['Name'] = df['Name'].apply(transliterate).str.strip()
# todo: assign death ration to rows (based on last Name?
df['Hometown'] = df['Hometown'].apply(transliterate).str.strip()

df.set_index("Name", inplace=True, verify_integrity=True)
birthplace_index = FuzzySet(df.index)

# Associate a minimal home region to each name in the Titanic passenger manifest
titanic['Home Region'] = titanic['Name'].apply(lambda x: df.loc[birthplace_index.get(x)[0][1]]['Hometown'])\
                                        .str.extract(regex_home_region, expand=False)\
                                        .str.strip()

#titanic['Home Region'] = titanic['Name'].apply(lambda x: df.loc[birthplace_index.get(x)[0][1]]['Hometown'])
#titanic['Home Region'] = titanic['Home Region'].str.extract(regex_home_region, expand=False).str.strip()

#replace known regions, top 5, with their greater region
titanic["Home Region"] = titanic['Home Region'].str.replace("New York City", "US")\
                                               .str.replace("UK", "England")\
                                               .str.replace("London", "England")\
                                               .str.replace("Smaland", "Sweden")\
                                               .str.replace("Cork", "Ireland")

# convert home region to rank order
tmp_df = titanic['Home Region'].value_counts()
tmp_df.iloc[:] = list(range(len(tmp_df))) # now in rank order, 0 is highest

titanic['Home Region (ranked)'] = titanic['Home Region'].map(tmp_df)

# Assign death ratio to rows (based on Home region)
survival_counts_region = pd.crosstab(titanic['Home Region'], titanic['Survived'])
survival_counts_region.columns = ["Died", "Lived"]

def get_survival(key, survival=survival_counts_region):
    ret, ret2 = np.nan, np.nan
    if key in survival.index:
        ret, ret2 = survival.loc[key].values
    return ret, ret2

tmp_df = pd.DataFrame(titanic['Home Region'].apply(get_survival)\
                                            .values\
                                            .tolist(),
                      columns=["Region Died", "Region Lived"]) # could I use crosstab approach, faster?

titanic[tmp_df.columns] = tmp_df

# assign number on ticket, ticket death ratios

# ... number on ticket
titanic['Ticket'] = titanic['Ticket'].str.extract("(\d*)$", expand=False)

tmp_df = titanic['Ticket'].value_counts()
titanic['People on Ticket'] = titanic['Ticket'].map(tmp_df)

# ... death ratios
tmp_df = pd.crosstab(titanic['Ticket'], titanic['Survived'])
tmp_df.columns = ["On Ticket Died", "On Ticket Lived"]
tmp_df.reset_index(inplace=True)

titanic = pd.merge(titanic, tmp_df, how='outer', on='Ticket')

titanic['Sex (int)'] = (titanic['Sex'] == 'male').astype(int)

# ... impute age with kNN imputation
# note: in practice one would do cross validation, proper cross frame analysis to assess the usefulness of the imputation
# but we'll assume it's better than what we had before (289 missing)
#
# Additionally this equally weights each variables which may be inappropriate here... but see above comment.
imputation_columns = ['Fare', 'Parch', 'Home Region (ranked)',
                      'Sex (int)', 'Region Died', 'Region Lived',
                      'People on Ticket', 'On Ticket Died', 'On Ticket Lived',
                      'Pclass', 'SibSp', 'Age']

impute = KNNimputer()
X_imputed = impute.knn(titanic[imputation_columns], column='Age', k=10)
# There is one value that remains nan (all neighbors were nan), so we impute the median
X_imputed[np.argwhere(np.isnan(X_imputed[:,0])), 0] = np.nanmedian(X_imputed[:, 0])
titanic['Age'] = X_imputed[:,]

del X_imputed # we're done with the imputed matrix

# So now we have the full transformed data frame, with self referential columns and additional insight
# from people's place of birth. Let's do some analysis of variables...

# see: https://github.com/scikit-learn-contrib/boruta_py
feat_rf = RandomForestClassifier(n_jobs=-1, class_weight='auto', max_depth=5)
feat_selector = BorutaPy(rf, n_estimators='auto', verbose=2, random_state=1)
feat_selector.fit(titanic.loc[mask, imputation_columns].values, titanic.loc[mask, 'Survived'].values)
# could use .transform() but that works on .values (numpy matrix) instead of pandas df ...
selected_cols = [i for (i, v) in zip(imputation_columns, feat_selector.support_) if v]


# ... now with selected columns we do a quick classification check, say logistic regression
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegressionCV

# so we know that we want to compare classifers, one good way is to compare AUC
# another point is that we should be doing adversarial validation, see: https://github.com/zygmuntz/adversarial-validation
# ... this would also be of benefit to the community, i think. Something to consider

# * case in point, actually, the next step woudl be to instead implement adversarial validation adn then throw
# automl at the train/test sets

rand_state = 42
fold = KFold(len(titanic)-1, n_folds=5, shuffle=True, random_state=rand_state)

searchCV = LogisticRegressionCV(
     Cs=list(np.power(10.0, np.arange(-10, 10)))
    ,penalty='l2'
    ,scoring='roc_auc'
    ,cv=fold
    ,random_state=rand_state
    ,max_iter=10000
    ,fit_intercept=True
    ,solver='newton-cg'
    ,tol=10
)
mask = titanic['Survived'].notnull()

searchCV.fit(titanic.loc[mask, filtered_cols].values, titanic.loc[mask, 'Survived'].values)
# getting IndexError: index 891 is out of bounds for axis 0 with size 891

# something is wrong between fold and the fit(), see
#In [154]: from sklearn.cross_validation import KFold
#.../cross_validation.py:44: DeprecationWarning: This module was deprecated in version 0.18 in favor of the model_selection module into which all the refactored classes and functions are moved. Also note that the interface of the new CV iterators are different from that of this module. This module will be removed in 0.20.
#  "This module will be removed in 0.20.", DeprecationWarning)
#
#

print ('Max auc_roc:', searchCV.scores_[1].max())

