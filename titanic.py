import os
import unicodedata
import numpy as np
import pandas as pd
from nameparser import HumanName # for extract mixed order first, last names
from fuzzyset import FuzzySet # for matching between wiki table names and Titanic roster names
from imputer import Imputer as KNNimputer #see: https://github.com/bwanglzu/Imputer.py
from boruta import BorutaPy # pip install git+https://github.com/scikit-learn-contrib/boruta_py.git
from sklearn.linear_model import LogisticRegression as LR
from sklearn.ensemble import RandomForestClassifier as RF
from sklearn.metrics import roc_auc_score as AUC

from adversarial_validation import mark_instances
from adversarial_validation import distinguish

# see: http://wikitable2csv.ggor.de/
regex_home_region = "(?:.*,){0,2}(.*)" # says: ignore first 0 to 2 matches if they exist, but capture remaining content
sep = ","
data = ["train.csv", "test.csv"]

titanic = pd.concat( pd.read_csv(filename, sep=sep) for filename in [os.path.join("./", "data", csv) for csv in data] )
titanic.reset_index(inplace=True) # concat'ed index from two data frames
df = pd.read_csv("./data/wiki_table.csv", sep=sep)

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

titanic.drop('index', axis=1, inplace=True)

# So now we have the full transformed data frame, with self referential columns and additional insight
# from people's place of birth. Let's do some analysis of variables...

## ... determine which variables together are the important
## note: probably not needed, validation score is so high
## see: https://github.com/scikit-learn-contrib/boruta_py
#feat_rf = RandomForestClassifier(n_jobs=-1, class_weight='auto', max_depth=5)
#feat_selector = BorutaPy(rf, n_estimators='auto', verbose=2, random_state=1)
#feat_selector.fit(titanic.loc[mask, imputation_columns].values, titanic.loc[mask, 'Survived'].values)
## could use .transform() but that works on .values (numpy matrix) instead of pandas df ...
#selected_cols = [i for (i, v) in zip(imputation_columns, feat_selector.support_) if v]

# ... construct adverserial train, validation set that should tend towards actual test scores
clfs = [RF(n_estimators=100, n_jobs=-1, verbose=False), LR()]
marked = mark_instances(train = titanic[~pd.isnull(titanic.Survived)],
                        test = titanic[pd.isnull(titanic.Survived)],
                        response_column='Survived')

# To prevent test/train indicator leakage we drop PassengerId since Ids part 839 or so are all
# test instances. Survived is dropped because it has NaNs in it and imputed would induce leakage :)
marked.drop(['PassengerId', 'Survived'], axis=1, inplace=True)
adversarial = distinguish(marked, clfs)

titanic['Is Test'] = adversarial['predicted label proba']
del adversarial # just needed the labeled probabilities

# Train and adversarially validate a model
# todo: fix silly copy message
instances = titanic[~pd.isnull(titanic.Survived)]
instances.sort_values(by='Is Test', axis='index', ascending=False, inplace=True)

validation_size = int(0.10 * len(instances))

train = instances.iloc[:-validation_size]
validate = instances.iloc[-validation_size:]

y_train = train.Survived
y_validate = validate.Survived

# could I keep Is Test?
x_train = train.select_dtypes(['number']).drop(['Survived', 'Is Test'], axis=1)
x_validate = validate.select_dtypes(['number']).drop(['Survived', 'Is Test'], axis=1)

clf = LR()
clf.fit(x_train, y_train)
auc = AUC(y_validate, clf.predict_proba(x_validate)[:,1])
# 0.9999

# I don't really believe, must be leakage somewhere, todo: take a hard look, if nothign, then submit predictions
