import numpy as np
import pandas as pd
import xgboost as xgb
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import balanced_accuracy_score, precision_score, recall_score
from sklearn.model_selection import GridSearchCV, KFold
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score
import os
from sklearn.impute import SimpleImputer
import category_encoders as ce
from category_encoders import woe

# data inladen
path_to_file = "/Users/thibaudseveri/PycharmProjects/pythonProject/importdata/finalprocessorienteddataset.xlsx"
merged_df = pd.read_excel(path_to_file)

#XGBoost uitvoeren op merged_df

# Define the target columns Y
Y = merged_df['act_2021']
Y = Y.to_frame()

# Y-dataset omzetting naar 1 voor leave en 0 voor al de andere act
Y.loc[Y['act_2021'] == 'leave', 'act_2021'] = 1
Y.loc[Y['act_2021'] != 1, 'act_2021'] = 0

#feature columns X
X = merged_df[['prev_act_2021', 'V01_2021', 'V02_2021', 'V03_2021', 'V04_2021', 'V05_2021', 'V06_2021', 'V07_2021', 'V08_2021', 'V09_2021', 'V10_2021', 'V11_2021',
                'prev_act_2020', 'act_2020', 'V01_2020', 'V02_2020', 'V03_2020', 'V04_2020', 'V05_2020', 'V06_2020', 'V07_2020', 'V08_2020', 'V09_2020', 'V10_2020', 'V11_2020',
                'prev_act_2019', 'act_2019', 'V01_2019', 'V02_2019', 'V03_2019', 'V04_2019', 'V05_2019', 'V06_2019', 'V07_2019', 'V08_2019', 'V09_2019', 'V10_2019', 'V11_2019',
                'prev_act_2018', 'act_2018', 'V01_2018', 'V02_2018', 'V03_2018', 'V04_2018', 'V05_2018', 'V06_2018', 'V07_2018', 'V08_2018', 'V09_2018', 'V10_2018', 'V11_2018',
                'prev_act_2017', 'act_2017', 'V01_2017', 'V02_2017', 'V03_2017', 'V04_2017', 'V05_2017', 'V06_2017', 'V07_2017', 'V08_2017', 'V09_2017', 'V10_2017', 'V11_2017',
                'prev_act_2016', 'act_2016', 'V01_2016', 'V02_2016', 'V03_2016', 'V04_2016', 'V05_2016', 'V06_2016', 'V07_2016', 'V08_2016', 'V09_2016', 'V10_2016', 'V11_2016',
                'prev_act_2015', 'act_2015', 'V01_2015', 'V02_2015', 'V03_2015', 'V04_2015', 'V05_2015', 'V06_2015', 'V07_2015', 'V08_2015', 'V09_2015', 'V10_2015', 'V11_2015',
                'prev_act_2014', 'act_2014', 'V01_2014', 'V02_2014', 'V03_2014', 'V04_2014', 'V05_2014', 'V06_2014', 'V07_2014', 'V08_2014', 'V09_2014', 'V10_2014', 'V11_2014',
                'prev_act_2013', 'act_2013', 'V01_2013', 'V02_2013', 'V03_2013', 'V04_2013', 'V05_2013', 'V06_2013', 'V07_2013', 'V08_2013', 'V09_2013', 'V10_2013', 'V11_2013',
                'prev_act_2012', 'act_2012', 'V01_2012', 'V02_2012', 'V03_2012', 'V04_2012', 'V05_2012', 'V06_2012', 'V07_2012', 'V08_2012', 'V09_2012', 'V10_2012', 'V11_2012']]



# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, Y, random_state=42, stratify=Y, test_size=0.2)

#missing_cols = X_train.columns[X_train.isnull().any()]
#Print columns with missing values
#for col in missing_cols:
    #print(f"{col}: {X_train[col].isnull().sum()} missing values")

#numerical columns (V08-V11) with missing values vervangen met de mean van de column op X_train
X_train = X_train.fillna(value={'V08_2012': X_train['V08_2012'].mean(), 'V09_2012': X_train['V09_2012'].mean(), 'V10_2012': X_train['V10_2012'].mean(), 'V11_2012': X_train['V11_2012'].mean(),
                                'V08_2013': X_train['V08_2013'].mean(), 'V09_2013': X_train['V09_2013'].mean(), 'V10_2013': X_train['V10_2013'].mean(), 'V11_2013': X_train['V11_2013'].mean(),
                                'V08_2014': X_train['V08_2014'].mean(), 'V09_2014': X_train['V09_2014'].mean(), 'V10_2014': X_train['V10_2014'].mean(), 'V11_2014': X_train['V11_2014'].mean(),
                                'V08_2015': X_train['V08_2015'].mean(), 'V09_2015': X_train['V09_2015'].mean(), 'V10_2015': X_train['V10_2015'].mean(), 'V11_2015': X_train['V11_2015'].mean(),
                                'V08_2016': X_train['V08_2016'].mean(), 'V09_2016': X_train['V09_2016'].mean(), 'V10_2016': X_train['V10_2016'].mean(), 'V11_2016': X_train['V11_2016'].mean(),
                                'V08_2017': X_train['V08_2017'].mean(), 'V09_2017': X_train['V09_2017'].mean(), 'V10_2017': X_train['V10_2017'].mean(), 'V11_2017': X_train['V11_2017'].mean(),
                                'V08_2018': X_train['V08_2018'].mean(), 'V09_2018': X_train['V09_2018'].mean(), 'V10_2018': X_train['V10_2018'].mean(), 'V11_2018': X_train['V11_2018'].mean(),
                                'V08_2019': X_train['V08_2019'].mean(), 'V09_2019': X_train['V09_2019'].mean(), 'V10_2019': X_train['V10_2019'].mean(), 'V11_2019': X_train['V11_2019'].mean(),
                                'V08_2020': X_train['V08_2020'].mean(), 'V09_2020': X_train['V09_2020'].mean(), 'V10_2020': X_train['V10_2020'].mean(), 'V11_2020': X_train['V11_2020'].mean(),
                                'V08_2021': X_train['V08_2021'].mean(), 'V09_2021': X_train['V09_2021'].mean(), 'V10_2021': X_train['V10_2021'].mean(), 'V11_2021': X_train['V11_2021'].mean()})
#ervoor zorgen dat de kolomnamen gelijk blijven en niet random getallen worden
columns_train = X_train.columns
#vervang de missing categorische waarden met de meest voorkomende waarden in die kolom
imputer = SimpleImputer(strategy='most_frequent')
X_train[columns_train] = imputer.fit_transform(X_train[columns_train])

# convert NumPy array to DataFrame with column names to view the excel sheet
X_train = pd.DataFrame(X_train)

#numerical columns (V08-V11) with missing values vervangen met de mean van de column op X_test
X_test = X_test.fillna(value={'V08_2012': X_test['V08_2012'].mean(), 'V09_2012': X_test['V09_2012'].mean(), 'V10_2012': X_test['V10_2012'].mean(), 'V11_2012': X_test['V11_2012'].mean(),
                                'V08_2013': X_test['V08_2013'].mean(), 'V09_2013': X_test['V09_2013'].mean(), 'V10_2013': X_test['V10_2013'].mean(), 'V11_2013': X_test['V11_2013'].mean(),
                                'V08_2014': X_test['V08_2014'].mean(), 'V09_2014': X_test['V09_2014'].mean(), 'V10_2014': X_test['V10_2014'].mean(), 'V11_2014': X_test['V11_2014'].mean(),
                                'V08_2015': X_test['V08_2015'].mean(), 'V09_2015': X_test['V09_2015'].mean(), 'V10_2015': X_test['V10_2015'].mean(), 'V11_2015': X_test['V11_2015'].mean(),
                                'V08_2016': X_test['V08_2016'].mean(), 'V09_2016': X_test['V09_2016'].mean(), 'V10_2016': X_test['V10_2016'].mean(), 'V11_2016': X_test['V11_2016'].mean(),
                                'V08_2017': X_test['V08_2017'].mean(), 'V09_2017': X_test['V09_2017'].mean(), 'V10_2017': X_test['V10_2017'].mean(), 'V11_2017': X_test['V11_2017'].mean(),
                                'V08_2018': X_test['V08_2018'].mean(), 'V09_2018': X_test['V09_2018'].mean(), 'V10_2018': X_test['V10_2018'].mean(), 'V11_2018': X_test['V11_2018'].mean(),
                                'V08_2019': X_test['V08_2019'].mean(), 'V09_2019': X_test['V09_2019'].mean(), 'V10_2019': X_test['V10_2019'].mean(), 'V11_2019': X_test['V11_2019'].mean(),
                                'V08_2020': X_test['V08_2020'].mean(), 'V09_2020': X_test['V09_2020'].mean(), 'V10_2020': X_test['V10_2020'].mean(), 'V11_2020': X_test['V11_2020'].mean(),
                                'V08_2021': X_test['V08_2021'].mean(), 'V09_2021': X_test['V09_2021'].mean(), 'V10_2021': X_test['V10_2021'].mean(), 'V11_2021': X_test['V11_2021'].mean()})

#ervoor zorgen dat de kolomnamen gelijk blijven en niet random getallen worden
columns_test = X_test.columns
#vervang de missing categorische waarden met de meest voorkomende waarden in die kolom
X_test[columns_test] = imputer.fit_transform(X_test[columns_test])

# convert NumPy array to DataFrame with column names to view the excel sheet
X_test = pd.DataFrame(X_test)

#print(X_train)
#print(X_test)
#print(y_train)
#print(y_test)

#check if there are any missing values now
#print(X_test_imputed_df.isnull().sum().sum())

# make a list with your categorical features by dropping the numericals
categorical_variables = X.drop(columns=['V08_2012', 'V09_2012', 'V10_2012', 'V11_2012', 'V08_2013', 'V09_2013', 'V10_2013', 'V11_2013', 'V08_2014', 'V09_2014', 'V10_2014', 'V11_2014', 'V08_2015', 'V09_2015', 'V10_2015', 'V11_2015', 'V08_2016', 'V09_2016', 'V10_2016', 'V11_2016', 'V08_2017', 'V09_2017', 'V10_2017', 'V11_2017', 'V08_2018', 'V09_2018', 'V10_2018', 'V11_2018', 'V08_2019', 'V09_2019', 'V10_2019', 'V11_2019', 'V08_2020', 'V09_2020', 'V10_2020', 'V11_2020', 'V08_2021', 'V09_2021', 'V10_2021', 'V11_2021'])

#alle categorische waarden omzetten naar numerieke aan de hand van one-hot encoding
def convert_categorical_variables(X_train, y_train, X_test, categorical_variables, cat_encoder):
    if cat_encoder == 'onehot':
        ce_one_hot = ce.OneHotEncoder(
            cols=categorical_variables,
            use_cat_names=True)
        ce_one_hot.fit(X_train)
        X_train = ce_one_hot.transform(X_train)
        X_test = ce_one_hot.transform(X_test)

    elif cat_encoder == 'woe':
        # Use weight of evidence encoding: WOE = ln (p(1) / p(0))
        woe_encoder = woe.WOEEncoder(verbose=1, cols=categorical_variables)
        woe_encoder.fit(X_train, y_train)
        X_train = woe_encoder.transform(X_train)
        X_test = woe_encoder.transform(X_test)

    return X_train, X_test

# Select which encoding method you want to use (is passed to the function above)
cat_encoder = 'onehot'  # alternative: 'onehot' or 'woe'

#X_train en X_test die encoded zijn met one-hot encoding
X_train, X_test = convert_categorical_variables(X_train, y_train, X_test, categorical_variables, cat_encoder)

# Convert object dtype columns to numeric dtype columns
X_train[X_train.select_dtypes(include='object').columns] = X_train.select_dtypes(include='object').astype('float64')
X_test[X_test.select_dtypes(include='object').columns] = X_test.select_dtypes(include='object').astype('float64')
y_train = y_train.astype('float64')
y_test = y_test.astype('float64')


#oversampling toepassen zodat we 80/20% hebben van non-leave en leave voor betere voorspelling using SMOTE
sm = SMOTE(sampling_strategy=0.25, random_state=42)
X_train, y_train = sm.fit_resample(X_train, y_train)

#print(X_train)
#print(X_test)
#print(y_train)
#print(y_test)


#XGBoost toepassen zonder optimal values
clf_xgb = xgb.XGBClassifier(objective='binary:logistic', seed=42, early_stopping_rounds=10, eval_metric='auc')  # objective='binary:logistic' is voor de classificatie omdat het de logistic regression approach gebruikt
clf_xgb.fit(X_train, y_train, verbose=True, eval_set=[(X_test, y_test)])

GBC = GradientBoostingClassifier()

#optimale values zoeken
param_grid = {
    #'max_depth': [8, 10, 12],  # max_depth is about how many levels the tree has,( 10 the best)
    #'learning_rate': [0.3, 0.4, 0.5], #(0.4 best)
    #'gamma': [1, 1.5, 2], # (1.5 best)
}

# perform grid search to find the best hyperparameters using 5-fold cross-validation
grid_search = GridSearchCV(clf_xgb, param_grid=param_grid, cv=10, scoring='roc_auc', n_jobs=-1)
grid_search.fit(X_train, y_train, eval_set=[(X_test, y_test)])

# print the best hyperparameters found
print("Best hyperparameters:", grid_search.best_params_)

# print the best hyperparameters and their corresponding MSE score
print("Best hyperparameters: ", grid_search.best_params_)
print("Best negative MSE score: ", grid_search.best_score_)

# create a new XGBoost classifier with the best hyperparameters found
clf_xgb2 = xgb.XGBClassifier(seed=42, objective='binary:logistic',
                             gamma=1.5, learning_rate=0.4,
                             max_depth=10, subsample=0.9,
                             colsample_bytree=0.5, n_estimators=160,
                             early_stopping_rounds=10, eval_metric='auc')


# train the model with the best hyperparameters and evaluate its performance
clf_xgb2.fit(X_train, y_train, verbose=True, eval_set=[(X_test, y_test)])

#make and show the confusion matrix
ConfusionMatrixDisplay.from_estimator(clf_xgb2, X_test, y_test,
                                      display_labels=["Did not leave", "Left"])
plt.show()

# perform 10-fold cross-validation and print the results for different metrics
kfold = KFold(n_splits=10, shuffle=True, random_state=42)
accuracy_scores = cross_val_score(clf_xgb2, X_train, y_train, cv=kfold, fit_params={'eval_set': [(X_test, y_test)]})
print("Accuracy: %0.2f (+/- %0.2f)" % (accuracy_scores.mean(), accuracy_scores.std() * 2))

auc_scores = cross_val_score(clf_xgb2, X_train, y_train, cv=kfold, scoring='roc_auc', fit_params={'eval_set': [(X_test, y_test)]})
print("AUC: %0.2f (+/- %0.2f)" % (auc_scores.mean(), auc_scores.std() * 2))

precision_scores = cross_val_score(clf_xgb2, X_train, y_train, cv=kfold, scoring='precision', fit_params={'eval_set': [(X_test, y_test)]})
print("Precision: %0.2f (+/- %0.2f)" % (precision_scores.mean(), precision_scores.std() * 2))

recall_scores = cross_val_score(clf_xgb2, X_train, y_train, cv=kfold, scoring='recall', fit_params={'eval_set': [(X_test, y_test)]})
print("Recall: %0.2f (+/- %0.2f)" % (recall_scores.mean(), recall_scores.std() * 2))

#make the tree pretty
bst = clf_xgb2.get_booster()
for importance_type in ('weight', 'gain', 'cover', 'total_gain', 'total_cover'):
    print('%s: ' % importance_type, bst.get_score(importance_type=importance_type))

# make the nodes and leafs prettier
node_params = {'shape': 'box',
               'style': 'filled, rounded'
               }
leaf_params = {'shape': 'box',
               'style': 'filled',
               'fillcolor': '#e48038'}

graph_data = xgb.to_graphviz(clf_xgb2, num_trees=0, size="10,10",
                             condition_node_params=node_params,
                             leaf_node_params=leaf_params)

#get tree as a PDF
graph_data.view(filename='xgboost_tree_customer_churn')

#predict on test set
Y_predict = clf_xgb2.predict(X_test)
Y_predict2 = clf_xgb.predict(X_test)

#evaluate the model's performance
print("Accuracy: {:.2f}".format(balanced_accuracy_score(y_test, Y_predict)))
print("Precision: {:.2f}".format(precision_score(y_test, Y_predict)))
print("Recall: {:.2f}".format(recall_score(y_test, Y_predict)))
print("AUC: {:.2f}".format(roc_auc_score(y_test, Y_predict)))

print("Accuracy2: {:.2f}".format(balanced_accuracy_score(y_test, Y_predict2)))
print("Precision2: {:.2f}".format(precision_score(y_test, Y_predict2)))
print("Recall2: {:.2f}".format(recall_score(y_test, Y_predict2)))
print("AUC2: {:.2f}".format(roc_auc_score(y_test, Y_predict2)))

#importance of variables checken
var_columns = [c for c in X_train.columns if c not in ['act_2021', 'next_act']] #afhankelijke variabelen invullen hier
df_var_importance = pd.DataFrame({"Variable": var_columns, "Importance": clf_xgb2.feature_importances_}).sort_values(by="Importance", ascending=False)
print(df_var_importance[:10])
