import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, balanced_accuracy_score, precision_score, recall_score
from sklearn.model_selection import GridSearchCV, KFold

#Laad de gegevens op dezelfde manier als in uw XGBoost-model
path_to_file = "/Users/thibaudseveri/PycharmProjects/pythonProject/importdata/HR_log_data.xlsx"
df = pd.read_excel(path_to_file)

#Dataset X uniek maken op id en enkel de laatste activity behouden (=meest recente)
X = df.drop_duplicates(subset='id', keep='last')

#Dataset Y maken met alleen kolom 'act' erin
Y = X['act'].to_frame()

#Y-dataset omzetting naar 1 voor leave en 0 voor al de andere act
Y.loc[Y['act'] == 'leave', 'act'] = 1
Y.loc[Y['act'] != 1, 'act'] = 0

#geeft dataframe X weer met enkel tabulaire snapshot variabelen, deze data gebruiken we om te predicten
X = X.drop(['next_act', 'act', 'time_start', 'time_end', 'contract_start', 'contract_end'], axis=1).copy()

#pd.get_dummies() toepassen om categorische waarden om te vormen naar numerieke, dus kolommen bijvoegen van elke mogelijke categorische om aan te geven of die 0 of 1 hebben bij die value
X_encoded = pd.get_dummies(X, columns=['prev_act', 'V01', 'V02', 'V03', 'V04', 'V05', 'V06', 'V07'])

# Verdeel de gegevens in train- en testsets op dezelfde manier als in uw XGBoost-model
X_train, X_test, Y_train, Y_test = train_test_split(X_encoded, Y, random_state=42, stratify=Y, test_size=0.2)

#zet waarden van 'act' om van object naar integers, is nodig voor xgboost
Y['act'] = pd.to_numeric(Y['act'])
Y_test['act'] = pd.to_numeric(Y_test['act'])
Y_train['act'] = pd.to_numeric(Y_train['act'])

#undersampling toepassen zodat we 80/20% hebben van non-leave en leave voor betere voorspelling
# oversample using SMOTE
sm = SMOTE(sampling_strategy=0.25, random_state=42)
X_train, Y_train = sm.fit_resample(X_train, Y_train)

# Define the parameter grid
param_grid = {
    'n_estimators': [160], #160 best
    'max_depth': [12], #12 best
    'min_samples_split': [2], #2 best
    'max_features': [0.5] #0.5 best
}

# Create a random forest classifier object
rfc = RandomForestClassifier(random_state=42)

# Create the GridSearchCV object
grid_search = GridSearchCV(estimator=rfc, param_grid=param_grid, cv=5, n_jobs=-1, scoring='roc_auc')

# Fit the GridSearchCV object to the data
grid_search.fit(X_train, Y_train)

# Print the best parameters and best score
print("Best parameters: ", grid_search.best_params_)
print("Best score: ", grid_search.best_score_)

# Use the best parameters to train and evaluate the model
rfc_best = RandomForestClassifier(n_estimators=grid_search.best_params_['n_estimators'],
                                   max_depth=grid_search.best_params_['max_depth'],
                                   min_samples_split=grid_search.best_params_['min_samples_split'],
                                   max_features=grid_search.best_params_['max_features'],
                                   random_state=42)
#fit het rfc_best model
rfc_best.fit(X_train, Y_train)
#fit het rfc model
rfc.fit(X_train, Y_train)

# perform 10-fold cross-validation and print the results for different metrics
kfold = KFold(n_splits=10, shuffle=True, random_state=42)
accuracy_scores = cross_val_score(rfc_best, X_train, Y_train, cv=kfold)
print("Accuracy: %0.2f (+/- %0.2f)" % (accuracy_scores.mean(), accuracy_scores.std() * 2))

auc_scores = cross_val_score(rfc_best, X_train, Y_train, cv=kfold, scoring='roc_auc')
print("AUC: %0.2f (+/- %0.2f)" % (auc_scores.mean(), auc_scores.std() * 2))

precision_scores = cross_val_score(rfc_best, X_train, Y_train, cv=kfold, scoring='precision')
print("Precision: %0.2f (+/- %0.2f)" % (precision_scores.mean(), precision_scores.std() * 2))

recall_scores = cross_val_score(rfc_best, X_train, Y_train, cv=kfold, scoring='recall')
print("Recall: %0.2f (+/- %0.2f)" % (recall_scores.mean(), recall_scores.std() * 2))

y_test_pred = rfc_best.predict(X_test)

# Bereken nauwkeurigheid, precisie en recall
print("AUC: ", roc_auc_score(Y_test, y_test_pred))
print("Accuracy: {:.2f}".format(balanced_accuracy_score(Y_test, y_test_pred)))
print("Precision: {:.2f}".format(precision_score(Y_test, y_test_pred)))
print("Recall: {:.2f}".format(recall_score(Y_test, y_test_pred)))
