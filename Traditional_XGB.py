from imblearn.over_sampling import SMOTE
import pandas as pd
import xgboost as xgb
from sklearn.metrics import balanced_accuracy_score, roc_auc_score, precision_score, recall_score
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.model_selection import GridSearchCV

# data inladen
path_to_file = "/Users/thibaudseveri/PycharmProjects/pythonProject/importdata/HR_log_data.xlsx"
df = pd.read_excel(path_to_file)

#dataset X uniek maken op id en enkel de laatste activity behouden (=meest recente)
X = df
X.drop_duplicates(subset='id', keep='last', inplace=True)
#print(X)

#datasets scheiden van elkaar, dus dataframe Y maken met enkel kolom 'act' erin
Y = X['act'] # deze df is enkel 'act' kolom, dit is de data die we willen predicten
Y = Y.to_frame()
#print(Y)

#geeft dataframe X weer met enkel tabulaire snapshot variabelen, deze data gebruiken we om te predicten
X = df.drop(['next_act', 'act', 'time_start', 'time_end', 'contract_start',	'contract_end'], axis=1).copy()

# Y-dataset omzetting naar 1 voor leave en 0 voor al de andere act
Y.loc[Y['act'] == 'leave', 'act'] = 1
Y.loc[Y['act'] != 1, 'act'] = 0
#print(Y.head(15)['act'])

# pd.get_dummies() toepassen om categorische waarden om te zetten naar numerieke, dus kolommen bijvoegen van elke mogelijke categorische om aan te geven of die 0 of 1 hebben bij die value
X_encoded = pd.get_dummies(X, columns=['prev_act', 'V01', 'V02', 'V03', 'V04', 'V05', 'V06', 'V07'])

# XGBoost beginnen

#check how many leavers and non-leavers we have in our target column, 83,5% non-leavers and 16,5% leavers
#print(sum(Y['act'])/len(Y))

#print(Counter(Y_train1))

#X_train, X_test, Y_train, Y_test maken met zelfde verhouding van leavers en non-leavers
X_train, X_test, Y_train, Y_test = train_test_split(X_encoded, Y, random_state=42, stratify=Y, test_size=0.2) #stratify = Y zorgt ervoor dat de proportie van uw orignele dataset met leavers en non leavers  ook zo verdeeld wordt in uw train en test set, dus dat het percentage van zowel leavers als non-leavers gelijk zijn hier als in originele dataset

#checken of Y_test zelfde percentage geeft als hierboven, want mag niet veranderen
#print(Y_test['act'].value_counts()/len(Y_test['act']))

#print(X_train)
#print(Y_train)
#print(X_test)
#print(Y_test)

# zet waarden van 'act' om van object naar integers, is nodig voor xgboost
Y['act'] = pd.to_numeric(Y['act'])
Y_test['act'] = pd.to_numeric(Y_test['act'])
Y_train['act'] = pd.to_numeric(Y_train['act'])

#undersampling toepassen zodat we 80/20% hebben van non-leave en leave voor betere voorspelling
# oversample using SMOTE
sm = SMOTE(sampling_strategy=0.25, random_state=42)
X_train, Y_train = sm.fit_resample(X_train, Y_train)

#check number of leavers and non-leavers
#print(Y_test['act'].value_counts()/len(Y_test['act'])) # test set blijft de originele verhoudingen tussen leavers en non-leavers
#print(Y_train['act'].value_counts()/len(Y_train['act'])) # training set is de verhouding 80/20 nu voor betere predictions (undersampling)

#XGBoost toepassen zonder optimal values
clf_xgb = xgb.XGBClassifier(objective='binary:logistic', seed=42)  # objective='binary:logistic' is voor de classificatie omdat het de logistic regression aproach gebruikt
clf_xgb.fit(X_train, Y_train, verbose=True, early_stopping_rounds=10, eval_metric='aucpr', eval_set=[(X_test, Y_test)])  # de AUC gebruiken als maatstaf om te zien hoe goed de predictions zijn

#make and show the confusion matrix on not optimalized xgboost model
ConfusionMatrixDisplay.from_estimator(clf_xgb, X_test, Y_test,
                                      display_labels=["Did not leave", "Left"])
plt.show() # in this matrix we see that of the 329 people in our test set that did not leave, 318 (96,66%) were correctly classified. And of the 65 employees that left the company, 37 (56,92%) were correctly classified. So the XGBboost model was not awesome actually.

#optimale values zoeken
param_grid = {
    #'max_depth': [8, 10, 12],  # max_depth is about how many levels the tree has, (10 best)
    #'learning_rate': [0.05, 0.1, 1], #(0,1 best)
    #'gamma': [0.2, 0.5, 1], #(0.5 best)
}

# perform grid search to find the best hyperparameters using 5-fold cross-validation
#grid_search = GridSearchCV(clf_xgb, param_grid=param_grid, cv=5, scoring='roc_auc', n_jobs=-1)
#grid_search.fit(X_train, Y_train)

# print the best hyperparameters found
#print("Best hyperparameters:", grid_search.best_params_)

# create a new XGBoost classifier with the best hyperparameters found
clf_xgb2 = xgb.XGBClassifier(seed=42, objective='binary:logistic',
                             gamma=0.5, learning_rate=0.1,
                             max_depth=10, subsample=0.9,
                             colsample_bytree=0.5)

# train the model with the best hyperparameters and evaluate its performance
clf_xgb2.fit(X_train, Y_train, verbose=True, early_stopping_rounds=10,
             eval_metric='aucpr', eval_set=[(X_test, Y_test)])

#make and show confusion matrix of optimized xgboost model
ConfusionMatrixDisplay.from_estimator(clf_xgb2, X_test, Y_test,
                                      display_labels=["Did not leave", "Left"])
plt.show()

#de decision tree echt weergeven
clf_xgb2 = xgb.XGBClassifier(seed=42, objective='binary:logistic',
                             gamma=0.5, learning_rate=0.1,
                             max_depth=10, subsample=0.9,
                             colsample_bytree=0.5, n_estimators=180)
clf_xgb2.fit(X_train, Y_train)

# perform 10-fold cross-validation and print the results for different metrics
kfold = KFold(n_splits=10, shuffle=True, random_state=42)
accuracy_scores = cross_val_score(clf_xgb2, X_train, Y_train, cv=kfold)
print("Accuracy: %0.2f (+/- %0.2f)" % (accuracy_scores.mean(), accuracy_scores.std() * 2))

auc_scores = cross_val_score(clf_xgb2, X_train, Y_train, cv=kfold, scoring='roc_auc')
print("AUC: %0.2f (+/- %0.2f)" % (auc_scores.mean(), auc_scores.std() * 2))

precision_scores = cross_val_score(clf_xgb2, X_train, Y_train, cv=kfold, scoring='precision')
print("Precision: %0.2f (+/- %0.2f)" % (precision_scores.mean(), precision_scores.std() * 2))

recall_scores = cross_val_score(clf_xgb2, X_train, Y_train, cv=kfold, scoring='recall')
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

#evaluate the model's performance
print("Accuracy: {:.2f}".format(balanced_accuracy_score(Y_test, Y_predict)))
print("Precision: {:.2f}".format(precision_score(Y_test, Y_predict)))
print("Recall: {:.2f}".format(recall_score(Y_test, Y_predict)))
print("AUC: {:.2f}".format(roc_auc_score(Y_test, Y_predict)))

#importance of variables checken
var_columns = [c for c in X_train.columns if c not in ['act', 'next_act']] #afhankelijke variabelen invullen hier
df_var_importance = pd.DataFrame({"Variable": var_columns, "Importance": clf_xgb2.feature_importances_}).sort_values(by="Importance", ascending=False)
print(df_var_importance[:10])
