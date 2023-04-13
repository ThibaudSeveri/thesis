import os
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier

# Load the data from the Excel file
path_to_file = "/Users/thibaudseveri/PycharmProjects/pythonProject/importdata/HR_log_data.xlsx"
data1 = pd.read_excel(path_to_file)
path_to_file2 = "/Users/thibaudseveri/Documents/Thesis/pred_next_act.xlsx"
data2 = pd.read_excel(path_to_file2)

#nieuwe dataset new_dataset
new_dataset = pd.DataFrame()
new_dataset2 = pd.DataFrame()
new_dataset['id'] = data1['id']
new_dataset2['id'] = data1['id']

# Count the number of occurrences of each ID
#id_counts = new_dataset['id'].value_counts()

# Count the number of IDs that appear a certain number of times
#count_dict = {}
#for count in range(1, id_counts.max() + 1):
    #count_dict[count] = (id_counts == count).sum()
#print(count_dict)

#maak 20 kopieën aan van de dataset data1
copy1 = data1.copy()
copy2 = data1.copy()
copy3 = data1.copy()
copy4 = data1.copy()
copy5 = data1.copy()
copy6 = data1.copy()
copy7 = data1.copy()
copy8 = data1.copy()
copy9 = data1.copy()
copy10 = data1.copy()
copy11 = data1.copy()
copy12 = data1.copy()
copy13 = data1.copy()
copy14 = data1.copy()
copy15 = data1.copy()
copy16 = data1.copy()
copy17 = data1.copy()
copy18 = data1.copy()
copy19 = data1.copy()
copy20 = data1.copy()

#V01-V11 first en last als aparte kolommen toevoegen in new_dataset
#V01
copy1.drop_duplicates(subset='id', keep= 'first', inplace=True)
new_dataset['V01_1'] = copy1['V01']
copy2.drop_duplicates(subset='id', keep='last', inplace=True)
new_dataset2['V01_2'] = copy2['V01']

#V02
copy3.drop_duplicates(subset='id', keep= 'first', inplace=True)
new_dataset['V02_1'] = copy3['V02']
copy4.drop_duplicates(subset='id', keep='last', inplace=True)
new_dataset2['V02_2'] = copy4['V02']

#V03
copy5.drop_duplicates(subset='id', keep= 'first', inplace=True)
new_dataset['V03_1'] = copy5['V03']
copy6.drop_duplicates(subset='id', keep='last', inplace=True)
new_dataset2['V03_2'] = copy6['V03']

#V04 (uniek per id, dus first of last maakt niet uit)
copy7.drop_duplicates(subset='id', keep='first', inplace=True)
new_dataset['V04'] = copy7['V04']

#V05 (uniek per id, dus first of last maakt niet uit)
copy8.drop_duplicates(subset='id', keep='first', inplace=True)
new_dataset['V05'] = copy8['V05']

#V06
copy9.drop_duplicates(subset='id', keep= 'first', inplace=True)
new_dataset['V06_1'] = copy9['V06']
copy10.drop_duplicates(subset='id', keep='last', inplace=True)
new_dataset2['V06_2'] = copy10['V06']

#V07
copy11.drop_duplicates(subset='id', keep= 'first', inplace=True)
new_dataset['V07_1'] = copy11['V07']
copy12.drop_duplicates(subset='id', keep='last', inplace=True)
new_dataset2['V07_2'] = copy12['V07']

#V08
copy13.drop_duplicates(subset='id', keep= 'first', inplace=True)
new_dataset['V08_1'] = copy13['V08']
copy14.drop_duplicates(subset='id', keep='last', inplace=True)
new_dataset2['V08_2'] = copy14['V08']

#V09
copy15.drop_duplicates(subset='id', keep= 'first', inplace=True)
new_dataset['V09_1'] = copy15['V09']
copy16.drop_duplicates(subset='id', keep='last', inplace=True)
new_dataset2['V09_2'] = copy16['V09']

#V10
copy17.drop_duplicates(subset='id', keep= 'first', inplace=True)
new_dataset['V10_1'] = copy17['V10']
copy18.drop_duplicates(subset='id', keep='last', inplace=True)
new_dataset2['V10_2'] = copy18['V10']

#V11
copy19.drop_duplicates(subset='id', keep= 'first', inplace=True)
new_dataset['V11_1'] = copy19['V11']
copy20.drop_duplicates(subset='id', keep='last', inplace=True)
new_dataset2['V11_2'] = copy20['V11']

#maakt new_dataset uniek op id
new_dataset.drop_duplicates(subset='id', keep='first', inplace=True)
new_dataset2.drop_duplicates(subset='id', keep='last', inplace=True)


#merge new_dataset en new_dataset2 together on 'id' in new_dataset
# specify columns to join on
join_columns = ['id']

# specify columns to include from new_dataset
new_dataset_columns = ['V01_1', 'V02_1', 'V03_1', 'V04', 'V05', 'V06_1', 'V07_1', 'V08_1', 'V09_1', 'V10_1', 'V11_1']

# merge the datasets on the specified columns
merged_dataset = pd.merge(new_dataset[join_columns + new_dataset_columns], new_dataset2, on=join_columns)

#voeg oorspronkelijke variabelen van data2 erin door te mergen met merged_dataset
# specify columns to join on
join_columns = ['id']

# specify columns to include from merged_dataset
new_dataset_columns = ['V01_1', 'V02_1', 'V03_1', 'V04', 'V05', 'V06_1', 'V07_1', 'V08_1', 'V09_1', 'V10_1', 'V11_1', 'V01_2', 'V02_2', 'V03_2', 'V06_2', 'V07_2', 'V08_2', 'V09_2', 'V10_2', 'V11_2']

# merge the datasets on the specified columns
merged_dataset2 = pd.merge(merged_dataset, data2[['id', 'Difference_time_total', 'Job_variation_count', 'Time_job_ratio', 'prev_act', 'act']], on='id', how='inner')

#print nieuwe dataframe new_dataset in excel bestand
#with pd.ExcelWriter('new.xlsx') as writer:
    #merged_dataset2.to_excel(writer, sheet_name='Sheet1', index=False)
#os.system('open ' + 'new.xlsx')

#random forest beginnen

# Define the target column Y
Y = merged_dataset2['act']
Y = Y.to_frame()

# Y-dataset omzetting naar 1 voor leave en 0 voor al de andere act
Y.loc[Y['act'] == 'leave', 'act'] = 1
Y.loc[Y['act'] != 1, 'act'] = 0

#define the feature columns merged_dataset2 without target column 'act'
merged_dataset2 = merged_dataset2.drop(['act'], axis=1).copy()

# Hot encoding toepassen, dus kolommen bijvoegen van elke mogelijke categorische om aan te geven of die 0 of 1 hebben bij die value
X_encoded = pd.get_dummies(merged_dataset2, columns=['prev_act', 'V01_1', 'V02_1', 'V03_1', 'V04', 'V05', 'V06_1', 'V07_1', 'V01_2', 'V02_2', 'V03_2', 'V06_2', 'V07_2'])

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X_encoded, Y, random_state=42, stratify=Y)

# zet waarden van 'act' om van object naar integers, is nodig voor RF te fitten
Y['act'] = pd.to_numeric(Y['act'])
y_test['act'] = pd.to_numeric(y_test['act'])
y_train['act'] = pd.to_numeric(y_train['act'])

#undersampling toepassen zodat we 80/20% hebben van non-leave en leave voor betere voorspelling
# oversample using SMOTE
sm = SMOTE(sampling_strategy=0.25, random_state=42)
X_train, y_train = sm.fit_resample(X_train, y_train)

# Define the parameter grid
param_grid = {
    'n_estimators': [160], #150 (best) maar willen vergelijken met tabular data
    'max_depth': [6], #9 (best)
    'min_samples_split': [2],
    'class_weight': ['balanced']
}

# Create a random forest classifier object
rfc = RandomForestClassifier(random_state=42)

# Create the GridSearchCV object
grid_search = GridSearchCV(estimator=rfc, param_grid=param_grid, cv=5, n_jobs=-1, scoring='roc_auc')

# Fit the GridSearchCV object to the data
grid_search.fit(X_train, y_train)

# Print the best parameters and best score
print("Best parameters: ", grid_search.best_params_)
print("Best score: ", grid_search.best_score_)

# Use the best parameters to train and evaluate the model
rfc_best = RandomForestClassifier(n_estimators=grid_search.best_params_['n_estimators'],
                                   max_depth=grid_search.best_params_['max_depth'],
                                   min_samples_split=grid_search.best_params_['min_samples_split'],
                                   class_weight=grid_search.best_params_['class_weight'],
                                   random_state=42)

rfc_best.fit(X_train, y_train)

y_train_pred = rfc_best.predict_proba(X_train)[:, 1]
y_test_pred = rfc_best.predict_proba(X_test)[:, 1]

print("Train AUC: ", roc_auc_score(y_train, y_train_pred))
print("Test AUC: ", roc_auc_score(y_test, y_test_pred))

# Converteer y_test_pred naar binaire voorspellingen
y_test_pred_binary = [1 if x > 0.5 else 0 for x in y_test_pred]

# Bereken nauwkeurigheid, precisie en recall
print("Accuracy: {:.2f}".format(accuracy_score(y_test, y_test_pred_binary)))
print("Precision: {:.2f}".format(precision_score(y_test, y_test_pred_binary)))
print("Recall: {:.2f}".format(recall_score(y_test, y_test_pred_binary)))

# Save the updated dataset to a new Excel file
merged_dataset2.to_excel('predactadvancedRF.xlsx', index=False)
