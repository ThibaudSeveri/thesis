import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import balanced_accuracy_score, precision_score, recall_score
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.impute import SimpleImputer
import category_encoders as ce
from category_encoders import woe

# data inladen
path_to_file = "/Users/thibaudseveri/PycharmProjects/pythonProject/importdata/finalprocessorienteddataset.xlsx"
merged_df = pd.read_excel(path_to_file)

#Random Forest op merged_df

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

#missing_cols = X_train.columns[X_train.isnull().any()]
# Print columns with missing values
#for col in missing_cols:
    #print(f"{col}: {X_train[col].isnull().sum()} missing values")

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

# Convert object dtype columns to numeric dtype columns (nodig voor het model)
X_train[X_train.select_dtypes(include='object').columns] = X_train.select_dtypes(include='object').astype('float64')
X_test[X_test.select_dtypes(include='object').columns] = X_test.select_dtypes(include='object').astype('float64')
y_train = y_train.astype('float64')
y_test = y_test.astype('float64')

#oversampling toepassen zodat we 80/20% hebben van non-leave en leave voor betere voorspelling using SMOTE, want imblanced dataset
sm = SMOTE(sampling_strategy=0.25, random_state=42)
X_train, y_train = sm.fit_resample(X_train, y_train)

print(X_train)
print(X_test)
print(y_train)
print(y_test)

# Define the parameter grid
param_grid = {
    'n_estimators': [160, 180, 190], #zelfde als xgboost om te kunnen vergelijken (180)
    'max_depth': [12, 14, 16], # zelfde als xgboost om te kunnen vergelijken (14)
    'min_samples_split': [3, 4, 5], #4
    'max_features': [0.3, 0.4, 0.5],#0.4
}

# Create a random forest classifier object
rfc = RandomForestClassifier(random_state=42)

# Create the GridSearchCV object
grid_search = GridSearchCV(estimator=rfc, param_grid=param_grid, cv=10, n_jobs=-1, scoring='roc_auc')

#warning oplossen dat y_train 1D moet zijn
y_train = np.ravel(y_train)

# Fit the GridSearchCV object to the data
grid_search.fit(X_train, y_train)

# Print the best parameters and best score
print("Best parameters: ", grid_search.best_params_)
print("Best score: ", grid_search.best_score_)

# Use the best parameters to train and evaluate the model
rfc_best = RandomForestClassifier(n_estimators=160,
                                   max_depth=grid_search.best_params_['max_depth'],
                                   min_samples_split=grid_search.best_params_['min_samples_split'],
                                   class_weight=grid_search.best_params_['class_weight'],
                                   max_features=grid_search.best_params_['max_features'],
                                   random_state=42)
#fit beide modellen
rfc.fit(X_train, y_train)
rfc_best.fit(X_train, y_train)

# perform 10-fold cross-validation and print the results for different metrics
kfold = KFold(n_splits=10, shuffle=True, random_state=42)
accuracy_scores = cross_val_score(rfc_best, X_train, y_train, cv=kfold)
print("Accuracy: %0.2f (+/- %0.2f)" % (accuracy_scores.mean(), accuracy_scores.std() * 2))

auc_scores = cross_val_score(rfc_best, X_train, y_train, cv=kfold, scoring='roc_auc')
print("AUC: %0.2f (+/- %0.2f)" % (auc_scores.mean(), auc_scores.std() * 2))

precision_scores = cross_val_score(rfc_best, X_train, y_train, cv=kfold, scoring='precision')
print("Precision: %0.2f (+/- %0.2f)" % (precision_scores.mean(), precision_scores.std() * 2))

recall_scores = cross_val_score(rfc_best, X_train, y_train, cv=kfold, scoring='recall')
print("Recall: %0.2f (+/- %0.2f)" % (recall_scores.mean(), recall_scores.std() * 2))

y_train_pred = rfc_best.predict(X_train)
y_test_pred = rfc_best.predict(X_test)
y_test_pred2 = rfc.predict(X_test)

print("Train AUC: ", roc_auc_score(y_train, y_train_pred))
print("Test AUC: ", roc_auc_score(y_test, y_test_pred))

# Bereken nauwkeurigheid, precisie en recall voor optimized model
print("Accuracy: {:.2f}".format(balanced_accuracy_score(y_test, y_test_pred)))
print("Precision: {:.2f}".format(precision_score(y_test, y_test_pred)))
print("Recall: {:.2f}".format(recall_score(y_test, y_test_pred)))

#voor non-optimized model
print("AUC: ", roc_auc_score(y_test, y_test_pred2))
print("Accuracy2: {:.2f}".format(balanced_accuracy_score(y_test, y_test_pred2)))
print("Precision2: {:.2f}".format(precision_score(y_test, y_test_pred2)))
print("Recall2: {:.2f}".format(recall_score(y_test, y_test_pred2)))
