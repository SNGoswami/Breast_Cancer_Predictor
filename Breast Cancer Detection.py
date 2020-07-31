# import libraries
import os
import pandas as pd  # for data manipulation or analysis
import numpy as np  # for numeric calculation
import matplotlib.pyplot as plt  # for data visualization
import seaborn as sns  # for data visualization
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.utils import Bunch
from xgboost import XGBClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import cross_val_score
import pickle

# Load breast cancer dataset
from sklearn.datasets import load_breast_cancer

cancer_dataset = load_breast_cancer()

type(cancer_dataset)

# keys in dataset
cancer_dataset.keys()

# features of each cells in numeric format
D = cancer_dataset['data']

# malignant or benign value
F = cancer_dataset['target']

# target value name malignant or benign tumor
G = cancer_dataset['target_names']

# description of data
print(cancer_dataset['DESCR'])

#  name of features
print(cancer_dataset['feature_names'])

# location/path of data file
print(cancer_dataset['filename'])

# create datafrmae
cancer_df = pd.DataFrame(np.c_[cancer_dataset['data'], cancer_dataset['target']],
                         columns=np.append(cancer_dataset['feature_names'], ['target']))

# Head of cancer DataFrame
cancer_df.head(6)

# Tail of cancer DataFrame
cancer_df.tail(6)

# Information of cancer Dataframe
cancer_df.info()

# Numerical distribution of data
cancer_df.describe()

# Pairplot of cancer dataframe
sns.pairplot(cancer_df, hue='target')

# pair plot of sample feature
sns.pairplot(cancer_df, hue='target',
             vars=['mean radius', 'mean texture', 'mean perimeter', 'mean area', 'mean smoothness'])

# Count the target class
sns.countplot(cancer_df['target'])

# counter plot of feature mean radius
plt.figure(figsize=(20, 8))
sns.countplot(cancer_df['mean radius'])

# heatmap of DataFrame
plt.figure(figsize=(16, 9))
sns.heatmap(cancer_df)

# Heatmap of Correlation matrix of breast cancer DataFrame
plt.figure(figsize=(20, 20))
sns.heatmap(cancer_df.corr(), annot=True, cmap='coolwarm', linewidths=2)

# create second DataFrame by droping target
cancer_df2 = cancer_df.drop(['target'], axis=1)
print("The shape of 'cancer_df2' is : ", cancer_df2.shape)

# visualize correlation barplot
plt.figure(figsize=(16, 5))
ax = sns.barplot(cancer_df2.corrwith(cancer_df.target).index, cancer_df2.corrwith(cancer_df.target))
ax.tick_params(labelrotation=90)

# input variable
X = cancer_df.drop(['target'], axis=1)
X.head(6)

# output variable
y = cancer_df['target']
y.head(6)

# split dataset into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=5)

# Feature scaling
sc = StandardScaler()
X_train_sc = sc.fit_transform(X_train)
X_test_sc = sc.transform(X_test)

# XGBoost Classifier

xgb_classifier = XGBClassifier()
xgb_classifier.fit(X_train, y_train)
y_pred_xgb = xgb_classifier.predict(X_test)
accuracy_score(y_test, y_pred_xgb)

# Train with Standard scaled Data
xgb_classifier2 = XGBClassifier()
xgb_classifier2.fit(X_train_sc, y_train)
y_pred_xgb_sc = xgb_classifier2.predict(X_test_sc)
accuracy_score(y_test, y_pred_xgb_sc)

# XGBoost classifier most required parameters
params = {
 "learning_rate": [0.05, 0.10, 0.15, 0.20, 0.25, 0.30],
 "max_depth": [3, 4, 5, 6, 8, 10, 12, 15],
 "min_child_weight": [1, 3, 5, 7],
 "gamma": [0.0, 0.1, 0.2, 0.3, 0.4],
 "colsample_bytree": [0.3, 0.4, 0.5, 0.7]
}

# Randomized Search
random_search = RandomizedSearchCV(xgb_classifier, param_distributions=params, scoring='roc_auc', n_jobs=-1, verbose=3)
random_search.fit(X_train, y_train)

T = random_search.best_params_

V = random_search.best_estimator_

# training XGBoost classifier with best parameters
xgb_classifier_pt = XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,
                                  colsample_bynode=1, colsample_bytree=0.4, gamma=0.2,
                                  learning_rate=0.1, max_delta_step=0, max_depth=15,
                                  min_child_weight=1, missing=None, n_estimators=100, n_jobs=1,
                                  nthread=None, objective='binary:logistic', random_state=0,
                                  reg_alpha=0, reg_lambda=1, scale_pos_weight=1, seed=None,
                                  silent=None, subsample=1, verbosity=1)

xgb_classifier_pt.fit(X_train, y_train)
y_pred_xgb_pt = xgb_classifier_pt.predict(X_test)

accuracy_score(y_test, y_pred_xgb_pt)

cm = confusion_matrix(y_test, y_pred_xgb_pt)
plt.title('Heatmap of Confusion Matrix', fontsize=15)
sns.heatmap(cm, annot=True)
plt.show()

print(classification_report(y_test, y_pred_xgb_pt))

# Cross validation
cross_validation = cross_val_score(estimator=xgb_classifier_pt, X=X_train_sc, y=y_train, cv=10)
print("Cross validation of XGBoost model = ", cross_validation)
print("Cross validation of XGBoost model (in mean) = ", cross_validation.mean())


cross_validation = cross_val_score(estimator=xgb_classifier_pt, X=X_train_sc, y=y_train, cv=10)
print("Cross validation accuracy of XGBoost model = ", cross_validation)
print("\nCross validation mean accuracy of XGBoost model = ", cross_validation.mean())

# Pickle

# save model
pickle.dump(xgb_classifier_pt, open('model.pkl', 'wb'))

# load model
breast_cancer_detector_model = pickle.load(open('model.pkl', 'rb'))

# predict the output
y_pred = breast_cancer_detector_model.predict(X_test)

# confusion matrix
print('Confusion matrix of XGBoost model: \n', confusion_matrix(y_test, y_pred), '\n')

# show the accuracy
print('Accuracy of XGBoost model = ', accuracy_score(y_test, y_pred))
