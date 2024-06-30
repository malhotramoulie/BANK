from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_selection import SelectKBest, f_classif
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline
from collections import Counter
from sklearn.model_selection import train_test_split, RepeatedStratifiedKFold, cross_val_score
from sklearn.metrics import roc_auc_score, confusion_matrix, classification_report, roc_curve, auc
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

app = Flask(__name__)

# Load and preprocess data
file_path = r'C:\Users\MOULIE\OneDrive\Desktop\mumbai\archive (1)\BankChurners.csv'
data = pd.read_csv(file_path)

# Feature selection
features = data.loc[:, :'Amount']
target = data.loc[:, 'Class']
best_features = SelectKBest(score_func=f_classif, k='all')
fit = best_features.fit(features, target)
featureScores = pd.DataFrame(data=fit.scores_, index=list(features.columns), columns=['ANOVA Score'])
featureScores = featureScores.sort_values(ascending=False, by='ANOVA Score')

# Create datasets
df1 = data[['V3', 'V4', 'V7', 'V10', 'V11', 'V12', 'V14', 'V16', 'V17', 'Class']].copy(deep=True)
df2 = data.copy(deep=True)
df2.drop(columns=list(featureScores.index[20:]), inplace=True)

# Data balancing
def balance_data(df):
    over = SMOTE(sampling_strategy=0.5)
    under = RandomUnderSampler(sampling_strategy=0.1)
    features = df.iloc[:, :-1].values
    target = df.iloc[:, -1].values
    steps = [('under', under), ('over', over)]
    pipeline = Pipeline(steps=steps)
    features, target = pipeline.fit_resample(features, target)
    return features, target

f1, t1 = balance_data(df1)
f2, t2 = balance_data(df2)

# Train-test split
x_train1, x_test1, y_train1, y_test1 = train_test_split(f1, t1, test_size=0.20, random_state=2)
x_train2, x_test2, y_train2, y_test2 = train_test_split(f2, t2, test_size=0.20, random_state=2)

# Custom function to plot ROC curve
def custom_plot_roc_curve(classifier, x_test, y_test):
    y_score = classifier.predict_proba(x_test)[:,1]
    fpr, tpr, _ = roc_curve(y_test, y_score)
    roc_auc = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.show()

# Model training and evaluation function
def model(classifier, x_train, y_train, x_test, y_test):
    classifier.fit(x_train, y_train)
    prediction = classifier.predict(x_test)
    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
    cv_score = cross_val_score(classifier, x_train, y_train, cv=cv, scoring='roc_auc').mean()
    roc_auc = roc_auc_score(y_test, prediction)
    custom_plot_roc_curve(classifier, x_test, y_test)
    return cv_score, roc_auc

def model_evaluation(classifier, x_test, y_test):
    cm = confusion_matrix(y_test, classifier.predict(x_test))
    report = classification_report(y_test, classifier.predict(x_test), output_dict=True)
    return cm, report

# Define Flask routes
@app.route('/')
def home():
    return "Welcome to the API. Use /api/data or /train_model endpoints."

@app.route('/train_model', methods=['POST'])
def train_model():
    data = request.get_json()
    model_type = data.get('model_type')
    dataset = data.get('dataset')

    if dataset == 'df1':
        x_train, y_train, x_test, y_test = x_train1, y_train1, x_test1, y_test1
    else:
        x_train, y_train, x_test, y_test = x_train2, y_train2, x_test2, y_test2

    if model_type == 'logistic_regression':
        classifier = LogisticRegression(random_state=0, C=10, penalty='l2')
    elif model_type == 'svc':
        classifier = SVC(kernel='linear', C=0.1, probability=True)
    elif model_type == 'decision_tree':
        classifier = DecisionTreeClassifier(random_state=1000, max_depth=4, min_samples_leaf=1)
    elif model_type == 'random_forest':
        classifier = RandomForestClassifier(max_depth=4, random_state=0)
    elif model_type == 'knn':
        classifier = KNeighborsClassifier(leaf_size=1, n_neighbors=3, p=1)
    else:
        return jsonify({"error": "Invalid model type"}), 400

    cv_score, roc_auc = model(classifier, x_train, y_train, x_test, y_test)
    cm, report = model_evaluation(classifier, x_test, y_test)

    return jsonify({
        "cv_score": f"{cv_score:.2%}",
        "roc_auc": f"{roc_auc:.2%}",
        "confusion_matrix": cm.tolist(),
        "classification_report": report
    })

@app.route('/api/data', methods=['GET'])
def get_data():
    return jsonify({"message": "Hello, World!"})

if __name__ == '__main__':
    app.run(debug=True)
