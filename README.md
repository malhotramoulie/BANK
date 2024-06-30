# Risk Management API for Bank of Baroda

## Overview

This project aims to develop a comprehensive risk management API for the Bank of Baroda. The API includes several features such as credit card fraud detection, loan fraud detection, a 24/7 helping chatbot named BOB, user activity tracking, and a personalized recommendation system for investments. The main codebase is developed using Jupyter Notebooks and utilizes various Python libraries for data processing, balancing, and modeling.

## Features

- **Credit Card Fraud Detection**: Identifies fraudulent credit card transactions.
- **Loan Fraud Detection**: Detects potential fraudulent loan applications.
- **24/7 Helping Chatbot (BOB)**: Provides round-the-clock assistance to users.
- **User Tracking**: Monitors user activity within the application.
- **Personalized Recommendation System**: Suggests investment opportunities and plans based on user preferences and goals.

## Libraries Used

### Data Handling and Visualization
- `pandas`: Data manipulation and analysis.
- `numpy`: Numerical operations.
- `matplotlib`: Plotting and visualization.
- `seaborn`: Statistical data visualization.

### Data Balancing
- `imblearn`: Handling imbalanced datasets.
- `SMOTE`: Synthetic Minority Over-sampling Technique.
- `RandomUnderSampler`: Random under-sampling of the majority class.
- `Pipeline`: Utility for creating a composite estimator.

### Modeling and Evaluation
- `sklearn`: Machine learning library.
- `train_test_split`: Splitting datasets into training and testing sets.
- `confusion_matrix`: Evaluating the accuracy of classification.
- `roc_auc_score`: Calculating the area under the ROC curve.
- `RocCurveDisplay`: Visualizing the ROC curve.
- `cross_val_score`: Cross-validation of the model.
- `GridSearchCV`: Hyperparameter tuning.
- `classification_report`: Summarizing the performance of the model.
- `RepeatedStratifiedKFold`: Cross-validator with stratified folds.
- `precision_recall_curve`: Plotting precision-recall pairs.
- `roc_curve`, `auc`: Calculating and plotting ROC curves.

## Dataset Information

The project uses various datasets for training and testing the models. The datasets are preprocessed and balanced to ensure accurate and reliable results.

## Setup and Installation

### Clone the Repository:
`bash
git clone https://github.com/your-username/risk-management-api.git
cd risk-management-api`

## Install Dependencies

## Run the Jupyter Notebook
'bash
jupyter notebook
Open the main_notebook.ipynb file to start working with the code.

## Usage
- Credit Card Fraud Detection
- Load the dataset and preprocess it.
- Use SMOTE and RandomUnderSampler for data balancing.
- Train the model using the balanced dataset.
- Evaluate the model using ROC, precision-recall curves, and other metrics.
## Loan Fraud Detection
- Similar steps as credit card fraud detection, with a different dataset.
- 24/7 Helping Chatbot (BOB)
- Implement the chatbot using a suitable framework (e.g., Rasa, Dialogflow).
- Integrate the chatbot with the API.
## User Tracking
- Monitor user activity and store the data for analysis.
## Personalized Recommendation System
- Use user data to provide personalized investment recommendations.
- Implement algorithms to suggest investment plans based on user goals.
## Available Scripts
### In the project directory, you can run:

 `python start `\
--Runs the app in development mode.\
 `python test `\
--Launches the test runner.\
 `python run build `\
--Builds the app for production.\
 `python run eject `\
--Ejects the project from its current configuration.\
## Learn More
## Documentation
[Create React App documentation](https://facebook.github.io/create-react-app/docs/getting-started).
[React documentation](https://reactjs.org/).
### Code Splitting

This section has moved here: [https://facebook.github.io/create-react-app/docs/code-splitting](https://facebook.github.io/create-react-app/docs/code-splitting)
### Making a Progressive Web App

This section has moved here: [https://facebook.github.io/create-react-app/docs/making-a-progressive-web-app](https://facebook.github.io/create-react-app/docs/making-a-progressive-web-app)

### Advanced Configuration

This section has moved here: [https://facebook.github.io/create-react-app/docs/advanced-configuration](https://facebook.github.io/create-react-app/docs/advanced-configuration)

### Deployment

This section has moved here: [https://facebook.github.io/create-react-app/docs/deployment](https://facebook.github.io/create-react-app/docs/deployment)


