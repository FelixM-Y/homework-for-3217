import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Read the training set file, split the training set into features and labels and convert to Numpy arrays.
train_data = pd.read_csv('TrainingDataBinary.csv')
train_features = train_data.iloc[:, :-1].values
train_labels = train_data.iloc[:, -1].values

# Reading requires predicting the data file and passing. values to numpy array.
test_data = pd.read_csv('TestingDataBinary.csv', header=None)
test_features = test_data.values

# Feature normalization
scaler = StandardScaler()
train_features = scaler.fit_transform(train_features)
test_features = scaler.transform(test_features)

# The obtained split training set and validation set are divided into data set and test set, the data set is used for training, and the test set is used to test the accuracy.
# The ratio was 9 for the data set to 1 for the test set
X_train, X_test, y_train, y_test = train_test_split(train_features, train_labels, test_size=0.1, random_state=66)

# Defining the parameter grid(C is the penalty factor)
# Accuracy was improved by change C, gamma, and the kernel
param_grid = {
    'C': [0.1, 1, 10, 100],
    'kernel': ['linear', 'sigmoid'],
    'gamma': ['scale', 'auto']
}

# Creating SVM models
svm = SVC()

# Create a grid search object.CV is number of folds for cross-validation.
grid_search = GridSearchCV(svm, param_grid, cv=5)

# Performing a grid search
grid_search.fit(X_train, y_train)

# Get the best parameters and the best model
best_params = grid_search.best_params_
best_model = grid_search.best_estimator_

# Model accuracy was evaluated on the validation set
y_pred = best_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Use the trained model to make predictions
predictions = best_model.predict(test_features)
print(predictions)

#Concatenate the results with the data to output the model
test_data[128] = predictions
test_data.to_csv('TestingResultsBinary.csv', index=False, header=False)
