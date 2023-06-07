import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV

# The data is read and printed just to make sure the range is correct.
TrainingDataMulti = pd.read_csv('TrainingDataMulti.csv', header=None)
#print('traindata.shape', TrainingDataMulti.shape)
TestingDataMulti = pd.read_csv('TestingDataMulti.csv', header=None)
#print('testdata.shape', TestingDataMulti.shape)
#print(TestingDataMulti)
TestingDataMulti_X = TestingDataMulti.iloc[0:127]

TrainingDataMulti_X = TrainingDataMulti.iloc[:, :-1]
TrainingDataMulti_y = TrainingDataMulti.iloc[:, -1]
#print('TrainingDataMulti_X.shape', TrainingDataMulti_X.shape)
#print('TrainingDataMulti_y.shape', TrainingDataMulti_y.shape)

# Feature normalization
scaler = StandardScaler()
TrainingDataMulti_X_Scaled = scaler.fit_transform(TrainingDataMulti_X)
TestingDataMulti_X_Scaled = scaler.transform(TestingDataMulti_X)

# Split the training and test sets
X_train, X_test, y_train, y_test = train_test_split(TrainingDataMulti_X_Scaled, TrainingDataMulti_y, test_size=0.2, random_state=42)

# Defining the parameter grid
param_grid = {
    'n_estimators': [50, 100, 200],  # Different number of decision trees
    'max_depth': [None, 5, 10],  # Different maximum depths
    'min_samples_split': [2, 5, 10],  # Different minimum sample split scores
    'max_features': ['sqrt', 'log2']  # Different options for maximum number of features
}

# Create a random forest classifier
rf_classifier = RandomForestClassifier()

# Create a grid search object
grid_search = GridSearchCV(rf_classifier, param_grid, cv=3)

# Performing a grid search
grid_search.fit(X_train, y_train)

# Get the best parameters and best model
best_params = grid_search.best_params_
best_model = grid_search.best_estimator_

# Evaluate model performance using cross-validation
cv_scores = cross_val_score(best_model, X_train, y_train, cv=5)

# Output cross-validation score
print("Cross-validation scores:", cv_scores)

# Make predictions on the test set
y_pred = best_model.predict(X_test)

# calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Make predictions on the test set
pre = best_model.predict(TestingDataMulti_X_Scaled)

#output
print(pre)

# The first 128 column features and prediction results are merged and output
result = pd.concat([TestingDataMulti_X, pd.DataFrame(pre)], axis=1)
result.to_csv('TestingResultsMulti.csv.csv', index=False, header=False)