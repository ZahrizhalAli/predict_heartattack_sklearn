import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import numpy as np

# Read Dataset
df = pd.read_csv('./data/heart.csv')

# Split the label and the features
X = df.drop('target', axis=1)
y = df['target']

# Train Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=101)

# Performing Standardization
scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

#
base_model = LogisticRegression(solver='saga', multi_class='ovr', max_iter=5000)


penalty = ['l1', 'l2','elasticnet']
l1_ratio = np.linspace(0, 1, 20)
C = np.logspace(0, 10, 20) # lambda

param_grid = {'penalty': penalty,'l1_ratio':l1_ratio,'C':C }
grid_search = GridSearchCV(base_model, param_grid=param_grid)

grid_search.fit(X_train, y_train)
print(grid_search.best_params_)
