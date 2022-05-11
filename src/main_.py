import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import warnings

from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error as MSE

from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression

import Model

warnings.filterwarnings('ignore')

df = pd.read_csv('../data/final/data.csv')
df['runtime'] = df['runtime'].fillna(0)

X = df.drop(columns=['score'])
y = df['score']
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7)

Model.train(X_train, y_train, X_test, y_test)

# Model.linear_regression_task(X_train, y_train, X_test, y_test)
# Model.svr_task(X_train, y_train, X_test, y_test)

# pca_full = PCA()
# pca_full.fit(X_train)
#
# plt.figure()
# plt.title('PCA cumulated var.')
# plt.plot(np.cumsum(pca_full.explained_variance_ratio_))
# plt.show()
