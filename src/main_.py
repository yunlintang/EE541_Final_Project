import pandas as pd
import warnings
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error as MSE

from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression

import Model

warnings.filterwarnings('ignore')


def svr_task(train_x, train_y, test_x, test_y):
    # pca_full = PCA(n_components=35)
    # pca_full.fit(train.drop(['G1', 'G2', 'G3'], axis=1))
    # x_train_select = pca_full.transform(train.drop(['G1', 'G2', 'G3'], axis=1))
    # x_test_select = pca_full.transform(test.drop(['G1', 'G2', 'G3'], axis=1))
    #
    # pca_full.fit(train.drop(['G3'], axis=1))
    # x_train_select_3 = pca_full.transform(train.drop(['G3'], axis=1))
    # x_test_select_3 = pca_full.transform(test.drop(['G3'], axis=1))

    # define feature selection 43
    # fs = SelectKBest(score_func=f_classif, k=1500)
    # x_train_select = fs.fit_transform(train_x, train_y)
    # x_test_select = fs.transform(test_x)

    c_list = []
    for i in range(1, 2):
        c_list.append(i * 10)

    gamma_list = []
    for j in range(1, 2):
        gamma_list.append(j * 0.001)

    # cross validation
    param_grid = {'gamma': gamma_list, 'C': c_list, 'kernel': ['rbf']}
    # model = GridSearchCV(SVR(), param_grid, cv=10, scoring='r2')

    model = SVR(kernel='rbf', C=1000, gamma=0.1)
    model.fit(train_x, train_y)

    # mission 1
    model.fit(train_x, train_y)
    y_pre1 = model.predict(test_x)
    rmse1 = MSE(test_y, y_pre1)**0.5
    r1 = model.score(train_x, test_y)
    # print(model.best_params_)
    print(f"SVR for mission1 —— RMSE:{rmse1} R-squared:{r1}")

# writer = SummaryWriter("../NNmodel")
df = pd.read_csv('../data/final/data.csv')
df['runtime'] = df['runtime'].fillna(0)

X = df.drop(columns=['score'])
y = df['score']
X_train, X_test, y_train, y_test = train_test_split(X,y,train_size=0.7)

# Model.train(X_train, y_train, X_test, y_test)

# Model.linear_regression_task(X_train, y_train, X_test, y_test)
svr_task(X_train, y_train, X_test, y_test)

