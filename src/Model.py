import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif
from torch.optim.lr_scheduler import StepLR
import torchvision
import torchvision.transforms as transforms
import torch.utils.data as Data
from torch.utils.tensorboard import SummaryWriter
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error as MSE

from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression

import matplotlib.pyplot as plt


def svr_task(train_x, train_y, test_x, test_y):
    pca_full = PCA(n_components=1000)
    pca_full.fit(train_x)
    x_train_select = pca_full.transform(train_x)
    x_test_select = pca_full.transform(test_x)

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

    # mission 1
    model.fit(x_train_select, train_y)
    y_pre1 = model.predict(x_test_select)
    rmse1 = MSE(test_y, y_pre1) ** 0.5
    r1 = model.score(x_train_select, test_y)
    # print(model.best_params_)
    print(f"SVR for mission1 —— RMSE:{rmse1} R-squared:{r1}")


def linear_regression_task(train_x, train_y, test_x, test_y):
    model_linear = LinearRegression()
    # define feature selection 43
    # fs = SelectKBest(score_func=f_classif, k=2000)
    # x_train_select = fs.fit_transform(train_x, train_y)
    # x_test_select = fs.transform(test_x)

    pca_full = PCA(n_components=1500)
    pca_full.fit(train_x)
    x_train_select = pca_full.transform(train_x)
    x_test_select = pca_full.transform(test_x)

    model_linear.fit(x_train_select, train_y)
    y_linear_hat1 = model_linear.predict(x_test_select)
    rmse_linear1 = MSE(test_y, y_linear_hat1)**0.5
    r2_linear1 = model_linear.score(x_test_select, test_y)
    print(f"linear regression —— RMSE:{rmse_linear1}, R-squared:{r2_linear1}")


class Net(nn.Module):
    def __init__(self, n_input, n_hidden=(500, 500), n_output=1, drop_out=0.3):
        super(Net, self).__init__()
        self.hidden1 = nn.Linear(n_input, n_hidden[0])
        self.hidden2 = nn.Linear(n_hidden[0], n_hidden[1])
        self.output = nn.Linear(n_hidden[1], n_output)
        self.dropout = nn.Dropout(p=drop_out)

    def forward(self, input_data):
        out = F.relu(self.hidden1(input_data))
        out = self.dropout(out)
        out = F.relu(self.hidden2(out))
        out = self.dropout(out)
        out = self.output(out)
        return out


def train(train_x, train_y, test_x, test_y):
    pca_full = PCA(n_components=1500)
    pca_full.fit(train_x)
    x_train_select = pca_full.transform(train_x)
    x_test_select = pca_full.transform(test_x)

    # define feature selection 43
    # fs = SelectKBest(score_func=f_classif, k=1500)
    # x_train_select = fs.fit_transform(train_x, train_y)
    # x_test_select = fs.transform(test_x)

    learning_rate = 0.0005
    epochs = 40
    batch_size = 100
    features_num = x_train_select.shape[1]
    writer = SummaryWriter("../NN_model")

    # tensor_x = torch.Tensor(train_x.to_numpy())
    tensor_x = torch.Tensor(x_train_select)
    tensor_y = torch.Tensor(train_y.to_numpy())
    torch_train = Data.TensorDataset(tensor_x, tensor_y)
    train_loader = Data.DataLoader(
        dataset=torch_train,
        batch_size=batch_size,
        shuffle=True
    )

    # tensor_test_x = torch.Tensor(test_x.to_numpy())
    tensor_test_x = torch.Tensor(x_test_select)
    tensor_test_y = torch.Tensor(test_y.to_numpy())
    # torch_test = Data.TensorDataset(tensor_test_x, tensor_test_y)
    # test_loader = Data.DataLoader(
    #     dataset=torch_test,
    #     batch_size=batch_size,
    #     shuffle=True
    # )

    model = Net(n_input=features_num)
    dummy_input = torch.rand(20, features_num)
    writer.add_graph(model, dummy_input)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0.0001)
    loss_func = nn.MSELoss()
    scheduler = StepLR(optimizer, step_size=20, gamma=0.5)

    for ep in range(epochs):
        loss_ep = 0
        N = 0
        for sample_feature, score in train_loader:
            y_hat = model(sample_feature)
            loss = loss_func(y_hat, score)
            loss_ep += loss
            N += 1
            optimizer.zero_grad()
            loss.backward()

            optimizer.step()
        scheduler.step()
        writer.add_scalar(tag="Train_loss", scalar_value=loss_ep/N, global_step=ep)
        for name, param in model.named_parameters():
            writer.add_histogram(tag=name + '_data', values=param.data, global_step=ep)
        print(f'Epoch:{ep}, loss: {loss_ep / N}, step:{scheduler.get_lr()}')
    for name, param in model.named_parameters():
        writer.add_histogram(tag='Final_' + name + '_data', values=param.data)

    # torch.save(model.state_dict(), "../Model_param")
    print("Finished Training")

    with torch.no_grad():
        model.eval()
        outputs = model(tensor_test_x)
        r2 = r2_score(tensor_test_y, outputs)
        rmse = MSE(tensor_test_y, outputs) ** 0.5
        print(f"R-2:{r2}, RMSE:{rmse}")
        pre = outputs.tolist()
        true = tensor_test_y.tolist()
        plt.title("Movie ratings vs Predicted ratings")
        plt.xlabel("movie ratings")
        plt.ylabel("predicted ratings")
        plt.plot(true, pre, 'o', color="b")

    plt.axline((0, 0), slope=1, color="r", ls="--", lw=2.5)
    plt.show()
