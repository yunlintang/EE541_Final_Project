import astropy
from ast import literal_eval
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MultiLabelBinarizer
import warnings
import data_clean as proc
import feature_engineering as engineering
from collections import Counter
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import torch.utils.data as Data

warnings.filterwarnings('ignore')

class Net(nn.Module):
    def __init__(self, n_input, n_hidden=(1000, 500), n_output=1, drop_out=0.3):
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



# data transformation
# OUT_DATA = "src/data/interim/"
# RAW_DATA = "src/data/raw/"
# proc.PreprocessMM(RAW_DATA, OUT_DATA, save=True)


META_DATA = "data/interim/movies_metadata_clean.csv"
CREDITS_DATA = "data/interim/credits_clean.csv"
KEYWORDS_DATA = "data/interim/keywords_clean.csv"
COMBINE_DATA = "data/interim/combine.csv"
meta_data = pd.read_csv(META_DATA)
credits_data = pd.read_csv(CREDITS_DATA)
keywords_data = pd.read_csv(KEYWORDS_DATA)
combine_data = pd.read_csv(COMBINE_DATA)

COMBINE_DATA_FINAL = "data/interim/combine_clean_oh.csv"
combine_data_final = pd.read_csv(COMBINE_DATA_FINAL)
new_col = [col for col in combine_data_final.columns if col != 'score'] + ['score']
combine_data_final = combine_data_final[new_col]
combine_data_final = combine_data_final.drop(['id', 'keywords', 'overview', 'title'], axis=1)

# split to train and test dataset
train_data = combine_data_final.sample(frac=0.7, random_state=0, axis=0)
test_data = combine_data_final[~combine_data_final.index.isin(train_data.index)]

learning_rate = 0.0001
epochs = 40
batch_size = 500
num_batch = combine_data_final.shape[0] / batch_size
features_num = combine_data_final.shape[1]

train_x = train_data.values[:, :-1]
train_y = train_data.values[:, -1]
tensor_x = torch.Tensor(train_x)
tensor_y = torch.Tensor(train_y)
torch_train = Data.TensorDataset(tensor_x, tensor_y)
train_loader = Data.DataLoader(
    dataset=torch_train,
    batch_size=batch_size,
    shuffle=True
)


model = Net(n_input=features_num-1)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0.0001)
loss_func = nn.MSELoss()

for ep in range(epochs):
    for sample_feature, score in train_loader:
        y_hat = model(sample_feature)
        loss = loss_func(y_hat, score)

        optimizer.zero_grad()
        loss.backward()

        optimizer.step()
    print(f'Epoch:{ep}, loss: {loss.data}')

print("Finished Training")
