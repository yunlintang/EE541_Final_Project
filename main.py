import astropy
from ast import literal_eval
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MultiLabelBinarizer
import warnings
import src.data_clean as proc
import src.data_engineering as engineering
from collections import Counter

warnings.filterwarnings('ignore')




# data transformation

# OUT_DATA = "src/data/interim/"
# RAW_DATA = "src/data/raw/"
# proc.PreprocessMM(RAW_DATA, OUT_DATA, save=True)

vote_rating = ["vote_average", "vote_count"]
META_DATA = "data/interim/movies_metadata_clean.csv"
CREDITS_DATA = "data/interim/credits_clean.csv"
KEYWORDS_DATA = "data/interim/keywords_clean.csv"
COMBINE_DATA = "data/interim/combine.csv"
meta_data = pd.read_csv(META_DATA)
credits_data = pd.read_csv(CREDITS_DATA)
keywords_data = pd.read_csv(KEYWORDS_DATA)
combine_data = pd.read_csv(COMBINE_DATA)


# one-hot genres
combine_data = engineering.data_normalization(combine_data)
combine_data = engineering.clean_data(combine_data)
combine_data = engineering.multi_onehot(combine_data)

print(combine_data)
