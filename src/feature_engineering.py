from ast import literal_eval
import numpy as np
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer
import warnings
from collections import Counter

def clean_credits(x):
    '''
    :param x:
    :return: convert all strings to lower case and strip names of spaces
    '''
    global SUM
    temp = []
    x_new = list(filter(None, x))
    SUM += 1
    # print(SUM)
    if 0 < len(x_new) <= 1:
        for i in x_new:
            temp.append(str.lower(i.replace(" ", "")))
    elif len(x_new) > 1:
        for i in range(1):
            temp.append(str.lower(x_new[i].replace(" ", "")))
    return temp


def weighted_rating(x, m, C):
    '''

    :param x: data
    :param m: the minimum votes required to be listed in the chart
    :param C: the mean vote across the whole report
    :return: weighted rating
    '''
    v = x['vote_count']
    R = x['vote_average']
    return (v / (v + m) * R) + (m / (m + v) * C)


def clean_data(data):
    '''

    :param data: data
    calculate the weighted score using imdb algorithm
    convert data to list,
    :return:converted data
    '''

    vote_average = data[data['vote_average'].notnull()]['vote_average'].astype('int')
    C = data['vote_average'].mean()
    m = data['vote_average'].quantile(0.90)

    data = data[data['vote_count'] >= m]
    data['score'] = data.apply(weighted_rating, axis=1, m=m, C=C)
    data.reset_index(drop=True, inplace=True)

    # convert to list
    data['genres'] = data['genres'].replace(np.nan, "[]")
    data['genres'] = data['genres'].apply(literal_eval)
    data['cast'] = data['cast'].replace(np.nan, "[]")
    data['cast'] = data['cast'].apply(literal_eval)
    data['crew'] = data['crew'].replace(np.nan, "[]")
    data['crew'] = data['crew'].apply(literal_eval)
    data['production_companies'] = data['production_companies'].replace(np.nan, "[]")
    data['production_companies'] = data['production_companies'].apply(literal_eval)
    data['production_countries'] = data['production_countries'].replace(np.nan, "[]")
    data['production_countries'] = data['production_countries'].apply(literal_eval)
    data['spoken_languages'] = data['spoken_languages'].replace(np.nan, "[]")
    data['spoken_languages'] = data['spoken_languages'].apply(literal_eval)

    # drop
    data = data.drop(['original_language', 'vote_average', 'vote_count', 'original_title'], axis=1)
    data.reset_index(drop=True, inplace=True)
    return data


def data_normalization(data):
    '''

    :param data:
    :return: normalized data
    '''
    max_min_scalar = lambda x: (x-np.min(x))/(np.max(x)-np.min(x))
    data['budget'] = data[['budget']].apply(max_min_scalar)
    data['runtime'] = data[['runtime']].apply(max_min_scalar)
    return data


def multi_onehot(data, n_cast=500, n_crew=500):
    '''

    :param data:
    :param n_cast: the number of cast want to preserved
    :param n_crew: the number of crew want to preserved
    :return: multi-onehot features
    '''
    # combine_data = combine_data.dropna(subset=['cast'])

    # extract features
    cast_statistics = data.explode('cast')
    cast_statistics = cast_statistics.loc[:, 'cast'].value_counts().reset_index()
    cast = cast_statistics['index'].tolist()

    crew_statistics = data.explode('crew')
    crew_statistics = crew_statistics.loc[:, 'crew'].value_counts().reset_index()
    crew = crew_statistics['index'].tolist()

    production_companies_statistics = data.explode('production_companies')
    production_companies_statistics = production_companies_statistics.loc[:, 'production_companies'].value_counts().reset_index()
    production_companies = production_companies_statistics['index'].tolist()

    production_countries_statistics = data.explode('production_countries')
    production_countries_statistics = production_countries_statistics.loc[:, 'production_countries'].value_counts().reset_index()
    production_countries = production_countries_statistics['index'].tolist()

    spoken_languages_statistics = data.explode('spoken_languages')
    spoken_languages_statistics = spoken_languages_statistics.loc[:, 'spoken_languages'].value_counts().reset_index()
    spoken_languages = spoken_languages_statistics['index'].tolist()


    # genres one-hot
    mlb_genres = MultiLabelBinarizer()
    mlb_result_genres = mlb_genres.fit_transform([i for i in data['genres']])
    mlb_result_genres = pd.DataFrame(mlb_result_genres, columns=list(mlb_genres.classes_))
    data = data.drop(['genres'], axis=1)
    data = pd.concat([data, mlb_result_genres], axis=1)

    # cast one-hot
    # combine_data['cast'] = combine_data['cast'].apply(clean_credits)
    mlb_cast = MultiLabelBinarizer(classes=cast[:n_cast])
    mlb_result_cast = mlb_cast.fit_transform([i for i in data['cast']])
    mlb_result_cast = pd.DataFrame(mlb_result_cast, columns=list(mlb_cast.classes_))
    data = data.drop(['cast'], axis=1)
    data = pd.concat([data, mlb_result_cast], axis=1)

    # crew one-hot
    # combine_data['crew'] = combine_data['crew'].apply(clean_credits)
    mlb_crew = MultiLabelBinarizer(classes=crew[:n_crew])
    mlb_result_crew = mlb_crew.fit_transform([i for i in data['crew']])
    mlb_result_crew = pd.DataFrame(mlb_result_crew, columns=list(mlb_crew.classes_))
    data = data.drop(['crew'], axis=1)
    data = pd.concat([data, mlb_result_crew], axis=1)

    # companies one-hot
    mlb = MultiLabelBinarizer(classes=production_companies[:500])
    mlb_result = mlb.fit_transform([i for i in data['production_companies']])
    mlb_result = pd.DataFrame(mlb_result, columns=list(mlb.classes_))
    data = data.drop(['production_companies'], axis=1)
    data = pd.concat([data, mlb_result], axis=1)

    # countries one-hot
    mlb = MultiLabelBinarizer(classes=production_countries[: 50])
    mlb_result = mlb.fit_transform([i for i in data['production_countries']])
    mlb_result = pd.DataFrame(mlb_result, columns=list(mlb.classes_))
    data = data.drop(['production_countries'], axis=1)
    data = pd.concat([data, mlb_result], axis=1)

    # spoken_languages one-hot
    mlb = MultiLabelBinarizer(classes=spoken_languages[: 50])
    mlb_result = mlb.fit_transform([i for i in data['spoken_languages']])
    mlb_result = pd.DataFrame(mlb_result, columns=list(mlb.classes_))
    data = data.drop(['spoken_languages'], axis=1)
    data = pd.concat([data, mlb_result], axis=1)

    return data

warnings.filterwarnings('ignore')

def FeatureEng_OneHot(inpath,outpath,save=False):
    # apply all one-hot process on the data
    combine_data = pd.read_csv(inpath+"combine.csv")
    combine_data = data_normalization(combine_data)
    combine_data = clean_data(combine_data)
    combine_data = multi_onehot(combine_data)

    if save:
        filename = "combine_clean_oh.csv"
        combine_data.to_csv(outpath+filename,index=False)
        print("the combined dataset is one-hot engineered and saved in {}".format(outpath))

    return combine_data
