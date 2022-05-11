import numpy as np
import pandas as pd
from os.path import exists

from ast import literal_eval

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import MultiLabelBinarizer

import torch
import torch.nn as nn

from sentence_transformers import SentenceTransformer
from gensim.models.doc2vec import Doc2Vec, TaggedDocument

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


def multi_onehot(data, n_cast=300, n_crew=300):
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



# download necessary packages
nltk.download('wordnet')
nltk.download('stopwords')
nltk.download("omw-1.4")
sw = stopwords.words('english')
lemmatizer = WordNetLemmatizer()

def remove_stop(x):
    '''
    function to remove the stopwords(English) from the
    input list of string
    
    Params:
        x: a list of string (ex. ['a','b','c'])
        
    Returns:
        returns a new a list of string
    '''
    try:
        new_list = [i for i in x.split(" ") if i not in sw ]
    except:
        return np.nan
    return new_list


def lemma(x):
    '''
    function to lemmatize the input(a list of string)
    
    Params:
        x: a list of string (ex. ['a','b','c'])
        
    Returns:
        a string the combine all strings in the result list
        (ex. "a b c")
    '''
    if x is np.nan:
        return ""
    new_list = []
    for i in x:
        new_list += [lemmatizer.lemmatize(i)]
    
    return ' '.join(new_list)


def tfidf(df,var,mdf=1,mfeatures=None,return_vec=False):
    '''
    function to vectorize the text feature(ex. "a b c")
    into numeric vector by using the TF-IDF method.
    Note: can be modified to reduce dimension(use "min_df" and "max_features")
    
    Params:
        df: input dataset
        var: text feature that need to be vectorized
        mdf: min_df
        mfeatures: max_features
        return_vec: if true, return the fitted vectorizer
        
    Returns:
        the vectorized text feature(vector of numeric vectors)
    '''
    vectorizer = TfidfVectorizer(stop_words='english',
                                 min_df=mdf, max_features=mfeatures)
    vec = vectorizer.fit_transform(df[var])
    
    # Note: to vectorize a unseen vector(a string), use:
    # model.transform()
    if return_vec:
        return (vectorizer, vec.toarray())
    return vec.toarray()


def nnemb(df,var,d):
    '''
    function to vectorize the input text feature(ex. "a b c") by
    using the word embedding method from PyTorch
    Note: compute meach word embedding vecor for each input text 
    then use it as the numeric representaion of the input text
    
    Params:
        df: the input dataset
        var: the feature name
        d: dimension want to be kept
        
    Returns:
        return the vectorized text
    '''
    # use countvectorizer to compute the vocab. for the input
    vectorizer = CountVectorizer()
    vectorizer.fit(df[var])
    vocab = vectorizer.vocabulary_
    # construct the word embedding model
    embeds = nn.Embedding(len(vocab),d)
    
    def ConvertToVec(x):
        '''
        function to vectorize the single input of string
        '''
        # function to preprocess the string by the countvectorizer
        t = vectorizer.build_analyzer()
        # generate tensors for each word in the input string by
        # indexing the vocab
        lookup_tensor = torch.tensor([vocab[i] for i in t(x)],dtype=torch.long)
        # compute the numeric vector
        x_vec = embeds(lookup_tensor)
        x_vec = torch.mean(x_vec,axis=0)
        return x_vec
    
    return df[var].apply(ConvertToVec)


def senbert(df,var,model_name,params=None,return_vec=False):
    '''
    function to vectorize the input text feature by using the
    selected Sentence-BERT model.
    (models can be found on:
    https://docs.google.com/spreadsheets/d/14QplCdTCDwEmTqrn1LH4yrbKvdogK4oQvYO1K1aPR5M/edit#gid=0)
    
    Params:
        df: input dataset
        var: the name of text feature
        model_name: name of model
        params: params for the model
        return_vec: if true, return the fitted vectorizer
        
    Return:
        the vectorized feature text
    '''
    model = SentenceTransformer(model_name)
    if params is not None:
        params['sentences'] = df[var]
        X = model.encode(**params)
    else:
        X = model.encode(df[var],batch_size=10)
        
    # return the trained model
    # Note: to vectorize a unseen vector(a string), use:
    # model.encode()[0]
    if return_vec:
        return (model,X)
    return X


def doc2vec(df,var,d,return_vec=False):
    '''
    function to convert the input text feature by using
    the Doc2Vec model by Gensim.
    
    Params:
        df: input dataset
        var: the name of text feature
        d: dimension of feature to be kept
        return_vec: if true, return the fitted vectorizer
        
    Returns:
        
    '''
    sen = [TaggedDocument(sen,[i]) for i,sen in enumerate(df[var].values)]
    model = Doc2Vec(sen,vector_size=d)
    X = [model.dv[i] for i in range(len(df[var]))]
    
    # return the trained model
    # Note: to vectorize a unseen vector(a list of strings), use:
    # model.infer_vector()
    if return_vec:
        return (model,X)
    
    return X


def vectorize(df,vectpath):
    '''
    function to compute all vectorized text features by using
    3 different vectorizers: hashing, doc2vec, and sentence-BERT
    '''
    # save senbert vectorized vectors for overview
    X_senbert = senbert(df,'overview','stsb-distilroberta-base-v2')
    np.savetxt(vectpath+"overview_senbert.txt", X_senbert,delimiter=',')

    # save Doc2Vec vectorized vectors for overview
    X_doc2vec = doc2vec(df,'overview',500)
    np.savetxt(vectpath+"overview_doc2vec.txt", X_doc2vec,delimiter=',')

    # save hashing vectorized vectors for overview
    X_hash = hashing(df,'overview',500)
    np.savetxt(vectpath+"overview_hash.txt", X_hash,delimiter=',')
    
    # save senbert vectorized vectors for title
    X_senbert_t = senbert(df,'title','stsb-distilroberta-base-v2')
    np.savetxt(vectpath+"title_senbert.txt", X_senbert_t,delimiter=',')

    # save Doc2Vec vectorized vectors for title
    X_doc2vec_t = doc2vec(df,'title',500)
    np.savetxt(vectpath+"title_doc2vec.txt", X_doc2vec_t,delimiter=',')

    # save hashing vectorized vectors for title
    X_hash_t = hashing(df,'title',500)
    np.savetxt(vectpath+"title_hash.txt", X_hash_t,delimiter=',')
    
    print("all vectorized text features are saved in {}".format(vectpath))
    
    return



def Combine_Features(inpath,vectpath,title_,overview_,outpath,save=False):
    '''
    function to combine all features

    Params:
        inpath: the path for the cleaned dataset
        vectpath: path for the vectorized text features
        title_: the file name of selected vectors for "title"
        overview_: the file name of selected vectors for "overview"
        outpath: the path for final dataset
    '''
    # read the combine data after one-hot enginnering
    df = pd.read_csv(inpath+'combine_clean_oh.csv')
    
    # preprocess the text features
    df['title'] = df['title'].apply(remove_stop).apply(lemma)
    df['overview'] = df['overview'].apply(remove_stop).apply(lemma)
    
    # vectorize the text features if not exsited
    if not (exists(vectpath+title_) and exists(vectpath+overview_)):
        vectorize(df,vectpath)
        
    # read the vectorized text features("overview" and "title")
    df_overview = pd.read_csv(vectpath+overview_,header=None).add_prefix("Overview_")
    df_title = pd.read_csv(vectpath+title_,header=None).add_prefix("title_")
    
    # combine the dataset
    df_comb = pd.concat([df,df_overview,df_title],axis=1)
    # drop features and move score to the last column
    df_comb['new_score'] = df_comb['score']
    df_comb = df_comb.drop(columns=['overview','title','keywords','score','id'])
    df_comb = df_comb.rename(columns={'new_score':'score'})
    
    # save
    if save:
        filename = "data.csv"
        df_comb.to_csv(outpath+filename,index=False)
        print("the final dataset after all feature enginnerings is saved in {}".format(outpath))
        
    return df_comb