import pandas as pd
import numpy as np
import json
import re

def Spok_Lang(x):
    '''
    For spoken_languages feature:
    Input a string of JSON, get the values for which the
    keys are "iso_639_1" (stands for country code) then 
    convert the string into a list
    '''
    if ((x is np.nan) or (x=='[]')):
        return np.nan
    
    # would have invalid escape error
    x = re.sub(r'\\','',x)
    x_json = json.loads(x.replace("\'","\""))
    return [i['iso_639_1'] for i in x_json] 


def Gen(x):
    '''
    For genres feature:
    Input a string of JSON, get the values for which the
    keys are "name" then convert the string into a list
    '''
    if ((x is np.nan) or (x=='[]')):
        return np.nan
    
    x_json = json.loads(x.replace("\'","\""))
    return [i['name'] for i in x_json]

    
def Prod_Count(x):
    '''
    For production_countries feature:
    Input a string of JSON, get the values for which the
    keys are "iso_639_1" (stands for country code) then 
    convert the string into a list
    '''
    if ((x is np.nan) or (x=='[]')):
        return np.nan
    
    try:
        # some observations have ill-formatted values just drop them
        x_json = json.loads(x.replace("\'","\""))
        x_list = [i['iso_3166_1'] for i in x_json]
    except:
        return np.nan
    
    return x_list


def Prod_Com(x):
    '''
    For production_companies feature:
    Input a string of JSON, get the values for which the
    keys are "name" then convert the string into a list
    '''
    if ((x is np.nan) or (x=='[]')):
        return np.nan
    
    try:
        # some observations have ill-formatted values just drop them
        x_json = json.loads(x.replace("\'","\""))
        x_list = [i['name'] for i in x_json]
    except:
        return np.nan
    
    return x_list


def ConvertToFloat(x):
    '''
    for feature budget: convert the string
    into float. If the input is ill-formatted,
    then return Nan
    '''
    try:
        float(x)
    except:
        return np.nan
    return float(x)


def Cas(X):
    '''
    For cast feature:
    Input a string of JSON, get the values for which the
    keys are "name"(name of actors) then convert the string into a list
    '''
    # remove some invalid char
    x = re.sub("\"[^\"]*\",","null,",X)
    x = re.sub("\"[^\"]*\"}","null}",x)
    x = x.replace("\"", "")
    x = x.replace("\'","\"").replace("None","null")
    # replace the character into null
    x = re.sub(r'\\','',x)
    x = re.sub("\"character\":\s.+?(?=,\s\")","\"character\": null",x)
    
    try:
        x_json = json.loads(x)
    except:
        return
    
    x_list = [i['name'] for i in x_json]
    
    if len(x_list) == 0:
        return
    return x_list


def Cre(x):
    '''
    For crew feature:
    Input a string of JSON, get the values for which the
    keys are "name"(name of directors) then convert the string into a list
    '''
    x = re.sub("\"[^\"]*\",","null,",x)
    x = x.replace("\"", "")
    x = x.replace("\'","\"").replace("None","null")
    x = re.sub(r'\\','',x)
    x_json = json.loads(x)
    x_list = [i['name'] for i in x_json if i['job'] == 'Director' ]
    
    if len(x_list) == 0:
        return 
    return x_list


def Keyw(x):
    x = re.sub("\"[^\"]*\",","null,",x)
    x = re.sub("\"[^\"]*\"}","null}",x)
    x = x.replace("\"", "")
    x = x.replace("\'","\"").replace("None","null")
    x = re.sub(r'\\','',x)
    
    try:
        x_json = json.loads(x)
    except:
        return
    
    x_list = [i['name'] for i in x_json]
    if len(x_list) == 0:
        return 
    
    return x_list


def PreprocessMM(inpath,outpath,save=False):
    '''
    clean the raw movies_metadata.csv dataset
    
    Params:
        path: path of the input dataset(Ex. '../data/raw/movies_metadata.csv')
        save: specify if the cleaned dataset 
              need to be saved in the '../data/interim/'
    '''
    # read data
    filename = "movies_metadata.csv"
    df_mm = pd.read_csv(inpath+filename,low_memory=False)
    # drop columns
    df_mm = df_mm.drop(columns=['belongs_to_collection','homepage','tagline','poster_path'])
    
    # convert JSON in string to list
    df_mm['spoken_languages'] = df_mm['spoken_languages'].apply(Spok_Lang)
    df_mm['genres'] = df_mm['genres'].apply(Gen)
    df_mm['production_countries'] = df_mm['production_countries'].apply(Prod_Count)
    df_mm['production_companies'] = df_mm['production_companies'].apply(Prod_Com)
    
    # convert the date string to datetime
    df_mm['release_date'] = pd.to_datetime(df_mm['release_date'],errors='coerce')
    # convert to boolean values
    df_mm['adult'] = df_mm['adult'].apply(lambda x: True if x == 'True' else False)
    # convert from string to float
    df_mm['budget'] = df_mm['budget'].apply(ConvertToFloat)
    df_mm['popularity'] = df_mm['popularity'].apply(ConvertToFloat)
    
    if save:
        filename = "movies_metadata_clean.csv"
        df_mm.to_csv(outpath+filename,index=False)
        print("the cleaned dataset of movies_metadata is saved in {}".format(outpath))
    
    return df_mm


def PreprocessC(inpath,outpath,save=False):
    '''
    clean the raw credits.csv dataset
    
    Params:
        path: path of the input dataset(Ex. '../data/raw/credits.csv')
        save: specify if the cleaned dataset 
              need to be saved in the '../data/interim/'
    '''
    # read data
    filename = "credits.csv"
    df_c = pd.read_csv(inpath+filename)
    
    # convert the json data
    df_c['cast'] = df_c['cast'].apply(Cas)
    df_c['crew'] = df_c['crew'].apply(Cre)
    
    if save:
        filename = "credits_clean.csv"
        df_c.to_csv(outpath+filename,index=False)
        print("the cleaned dataset of credits is saved in {}".format(outpath))
    
    return df_c


def PreprocessK(inpath,outpath,save=False):
    '''
    clean the raw keywords.csv dataset
    
    Params:
        path: path of the input dataset(Ex. '../data/raw/keywords.csv')
        save: specify if the cleaned dataset 
              need to be saved in the '../data/interim/'
    '''
    # read data
    filename = "keywords.csv"
    df_k = pd.read_csv(inpath+filename)
    
    # convert the json data
    df_k['keywords'] = df_k['keywords'].apply(Keyw)
    
    if save:
        filename = "keywords_clean.csv"
        df_k.to_csv(outpath+filename,index=False)
        print("the cleaned dataset of keywords is saved in {}".format(outpath))
    
    return df_k


def DataCleaning(inpath,outpath,save=False):
    # preprocess these three datasets
    df_mm = PreprocessMM(inpath,outpath,True)
    df_c = PreprocessC(inpath,outpath,True)
    df_k = PreprocessK(inpath,outpath,True)
    
    # convert the type of id to numeric format
    df_mm['id'] = pd.to_numeric(df_mm['id'],errors='coerce')
    df_mm = df_mm.dropna(subset=['id'])
    
    # merge three datasets
    df_comb = pd.merge(df_c,df_k,on='id',how='inner')
    df_comb = pd.merge(df_comb, df_mm,on='id',how='inner')
    
    # drop pulicates and reset index
    df_comb = df_comb.drop_duplicates(subset=['id']).reset_index(drop=True)
    df_comb = df_comb.drop(columns=['adult','imdb_id','popularity','release_date',
                                    'revenue','status','video'])
    
    if save:
        filename = "combine.csv"
        df_comb.to_csv(outpath+filename,index=False)
        print("the combined dataset is saved in {}".format(outpath))
    
    return df_comb