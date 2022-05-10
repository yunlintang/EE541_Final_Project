# EE541_Final_Project

## Requirement
- python 3
- If needed, install the used modules included in the requirements.txt:
```
pip install -r requirements.txt
```
- **IMPORTANT**: 
  - create folders under the root folder call `"data/raw"` and `"data/interim"`
  - then download all the datasets from the [Kaggle webstie](https://www.kaggle.com/datasets/rounakbanik/the-movies-dataset?select=ratings_small.csv) to the folder `"data/raw"`
- python3 -m nltk.downloader stopwords

## Layout
- `config` contains json files for all the input params.
- `data` contains two folder: `raw` and `interim`
- `notebooks` contains Jupyter notebook files
- `src` contains py files


## Building
- data
  - run **`python run.py data`** or **`python3 run.py data`** to clean the raw data, the results will be saved in the folder `data/interim`