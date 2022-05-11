# EE541_Final_Project

## Requirement
- python 3
- If needed, install the used modules included in the requirements.txt:
```
pip install -r requirements.txt
```
- **IMPORTANT**: 
  - create folders under the root folder call `"data"`
  - then create folders under the `"data"`: `"raw"`, `"interim"`, and `"final"`, also create a folder `"vectorizers"` under `"data/interim"`
  - then download all the datasets from the [Kaggle webstie](https://www.kaggle.com/datasets/rounakbanik/the-movies-dataset?select=ratings_small.csv) to the folder `"data/raw"`
- python3 -m nltk.downloader stopwords

## Layout
- `config` contains json files for all the input params.
- `data` contains two folder: `raw`, `interim` and `final`
  - `data/interim` contains one folder `vectorizers`
- `notebooks` contains Jupyter notebook files
- `src` contains py files


## Building
- data
  - run **`python run.py data`** or **`python3 run.py data`** to clean the raw data, the results will be saved in the folder `data/interim`
- feature
  - run **`python run.py feature`** to perform feature engineering on the combine dataset(saved in `data/interim`), and the final dataset is saved in `data/final`
- All targets
  - run **`python run.py data feature`**