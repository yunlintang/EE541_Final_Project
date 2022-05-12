# EE541_Final_Project
Authors: Yunlin Tang, Yixiang Zheng

## Set Up
- python 3
- If needed, install the used modules included in the requirements.txt:
```
pip install -r requirements.txt
```
- **IMPORTANT**:
  - run command **`python run.py`** under the project root repository to create data folder and its subfolders
  - then download all the datasets from the [Kaggle webstie](https://www.kaggle.com/datasets/rounakbanik/the-movies-dataset?select=ratings_small.csv) to the folder `"data/raw"`


## Repository Structure
```
├── config
│   ├── params.json
├── data
│   ├── raw
│   ├── interim
│   │   ├── vectorizers
│   ├── final
├── notebooks
├── src
│   ├── data_clean.py
│   ├── env_setup.py
│   ├── feature_engineering.py
│   ├── model_building.py
├── README.md
├── requirements.txt
├── run.py
└── .gitignore
```

- `config` contains json files for all the input params, can be modified as necessary
- `data` contains two folder: `raw`, `interim` and `final`
  - `data/interim` contains one folder `vectorizers`
- `notebooks` contains Jupyter notebook files
- `src` contains py files


## Building
- data
  - run **`python run.py data`** to clean the raw data, the results will be saved in the folder `data/interim`
- feature
  - run **`python run.py feature`** to perform feature engineering on the combine dataset(saved in `data/interim`), and the final dataset is saved in `data/final`
- train
  - run **`python run.py train`** to train two regression models
- All targets
  - run **`python run.py all`** to perform all above targers (equivalent command: **`python run.py data feature train`**)