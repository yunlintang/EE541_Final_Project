import sys
import os
import json

from src.data_clean import *
from src.feature_engineering import *

def main(targets):

  with open('config/params.json') as fh:
    param = json.load(fh)

  if 'data' in targets:
    DataCleaning(param["raw_data"], param["inte_data"],True)
  
  if 'feature' in targets:
    FeatureEng_OneHot(param['inte_data'],param['inte_data'],True)
    Combine_Features(param['inte_data'],param['vect_data'],param['title'],
                     param['overview'],param['final_data'],True)

  return

if __name__ == '__main__':
  targets = sys.argv[1:]
  main(targets)