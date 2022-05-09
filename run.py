import sys
import os
import json

from src.data_clean import *

def main(targets):

  if 'data' in targets:
    with open('config/data-params.json') as fh:
      data_param = json.load(fh)
    
    PreprocessMM(data_param["data_path"], data_param["out_path"],True)
    PreprocessC(data_param["data_path"], data_param["out_path"],True)

  return

if __name__ == '__main__':
  targets = sys.argv[1]
  main(targets)