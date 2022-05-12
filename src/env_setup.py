import os
import json

basedir = os.path.dirname(__file__)

def make_datadir():
  data_loc = os.path.join(basedir,'..','data')

  if not os.path.exists(data_loc):
    os.mkdir(data_loc)
    os.mkdir(os.path.join(data_loc,'raw'))
    os.mkdir(os.path.join(data_loc,'interim'))
    os.mkdir(os.path.join(data_loc,'final'))
    os.mkdir(os.path.join(data_loc,'interim','vectorizers'))
