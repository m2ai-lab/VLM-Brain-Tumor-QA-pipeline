import os
import sys
import socket
import pandas as pd
import numpy as np
import dask
import dask.dataframe as dd
import dask.array as da
import dask.bag as db
from dask_sql import Context
from dask_jobqueue import SGECluster
from dask.distributed import Client, LocalCluster

cluster = LocalCluster(n_workers=8, memory_limit='128gb')
client = Client(cluster)

rwd_output = './assets/data/'

def load_register_table(data_asset, table, **kwargs):
    return dd.read_parquet(f'/wynton/protected/project/ic/data/parquet/{data_asset}/{table}/', **kwargs)

client.dashboard_link

#Get image data and filter out the data based on AccessionNumber
#SINCE IT IS NONE NOTHING WILL SHOW UP. BE SURE TO CHANGE THIS
images = load_register_table("IMAGING", "series")
images = images[images['AccessionNumber'] == "None"]

#This gets the first row / entry
test = images.head(1)

#Use this to get each row and its data in the dataframe
for index, row in test.iterrows():
    print(dict(row.items()))
