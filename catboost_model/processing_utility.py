import sklearn as sk
import pandas as pd
import numpy as np


# Replaces all nan to median in pandas dataframe
def nan_to_median(dataframe, c_name):
    median = dataframe[c_name].median()
    dataframe[c_name].fillna(median, inplace=True)

# Replaces all nan to random variable with respect of probality in pandas dataframe
def nan_to_probability(dataframe, c_name):
    df = dataframe.dropna()
    probs_names = df.groupby(c_name).size().div(len(df)).index.to_numpy()
    probs = df.groupby(c_name).size().div(len(df)).to_numpy()
    dataframe[c_name].fillna(pick_random(probs_names, probs), inplace=True)

# Helper for nan_to_probability
def pick_random(probs_names, probs):
    pick = np.random.choice(probs_names, p=probs)
    return pick

# Replaces all nan's with constant variable
def nan_to_const(dataframe, c_name, const):
    dataframe[c_name].fillna(const, inplace=True)

