import pandas as pd
import os
from sklearn.preprocessing import LabelEncoder
from catboost_model.processing_utility import nan_to_probability, nan_to_median


TRAIN_DATASET_PATH = "dataset/train.csv"
TEST_DATASET_PATH = "dataset/test.csv"

cwd = os.getcwd()

def prepare_titanic_dataset(df_processed):
    # Encoding Sex values
    df_processed['Sex'] = LabelEncoder().fit_transform(df_processed['Sex'].values)

    # Replacing Embarked NaNs with random data 
    nan_to_probability(df_processed, 'Embarked')
    df_processed['Embarked'] = LabelEncoder().fit_transform(df_processed['Embarked'].values)

    return df_processed

train_df = pd.read_csv(os.path.join(cwd, TRAIN_DATASET_PATH))
train_processed = train_df[['Pclass', 'Sex', 'Fare', 'Embarked', 'Survived']]
train_processed = prepare_titanic_dataset(train_processed)
train_processed.to_csv(os.path.join(cwd, "dataset", "train_processed.csv"), index=False)

test_df = pd.read_csv(os.path.join(cwd, TEST_DATASET_PATH))
test_processed = test_df[['Pclass', 'Sex', 'Fare', 'Embarked']]
test_processed = prepare_titanic_dataset(test_processed)
test_processed.to_csv(os.path.join(cwd, "dataset", "test_processed.csv"), index=False)