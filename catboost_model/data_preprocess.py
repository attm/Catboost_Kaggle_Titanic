import pandas as pd
import os
from sklearn.preprocessing import LabelEncoder
from catboost_model.processing_utility import nan_to_probability, nan_to_median


TRAIN_DATASET_PATH = "dataset/train.csv"
TEST_DATASET_PATH = "dataset/test.csv"

cwd = os.getcwd()

# Processing train dataset
train_df = pd.read_csv(os.path.join(cwd, TRAIN_DATASET_PATH))
train_processed = train_df

nan_to_probability(train_processed, 'Embarked')
nan_to_median(train_processed, 'Age')

train_processed.to_csv(os.path.join(cwd, "dataset", "train_processed.csv"), index=False)

# Processing test dataset
test_df = pd.read_csv(os.path.join(cwd, TEST_DATASET_PATH))
test_processed = test_df

nan_to_probability(test_processed, 'Embarked')
nan_to_median(test_processed, 'Age')
nan_to_median(test_processed, 'Fare')

test_processed.to_csv(os.path.join(cwd, "dataset", "test_processed.csv"), index=False)

# Creating submission placeholder
submission_df = test_df[["PassengerId"]]
submission_df.to_csv(os.path.join(cwd, "dataset", "submission.csv"), index=False)