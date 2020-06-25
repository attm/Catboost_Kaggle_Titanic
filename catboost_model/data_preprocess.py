import pandas as pd 
import os 
from sklearn.preprocessing import LabelEncoder


TRAIN_DATASET_PATH = "dataset/train.csv"
TEST_DATASET_PATH = "dataset/test.csv"

cwd = os.getcwd()

train_df = pd.read_csv(os.path.join(cwd, TRAIN_DATASET_PATH))
train_df_picked = train_df[['Survived', 'Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Cabin', 'Embarked']]
train_df_picked['Sex'] = LabelEncoder().fit_transform(train_df_picked['Sex'].values) 
train_df_picked['Embarked'].fillna('C', inplace=True)
train_df_picked['Embarked'] = LabelEncoder().fit_transform(train_df_picked['Embarked'].values) 
print(train_df_picked)
train_df_picked.to_csv(os.path.join(cwd, "dataset", "train_processed.csv"))
