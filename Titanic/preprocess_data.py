import numpy as np
import pandas as pd



def preprocess(df, mean_age):
    # Fill missing Age
    df['Age'] = df['Age'].fillna(mean_age)
    
    # Fill missing Embarked (train only)
    if 'Embarked' in df.columns:
        df['Embarked'] = df['Embarked'].fillna('S')
        df = pd.get_dummies(df, columns=['Embarked'], prefix='Embarked')
    
    # Encode Sex
    df['Sex'] = df['Sex'].map({"male": 0, "female": 1})
    
    # IsAlone
    df['IsAlone'] = ((df['SibSp'] == 0) & (df['Parch'] == 0)).astype(int)
    
    # Extract titles
    df['Name'] = df['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)
    df['Name'] = df['Name'].replace(['Mlle','Mme', 'Ms', 'Lady','Dona'],'Miss')
    df['Name'] = df['Name'].replace(['Mme', 'Sir'],'Mrs')
    df['Name'] = df['Name'].replace(['Capt', 'Major'],'Col')
    df['Name'] = df['Name'].replace(['Don', 'Rev', 'Countess', 'Jonkheer'],'Others')
    df = pd.get_dummies(df, columns=['Name'], prefix='Name')
    
    # Drop unused columns
    drop_cols = [c for c in ['Cabin', 'Ticket'] if c in df.columns]
    df = df.drop(drop_cols, axis=1)
    
    return df


# Load data
train_data = pd.read_csv("Datasets/train.csv")
test_data = pd.read_csv("Datasets/test.csv")

mean_age = train_data['Age'].mean()
train_data = preprocess(train_data, mean_age)
test_data = preprocess(test_data, mean_age)

# Print column name, type, and null count for train_data
print("Column\tNulls\tType")
for col in train_data.columns:
    print(f"{col}\t{train_data[col].isnull().sum()}\t{train_data[col].dtype}")
print("------------------------------------------------------------------------")    
for col in test_data.columns:
    print(f"{col}\t{test_data[col].isnull().sum()}\t{test_data[col].dtype}")
    


# Save preprocessed train and test data
train_data.to_csv("Datasets/train_preprocessed.csv", index=False)
test_data.to_csv("Datasets/test_preprocessed.csv", index=False)
