import pandas as pd
from sklearn.model_selection import train_test_split
# create the over sampled dataset to help deal with the imbalanced nature of the data
from imblearn.over_sampling import SMOTE


def create_dataset():
    data = pd.read_csv("c:\\Users\\markm\\Desktop\\CAPSTONE\\capstone\\data\\external\\watson_healthcare_modified.csv")
    # let's slice the data into x and y, for building predictive models
    y = data['Attrition']
    x = data.drop(columns=['EmployeeID','Attrition'])

    # feature engineer the target variable so that it is quantitative
    y.replace('No', 0, inplace=True)
    y.replace('Yes', 1, inplace = True)

    # based off of this analysis, let's drop 'EmployeeCount' & 'StandardHours' because there is no variance in these features
    x = x.drop(columns=['Over18','EmployeeCount','StandardHours'], axis=1)

    x = convert_numeric(x)

    # split the data after numeric conversion
    x_train, x_test, y_train, y_test = train_test_split(x,y, test_size =0.2, random_state=42)

    # transform the dataset
    oversample = SMOTE()
    x_bal_train, y_bal_train = oversample.fit_resample(x_train, y_train)

    return x_bal_train, y_bal_train, x_train, y_train, x_test, y_test


### Script for encoding dummy variables
def convert_numeric(data):

    # Get one hot encoding of columns B
    data_ohe = data
    categorical_columns = data.select_dtypes(include=['object']).columns
    for col in categorical_columns:
        col_ohe = pd.get_dummies(data[col], prefix=col)
        data_ohe = pd.concat((data_ohe, col_ohe), axis=1).drop(col, axis=1)

    return data_ohe