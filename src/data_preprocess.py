import pandas as pd
import numpy as np
import utils as utils

from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.impute import SimpleImputer

from imblearn.under_sampling import RandomUnderSampler


def load_dataset(return_file=True):
    # Load train data
    X_train = utils.pickle_load(CONFIG_DATA['train_set_path'][0])
    y_train = utils.pickle_load(CONFIG_DATA['train_set_path'][1])

    # Load valid data
    X_valid = utils.pickle_load(CONFIG_DATA['valid_set_path'][0])
    y_valid = utils.pickle_load(CONFIG_DATA['valid_set_path'][1])

    # Load test data
    X_test = utils.pickle_load(CONFIG_DATA['test_set_path'][0])
    y_test = utils.pickle_load(CONFIG_DATA['test_set_path'][1])

    # Print 
    print("X_train shape :", X_train.shape)
    print("y_train shape :", y_train.shape)
    print("X_valid shape :", X_valid.shape)
    print("y_valid shape :", y_valid.shape)
    print("X_test shape  :", X_test.shape)
    print("y_test shape  :", y_test.shape)

    if return_file:
        return X_train, X_valid, X_test, y_train, y_valid, y_test

def fit_standardize(data, return_file=True, columns=['Time', 'Amount']):
    """Find standardizer data"""
    standardizer = RobustScaler()

    # Fit standardizer
    standardizer.fit(data[columns])

    # Dump standardizer
    utils.pickle_dump(standardizer, CONFIG_DATA['standardizer_path'])
    
    if return_file:
        return standardizer

def transform_standardize(data, standardizer, columns=['Time', 'Amount']):
    """Function to standardize data"""
    data_standard = pd.DataFrame(standardizer.transform(data[columns]))
    data_standard.index = data.index
    data[columns] = data_standard
    return data

def random_undersampler(X, y):
    """Function to under sample the majority data"""
    # Create resampling object
    ros = RandomUnderSampler(random_state = CONFIG_DATA['seed'])

    # Balancing the set data
    X_resample, y_resample = ros.fit_resample(X, y)

    # Print
    print('Distribution before resampling :')
    print(y.value_counts())
    print("")
    print('Distribution after resampling  :')
    print(y_resample.value_counts())

    return X_resample, y_resample

def clean_data(data,  standardizer):
    """Function to clean data"""

    # Standardize data
    data_standard = transform_standardize(data, standardizer)

    return data_standard

def _preprocess_data(data):
    """Function to preprocess data"""
    # Load preprocessor
    preprocessor = utils.pickle_load(CONFIG_DATA['preprocessor_path'])
    standardizer = preprocessor['standardizer']

    data_clean = clean_data(data,
                            standardizer)
    
    return data_clean

def generate_preprocessor(return_file=True):
    """Function to generate preprocessor"""
    # Load data
    X = utils.pickle_load(CONFIG_DATA['train_set_path'][0])
    y = utils.pickle_load(CONFIG_DATA['train_set_path'][1])

    # Generate preprocessor: standardizer
    standardizer = fit_standardize(X)

    # Dump file
    preprocessor = {
        'standardizer': standardizer
    }
    utils.pickle_dump(preprocessor, CONFIG_DATA['preprocessor_path'])
    
    if return_file:
        return preprocessor
    
def preprocess_data(type='train', return_file=True):
    """Function to preprocess train data"""
    # Load data
    X = utils.pickle_load(CONFIG_DATA[f'{type}_set_path'][0])
    y = utils.pickle_load(CONFIG_DATA[f'{type}_set_path'][1])
        
    # Preprocess data
    X_clean = _preprocess_data(X)
    y_clean = y

    # FOR TRAINING ONLY -> DO UNDERSAMPLING
    if type == 'train':
        X_clean, y_clean = random_undersampler(X_clean, y_clean)

    # Print shape
    print("X clean shape:", X_clean.shape)
    print("y clean shape:", y_clean.shape)

    # Dump file
    utils.pickle_dump(X_clean, CONFIG_DATA[f'{type}_clean_path'][0])
    utils.pickle_dump(y_clean, CONFIG_DATA[f'{type}_clean_path'][1])

    if return_file:
        return X_clean, y_clean   


if __name__ == '__main__':
    # 1. Load configuration file
    CONFIG_DATA = utils.config_load()

    # 2. Generate preprocessor
    generate_preprocessor()

    # 3. Preprocess Data
    preprocess_data(type='train')
    preprocess_data(type='valid')
    preprocess_data(type='test')
    