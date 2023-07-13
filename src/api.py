import pandas as pd
import utils as utils

from fastapi import FastAPI, File, UploadFile
import uvicorn


# CLASS & FUNCTIONS
# Load config data
CONFIG_DATA = utils.config_load()

def transform_standardize(data, standardizer, columns=['Time', 'Amount']):
    """Function to standardize data"""
    data_standard = pd.DataFrame(standardizer.transform(data[columns]))
    data_standard.index = data.index
    data[columns] = data_standard
    return data

class Model:
    def __init__(self):
        """Initialize preprocessor, model, & threshold"""
        self.preprocessor = utils.pickle_load(CONFIG_DATA['preprocessor_path'])
        self.model = utils.pickle_load(CONFIG_DATA['best_model_path'])    

    def preprocess(self, X):
        """Function to preprocess data"""
        X = X.copy()
        X_clean = transform_standardize(data = X,
                                        standardizer = self.preprocessor['standardizer'])
        
        return X_clean
    
    def predict(self, X):
        """Function to predict the data"""
        # Preprocess data
        X_clean = self.preprocess(X)

        # Predict data
        y_pred = self.model.predict(X_clean)

        # Predict dictionary
        y_pred_dict = {'label': [int(i) for i in y_pred]}
        return y_pred_dict

# FASTAPI
app = FastAPI()

@app.get('/')
def home():
    return {'text': 'our first route'}

@app.post('/predict')
def create_upload_file(file: UploadFile = File(...)):
    # Hanlde the file only if it is a csv
    if file.filename.endswith('.csv'):
        # Read file
        with open(file.filename, 'wb') as f:
            f.write(file.file.read())
        data = pd.read_csv(file.filename)

        # Modeling file
        model = Model()
        y_pred = model.predict(data)

        return y_pred


if __name__ == '__main__':
    uvicorn.run('api:app',
                host = '127.0.0.1',
                port = 8000)