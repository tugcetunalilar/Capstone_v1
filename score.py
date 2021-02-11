#from inference_schema.schema_decorators import input_schema, output_schema
#from inference_schema.parameter_types.numpy_parameter_type import NumpyParameterType
import json
import pickle
import pandas as pd
import numpy as np
import joblib
import os
from azureml.core.model import Model


# The init() method is called once, when the web service starts up.
#
# Typically you would deserialize the model file, as shown here using joblib,
# and store it in a global variable so your run() method can access it later.
def init():
    global model

    # The AZUREML_MODEL_DIR environment variable indicates
    # a directory containing the model file you registered.
    model_filename = 'model.pkl'
    #model_path = Model.get_model_path(model_filename)
    #model= joblib.load(model_path)
    model_path = os.path.join(os.environ['AZUREML_MODEL_DIR'], model_filename)

    model = joblib.load(model_path)


{"data": [{"age": 60,"anaemia":0,"creatinine_phosphokinase": 400,"diabetes": 0,"ejection_fraction": 20,"high_blood_pressure": 1,"platelets": 500000,"serum_creatinine": 1.3,"serum_sodium": 200,"sex": 0,"smoking": 0,"time": 1
}]}


# The run() method is called each time a request is made to the scoring API.
# Shown here are the optional input_schema and output_schema decorators
# from the inference-schema pip package. Using these decorators on your
# run() method parses and validates the incoming payload against
# the example input you provide here. This will also generate a Swagger
# API document for your web service.
#@input_schema('data', NumpyParameterType(np.array([[75,0,582,0,20,1,265000,1.9,130,1,0,4]])))
#@output_schema(NumpyParameterType(np.array([1,0])))
def run(data):
    try:
        data= json.loads(data)['data']
        data= pd.DataFrame.from_dict(data)
    # Use the model object loaded by init().
        result = model.predict(data)

    # You can return any JSON-serializable object.
        return result.tolist()
    except Exception as e:
        error= str(e)
        return error


