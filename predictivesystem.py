import numpy as np
from sklearn.linear_model import LogisticRegression
import pandas as pd
import matplotlib.pyplot as plt
import pickle
loaded_model = pickle.load(open('/Users/rakeshkanneeswaran/Desktop/trained_model.sav','rb'))
print("everthing is ok")


input_data = (11,138,76,0,0,33.2,0.42,35)
#changing input data into numpy array

input_data_as_numpy_array = np.asarray(input_data)
#reshape the array
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)
# standrdize the input data


prediction = loaded_model.predict(input_data_reshaped)
if (prediction[0] == 0):
  print("the preson is not diabetic")
else:
  print("The person is diabetic")  