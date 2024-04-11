#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
from flask import Flask, request, render_template
import pickle
import matplotlib.pyplot as plt
#import base64
#import io


# In[ ]:


app = Flask(__name__)


# In[ ]:


model = pickle.load(open('models/model.pkl', 'rb'))


# In[ ]:


#we define a function home which renders our HTML file we created in the previous step as our homepage
@app.route('/')
def home():
    return render_template('index.html')


# In[ ]:


@app.route('/predict',methods=['POST'])
def predict():
    features = [x for x in request.form.values()]
    features_2 = [np.array(features)]  
    prediction = model.predict(features_2) 
    result = prediction[0]

    return render_template('index.html', prediction=result)


# In[ ]:


if __name__ == "__main__":
    app.run(debug=True)


# In[ ]:




