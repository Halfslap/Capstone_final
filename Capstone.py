#!/usr/bin/env python
# coding: utf-8

# In[2960]:


import pandas as pd
##%matplotlib inline
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
from plotly import graph_objects
import streamlit as st


# ## My Info

# In[2961]:


st.title('Computer Science Capstone — C964')
st.header('Student – Mark Nefzger')
st.header('Student ID: 001411596')


# ## Version Info

# In[2962]:


pd_ver = pd.__version__
st.write("Pandas version: ", pd_ver)


# In[2963]:


np_ver = np.__version__
st.write("Numpy version: ", np_ver)


# In[2964]:


st_ver = st.__version__
st.write("Streamlit version: ", st_ver)


# In[2965]:


plt_ver = matplotlib.__version__
st.write("Matplotlib version: ", plt_ver)


# ## Import data

# In[2966]:


# Import Data
health_data = pd.read_csv("New Data/oura_2019-01-01_2023-09-09_trends_Shifted.csv")


# ## View data

# In[2967]:


st.header('Imported Data')


# In[2968]:


health_data


# In[2969]:


health_data.info()


# ## Describe Data

# In[2970]:


health_data.describe()


# ## Set up dataframe

# In[2971]:


df = pd.DataFrame(health_data)


# ## Convert Sleep Duration and Rest Time to hours

# In[2972]:


df["Total Sleep Duration"] = df["Total Sleep Duration"] / 3600
df["Rest Time"] = df["Rest Time"] / 3600


# In[2973]:


pd.crosstab(df["Total Sleep Duration"] > 7, df["Readiness Score"] >85)


# In[2974]:


(df["Readiness Score"].hist(figsize=(10, 10)))


# ## Manipulating Data

# In[2975]:


df.dropna(inplace=True)


# In[2976]:


# Randomize data 1 = 100%
df.sample(frac=1)


# In[2977]:


# Reset index if necessary
# df.reset_index(drop=True, inplace=True)


# ## Matplotlib

# In[2978]:


# 1. Prepare data
x = df["Total Sleep Duration"]
y = df["Readiness Score"]

# 2. Setup plot
fig, ax = plt.subplots(figsize=(10, 10))

# 3. Plot data
ax.scatter(x,y)

# 4. Customize plot
ax.set(title="Total Sleep vs. Readiness", 
       xlabel="Total Sleep Duration",
       ylabel="Readiness Score")

# 5. Save and show
fig.savefig("Figures/Figure_1.png")


# In[2979]:


# 1. Prepare data
x = df["Previous Night Score"]
y = df["Readiness Score"]

# 2. Setup plot
fig, ax = plt.subplots(figsize=(10, 10))

# 3. Plot data
ax.scatter(x,y)


# 4. Customize plot
ax.set(title="Previous Night Sleep vs Readiness", 
       xlabel="Previous Night Score",
       ylabel="Readiness Score")

# 5. Save and show 
fig.savefig("Figures/Figure_2.png")


# In[2980]:


# 1. Prepare data
x = df["Move Every Hour Score"]
y = df["Readiness Score"]

# 2. Setup plot
fig, ax = plt.subplots(figsize=(10, 10))

# 3. Plot data
ax.bar(x,y)

# 4. Customize plot
ax.set(title="Move Every Hour vs Readiness", 
       xlabel="Move Every Hour Score",
       ylabel="Readiness Score")

# 5. Save and show 
fig.savefig("Figures/Figure_3.png")


# In[2981]:


# 1. Prepare data
x = df["Non-wear Time"]
y = df["Readiness Score"]

# 2. Setup plot
fig, ax = plt.subplots(figsize=(10, 10))

# 3. Plot data
ax.scatter(x,y)

# 4. Customize plot
ax.set(title="Non-wear vs Readiness", 
       xlabel="Non-wear Time",
       ylabel="Readiness Score")

# 5. Save and show 
fig.savefig("Figures/Figure_4.png")


# In[2982]:


# 1. Prepare data
x = df["Rest Time"]
y = df["Readiness Score"]

# 2. Setup plot
fig, ax = plt.subplots(figsize=(10, 10))

# 3. Plot data
ax.scatter(x,y)

# 4. Customize plot
ax.set(title="Rest Time vs Readiness", 
       xlabel="Rest Time",
       ylabel="Readiness Score")

# 5. Save and show 
fig.savefig("Figures/Figure_5.png")


# In[2983]:


# 1. Prepare data
x = df["Previous Day Activity Score"]
y = df["Readiness Score"]

# 2. Setup plot
fig, ax = plt.subplots(figsize=(10, 10))

# 3. Plot data
ax.scatter(x,y)

# 4. Customize plot
ax.set(title="Previous Day Activity vs Readiness", 
       xlabel="Previous Day Activity Score",
       ylabel="Readiness Score")

# 5. Save and show 
fig.savefig("Figures/Figure_6.png")


# In[2984]:


# 1. Prepare data
x = df["Activity Score"]
y = df["Readiness Score"]

# 2. Setup plot
fig, ax = plt.subplots(figsize=(10, 10))

# 3. Plot data
ax.scatter(x,y)

# 4. Customize plot
ax.set(title="Activity Score vs Readiness", 
       xlabel="Activity Score",
       ylabel="Readiness Score")

# 5. Save and show 
fig.savefig("Figures/Figure_7.png")


# In[2985]:


# 1. Prepare data
x = df["Resting Heart Rate Score"]
y = df["Readiness Score"]

# 2. Setup plot
fig, ax = plt.subplots(figsize=(10, 10))

# 3. Plot data
ax.scatter(x,y)

# 4. Customize plot
ax.set(title="Resting Heart Rate vs Readiness", 
       xlabel="Resting Heart Rate Score",
       ylabel="Readiness Score")

# 5. Save and show 
fig.savefig("Figures/Figure_8.png")


# In[2986]:


# 1. Prepare data
x = df["Temperature Score"]
y = df["Readiness Score"]

# 2. Setup plot
fig, ax = plt.subplots(figsize=(10, 10))

# 3. Plot data
ax.scatter(x,y)

# 4. Customize plot
ax.set(title="Temperature vs Readiness", 
       xlabel="Temperature Score",
       ylabel="Readiness Score")

# 5. Save and show 
fig.savefig("Figures/Figure_9.png")


# In[2987]:


# 1. Prepare data
x = df["Activity Balance Score"]
y = df["Readiness Score"]

# 2. Setup plot
fig, ax = plt.subplots(figsize=(10, 10))

# 3. Plot data
ax.scatter(x,y)

# 4. Customize plot
ax.set(title="Activity Balance Score vs Readiness", 
       xlabel="Activity Balance Score",
       ylabel="Readiness Score")

# 5. Save and show 
fig.savefig("Figures/Figure_10.png")


# In[2988]:


# 1. Prepare data
x = df["Temperature Trend Deviation"]
y = df["Readiness Score"]

# 2. Setup plot
fig, ax = plt.subplots(figsize=(10, 10))

# 3. Plot data
ax.scatter(x,y)

# 4. Customize plot
ax.set(title="Temperature Trend Deviation vs Readiness", 
       xlabel="Temperature Trend Deviation",
       ylabel="Readiness Score")

# 5. Save and show 
fig.savefig("Figures/Figure_11.png")


# In[2989]:


# 1. Prepare data
x = df["Previous Day Activity Score"]
y = df["Readiness Score"]

# 2. Setup plot
fig, ax = plt.subplots(figsize=(10, 10))

# 3. Plot data
ax.scatter(x,y)

# 4. Customize plot
ax.set(title="Previous Day Activity Score vs Readiness", 
       xlabel="Previous Day Activity Score",
       ylabel="Readiness Score")

# 5. Save and show 
fig.savefig("Figures/Figure_12.png")


# In[2990]:


st.header('Relevent Data')


# In[2991]:


st.image('Figures/Figure_1.png')
st.image('Figures/Figure_2.png')
st.image('Figures/Figure_5.png')
st.image('Figures/Figure_7.png')
st.image('Figures/Figure_10.png')
st.image('Figures/Figure_11.png')
st.image('Figures/Figure_12.png')


# ## Remove data columns that are lagging data fields or not necessary

# In[2992]:


df.drop(df.columns[[0,1,2,3,4,5,6,7,8,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,26,28,29,30,31,32,33,34,35,36,37,39,40,41,42,43,44,50,51,52,53]], axis=1, inplace=True)


# In[2993]:


df.info()


# ## Describe Relevent Data

# In[2994]:


st.header('Describe Relevent Data')
st.write(df.describe())
df.describe()


# ## Algorithm/Estimator

# In[2995]:


# Import algorithm/estimator

# Instantiate and fit the model (on the training set)
# Try RandomForest estimator


from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier()

# Setup random seed
np.random.seed(42)

# Create the data
df.dropna(inplace=True)
X = df.drop("Readiness Score", axis=1)
y = df["Readiness Score"] #target

# Split into train and test sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


# In[2996]:


model.get_params()


# In[2997]:


model.fit(X_train, y_train);


# In[2998]:


y_preds = model.predict(X_test)


# In[2999]:


model.score(X_test, y_test)


# In[3000]:


# Try Ridge Regression

from sklearn.linear_model import Ridge

model = Ridge()
model.fit(X_train, y_train)

# Check the score of the model (on the test set)
model.score(X_test, y_test)


# In[3001]:


# Try LassoLars

from sklearn import linear_model
model = linear_model.LassoLars(alpha=0.1)
model.fit(X_train, y_train)

# Check the score of the model (on the test set)
model.score(X_test, y_test) #Coefficient of determination of the prediction R^2


# In[3002]:


model.get_params()


# ## Make Predictions Using Machine Language Model

# ## Pick Typical Day (2/20/2023)

# In[3003]:


test_data = pd.read_csv("New Data/oura_2023-02-20_2023-02-20_trends.csv")


# In[3004]:


test_data.drop(test_data.columns[[0,1,2,3,4,5,6,7,8,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,26,28,29,30,31,32,33,34,35,36,37,39,40,41,42,43,44,50,51,52,53]], axis=1, inplace=True)


# In[3005]:


## Remove Readiness Score
test_data.drop(test_data.columns[4], axis=1, inplace=True) #Drop Readiness Score
test_data.info()


# In[3006]:


## Convert Total Sleep Duration to hours
test_data["Total Sleep Duration"] = test_data["Total Sleep Duration"] / 3600
test_data["Rest Time"] = test_data["Rest Time"] / 3600


# In[3007]:


st.header('Data for a Typical Day (2/20/2023)')


# In[3008]:


value = st.slider(
    'Select a estimated sleep',
    4.0, 12.0, 8.0)
st.write('Estimated Sleep:', value)
test_data["Total Sleep Duration"] = value
st.write(test_data)


# ## Prediction: 

# In[3009]:


st.header('Based on your estimated sleep, your readiness score for 2/21/23 is prediced to be: ')
Prediction = str(model.predict(test_data))
font_size = 50

html_str = f"""
<style>
p.a {{
  font: bold {font_size}px Courier;
}}
</style>
<p class="a">{Prediction}</p>
"""

st.markdown(html_str, unsafe_allow_html=True)
#value=model.predict(test_data)


# In[3010]:


