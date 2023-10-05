#!/usr/bin/env python
# coding: utf-8

# In[688]:


import pandas as pd
##%matplotlib inline
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
from plotly import graph_objects
import streamlit as st


# ## My Info

# In[689]:


st.title('Computer Science Capstone — C964')
st.header('Student – Mark Nefzger')
st.header('Student ID: 001411596')


# ## Version Info

# In[690]:


pd_ver = pd.__version__
st.write("Pandas version: ", pd_ver)


# In[691]:


np_ver = np.__version__
st.write("Numpy version: ", np_ver)


# In[692]:


st_ver = st.__version__
st.write("Streamlit version: ", st_ver)


# In[693]:


plt_ver = matplotlib.__version__
st.write("Matplotlib version: ", plt_ver)


# ## Import data

# In[694]:


# Import Data
health_data = pd.read_csv("New Data/oura_2019-01-01_2023-09-09_trends_Original.csv")


# ## View data

# In[695]:


st.header('Imported Data')


# In[696]:


health_data


# ## Describe Data

# In[697]:


# health_data.dtypes


# ## Set up dataframe

# In[698]:


df = pd.DataFrame(health_data)


# ## Convert Sleep Duration and Rest Time to hours

# In[699]:


df["Total Sleep Duration"] = df["Total Sleep Duration"] / 3600
df["Rest Time"] = df["Rest Time"] / 3600


# In[700]:


pd.crosstab(df["Total Sleep Duration"] > 7, df["Readiness Score"] >85)


# In[701]:


df["Readiness Score"].hist(figsize=(10, 10))


# ## Manipulating Data

# In[702]:


df.dropna(inplace=True)


# In[703]:


# Randomize data 1 = 100%
df.sample(frac=1)


# In[704]:


# Reset index if necessary
# df.reset_index(drop=True, inplace=True)


# ## Matplotlib

# In[705]:


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

# 5. Save and show (you save the whole figure)
fig.savefig("Figures/Figure_1.png")


# In[706]:


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

# 5. Save and show (you save the whole figure)
fig.savefig("Figures/Figure_2.png")


# In[707]:


# 1. Prepare data
x = df["Move Every Hour Score"]
y = df["Readiness Score"]

# 2. Setup plot
fig, ax = plt.subplots(figsize=(10, 10))

# 3. Plot data
ax.scatter(x,y)

# 4. Customize plot
ax.set(title="Move Every Hour vs Readiness", 
       xlabel="Move Every Hour Score",
       ylabel="Readiness Score")

# 5. Save and show (you save the whole figure)
fig.savefig("Figures/Figure_3.png")


# In[708]:


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

# 5. Save and show (you save the whole figure)
fig.savefig("Figures/Figure_4.png")


# In[709]:


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

# 5. Save and show (you save the whole figure)
fig.savefig("Figures/Figure_5.png")


# In[710]:


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

# 5. Save and show (you save the whole figure)
fig.savefig("Figures/Figure_6.png")


# In[711]:


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

# 5. Save and show (you save the whole figure)
fig.savefig("Figures/Figure_7.png")


# In[712]:


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

# 5. Save and show (you save the whole figure)
fig.savefig("Figures/Figure_8.png")


# In[713]:


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

# 5. Save and show (you save the whole figure)
fig.savefig("Figures/Figure_9.png")


# In[714]:


# 1. Prepare data
x = df["HRV Balance Score"]
y = df["Readiness Score"]

# 2. Setup plot
fig, ax = plt.subplots(figsize=(10, 10))

# 3. Plot data
ax.scatter(x,y)

# 4. Customize plot
ax.set(title="HRV Balance vs Readiness", 
       xlabel="HRV Balance Score",
       ylabel="Readiness Score")

# 5. Save and show (you save the whole figure)
fig.savefig("Figures/Figure_10.png")


# In[715]:


st.header('Relevent Data')


# In[716]:


st.image('Figures/Figure_1.png')
st.image('Figures/Figure_7.png')
st.image('Figures/Figure_8.png')
st.image('Figures/Figure_9.png')
st.image('Figures/Figure_10.png')


# ## Remove data columns that are lagging data fields or not necessary

# In[717]:


df.drop(df.columns[[0,1,2,3,4,5,6,7,8,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,46,47,48,49,53]], axis=1, inplace=True)


# In[718]:


df.info()


# ## Describe Relevent Data

# In[719]:


st.header('Describe Relevent Data')
st.write(df.describe())
df.describe()


# ## Algorithm/Estimator

# In[720]:


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


# In[721]:


model.get_params()


# In[722]:


model.fit(X_train, y_train);


# In[723]:


y_preds = model.predict(X_test)


# In[724]:


model.score(X_test, y_test)


# In[725]:


# Try Ridge Regression

from sklearn.linear_model import Ridge

model = Ridge()
model.fit(X_train, y_train)

# Check the score of the model (on the test set)
model.score(X_test, y_test)


# In[726]:


model.get_params()


# In[727]:


from sklearn import linear_model
model = linear_model.LassoLars(alpha=1.0)
model.fit(X_train, y_train)

# Check the score of the model (on the test set)
model.score(X_test, y_test)


# ## Make Predictions Using Machine Language Model

# ## Pick Typical Day (9/17/2023)

# In[728]:


test_data = pd.read_csv("New Data/oura_2023-09-17_2023-09-17_trends.csv")


# In[729]:


test_data.drop(test_data.columns[[0,1,2,3,4,5,6,7,8,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,46,47,48,49,53]], axis=1, inplace=True)


# In[730]:


## Remove Readiness Score
test_data.drop(test_data.columns[2], axis=1, inplace=True)
test_data.info()


# In[731]:


## Convert Total Sleep Duration to hours
test_data["Total Sleep Duration"] = test_data["Total Sleep Duration"] / 3600


# In[732]:


st.header('Data for a Typical Day (9/17/2023)')
st.write(test_data)


# In[733]:


value = st.slider(
    'Select a estimated sleep',
    4.0, 12.0, 8.0)
st.write('Estimated Sleep:', value)
test_data["Total Sleep Duration"] = value


# In[734]:


st.header('Based on your estimated sleep, your readiness score for 9/18/23 is prediced to be: ')
variable_output = st.text_input("Prediction: ", value=model.predict(test_data))
font_size = 50

html_str = f"""
<style>
p.a {{
  font: bold {font_size}px Courier;
}}
</style>
<p class="a">{variable_output}</p>
"""

st.markdown(html_str, unsafe_allow_html=True)


# In[735]:


