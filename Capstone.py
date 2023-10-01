#!/usr/bin/env python
# coding: utf-8

# In[251]:


import pandas as pd
##%matplotlib inline
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
from plotly import graph_objects
import streamlit as st


# In[252]:


##st.text('Fixed width text')
##st.markdown('_Markdown_') # see #*
##st.caption('Balloons. Hundreds of them...')
##st.latex(r''' e^{i\pi} + 1 = 0 ''')
##st.write('Most objects') # df, err, func, keras!
##st.write(['st', 'is <', 3]) # see *
st.title('Computer Science Capstone — C964')
st.header('Student – Mark Nefzger')
st.header('Student ID: 001411596')
##st.subheader('My sub')
##st.code('for i in range(8): foo()')


# In[253]:


print("Pandas version: ", pd.__version__)


# In[254]:


print("Numpy version: ", np.__version__)


# In[255]:


print("Streamlit version: ", st.__version__)


# In[256]:


print("Matplotlib version: ", matplotlib.__version__)


# ## Import data

# In[257]:


# Import Data
health_data = pd.read_csv("New Data/oura_2019-01-01_2023-09-09_trends_Original.csv")


# ## View data

# In[258]:


st.header('Imported Data')


# In[259]:


health_data


# ## Describe Data

# In[260]:


# Attribute
health_data.dtypes


# ## Set up dataframe

# In[261]:


df = pd.DataFrame(health_data)


# In[262]:


df.info()


# In[ ]:





# ## Convert Sleep Duration and Rest Time to hours

# In[263]:


df["Total Sleep Duration"] = df["Total Sleep Duration"] / 3600
df["Rest Time"] = df["Rest Time"] / 3600

df["Total Sleep Duration"], df["Rest Time"]


# In[264]:


pd.crosstab(df["Total Sleep Duration"] > 7, df["Readiness Score"] >85)


# In[265]:


df["Readiness Score"].hist(figsize=(10, 10))


# ## Manipulating Data

# In[266]:


df.dropna(inplace=True)


# In[267]:


df.info()


# In[268]:


# Randomize data 1 = 100%
df.sample(frac=1)


# In[269]:


# Reset index if necessary
# df.reset_index(drop=True, inplace=True)


# ## Matplotlib flow

# In[270]:


# 1. Prepare data
x = df["Total Sleep Duration"]
y = df["Readiness Score"]

# 2. Setup plot
fig, ax = plt.subplots(figsize=(10, 10))

# 3. Plot data
ax.scatter(x,y)

# 4. Customize plot
ax.set(title="Simple Plot", 
       xlabel="Total Sleep Duration",
       ylabel="Readiness Score")

# 5. Save and show (you save the whole figure)
# fig.savefig("C:/Users/McLovin/OneDrive/Desktop/Capstone/New Data/Images/Figure_1.png")


# In[271]:


# 1. Prepare data
x = df["Previous Night Score"]
y = df["Readiness Score"]

# 2. Setup plot
fig, ax = plt.subplots(figsize=(10, 10))

# 3. Plot data
ax.scatter(x,y)

# 4. Customize plot
ax.set(title="Simple Plot", 
       xlabel="Previous Night Score",
       ylabel="Readiness Score")

# 5. Save and show (you save the whole figure)
# fig.savefig("C:/Users/McLovin/OneDrive/Desktop/Capstone/New Data/Images/Figure_2.png")


# In[272]:


# 1. Prepare data
x = df["Move Every Hour Score"]
y = df["Readiness Score"]

# 2. Setup plot
fig, ax = plt.subplots(figsize=(10, 10))

# 3. Plot data
ax.scatter(x,y)

# 4. Customize plot
ax.set(title="Simple Plot", 
       xlabel="Move Every Hour Score",
       ylabel="Readiness Score")

# 5. Save and show (you save the whole figure)
# fig.savefig("C:/Users/McLovin/OneDrive/Desktop/Capstone/New Data/Images/Figure_1.png")


# In[273]:


# 1. Prepare data
x = df["Non-wear Time"]
y = df["Readiness Score"]

# 2. Setup plot
fig, ax = plt.subplots(figsize=(10, 10))

# 3. Plot data
ax.scatter(x,y)

# 4. Customize plot
ax.set(title="Simple Plot", 
       xlabel="Non-wear Time",
       ylabel="Readiness Score")

# 5. Save and show (you save the whole figure)
# fig.savefig("C:/Users/McLovin/OneDrive/Desktop/Capstone/New Data/Images/Figure_1.png")


# In[274]:


# 1. Prepare data
x = df["Rest Time"]
y = df["Readiness Score"]

# 2. Setup plot
fig, ax = plt.subplots(figsize=(10, 10))

# 3. Plot data
ax.scatter(x,y)

# 4. Customize plot
ax.set(title="Simple Plot", 
       xlabel="Rest Time",
       ylabel="Readiness Score")

# 5. Save and show (you save the whole figure)
# fig.savefig("C:/Users/McLovin/OneDrive/Desktop/Capstone/New Data/Images/Figure_1.png")


# In[275]:


# 1. Prepare data
x = df["Previous Day Activity Score"]
y = df["Readiness Score"]

# 2. Setup plot
fig, ax = plt.subplots(figsize=(10, 10))

# 3. Plot data
ax.scatter(x,y)

# 4. Customize plot
ax.set(title="Simple Plot", 
       xlabel="Previous Day Activity Score",
       ylabel="Readiness Score")

# 5. Save and show (you save the whole figure)
# fig.savefig("C:/Users/McLovin/OneDrive/Desktop/Capstone/New Data/Images/Figure_2.png")


# ## Remove data columns that are lagging data fields or not necessary

# In[276]:


df.drop(df.columns[[0,1,2,3,4,5,6,7,8,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,46,47,48,49,53]], axis=1, inplace=True)


# In[277]:


df.info()


# In[278]:


df.describe()


# In[279]:


# Average Readiness Score
df["Readiness Score"].mean()


# In[280]:


len(df)


# ## Algorithm/Estimator

# In[281]:


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


# In[282]:


model.get_params()


# In[283]:


model.fit(X_train, y_train);


# In[284]:


y_preds = model.predict(X_test)


# In[285]:


model.score(X_test, y_test)


# In[286]:


# Try Ridge Regression

from sklearn.linear_model import Ridge

model = Ridge()
model.fit(X_train, y_train)

# Check the score of the model (on the test set)
model.score(X_test, y_test)


# In[287]:


model.get_params()


# In[288]:


from sklearn import linear_model
model = linear_model.LassoLars(alpha=1.0)
model.fit(X_train, y_train)

# Check the score of the model (on the test set)
model.score(X_test, y_test)


# ## Make Predictions Using Machine Language Model

# In[289]:


test_data = pd.read_csv("New Data/oura_2023-09-17_2023-09-17_trends.csv")


# In[290]:


test_data.drop(test_data.columns[[0,1,2,3,4,5,6,7,8,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,46,47,48,49,53]], axis=1, inplace=True)
test_data.info()


# In[291]:


## Remove Readiness Score
test_data.drop(test_data.columns[2], axis=1, inplace=True)
test_data.info()


# In[292]:


test_data.info()


# In[293]:


## Convert Total Sleep Duration to hours
test_data["Total Sleep Duration"] = test_data["Total Sleep Duration"] / 3600


# In[294]:


value = st.slider(
    'Select a estimated sleep',
    4.0, 12.0, 8.0)
st.write('Estimated Sleep:', value)


# In[295]:


sleep_hours_pred = float(input("How many hours of planned sleep? "))
print(sleep_hours_pred)


# In[296]:


test_data


# In[297]:


test_data["Total Sleep Duration"] = sleep_hours_pred
test_data


# In[298]:


model.predict(test_data)


# In[299]:


