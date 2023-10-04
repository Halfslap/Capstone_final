#!/usr/bin/env python
# coding: utf-8

# In[380]:


import pandas as pd
##%matplotlib inline
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
from plotly import graph_objects
import streamlit as st


# In[381]:


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


# In[382]:


pd_ver = pd.__version__
st.write("Pandas version: ", pd_ver)


# In[383]:


np_ver = np.__version__
st.write("Numpy version: ", np_ver)


# In[384]:


st_ver = st.__version__
st.write("Streamlit version: ", st_ver)


# In[385]:


plt_ver = matplotlib.__version__
st.write("Matplotlib version: ", plt_ver)


# ## Import data

# In[386]:


# Import Data
health_data = pd.read_csv("New Data/oura_2019-01-01_2023-09-09_trends_Original.csv")


# ## View data

# In[387]:


st.header('Imported Data')


# In[388]:


health_data


# ## Describe Data

# In[389]:


# health_data.dtypes;


# ## Set up dataframe

# In[390]:


df = pd.DataFrame(health_data)


# ## Convert Sleep Duration and Rest Time to hours

# In[391]:


df["Total Sleep Duration"] = df["Total Sleep Duration"] / 3600
df["Rest Time"] = df["Rest Time"] / 3600


# In[392]:


pd.crosstab(df["Total Sleep Duration"] > 7, df["Readiness Score"] >85)


# In[393]:


df["Readiness Score"].hist(figsize=(10, 10))


# ## Manipulating Data

# In[394]:


df.dropna(inplace=True)


# In[395]:


# Randomize data 1 = 100%
df.sample(frac=1)


# In[396]:


# Reset index if necessary
# df.reset_index(drop=True, inplace=True)


# ## Matplotlib

# In[397]:


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
fig.savefig("C:/Users/McLovin/miniconda3/envs/spyder_env/Capstone/Capstone/Figures/Figure_1.png")


# In[398]:


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
fig.savefig("C:/Users/McLovin/miniconda3/envs/spyder_env/Capstone/Capstone/Figures/Figure_2.png")


# In[399]:


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
fig.savefig("C:/Users/McLovin/miniconda3/envs/spyder_env/Capstone/Capstone/Figures/Figure_3.png")


# In[400]:


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
fig.savefig("C:/Users/McLovin/miniconda3/envs/spyder_env/Capstone/Capstone/Figures/Figure_4.png")


# In[401]:


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
fig.savefig("C:/Users/McLovin/miniconda3/envs/spyder_env/Capstone/Capstone/Figures/Figure_5.png")


# In[402]:


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
fig.savefig("C:/Users/McLovin/miniconda3/envs/spyder_env/Capstone/Capstone/Figures/Figure_6.png")


# In[403]:


# 1. Prepare data
x = df["Activity Score"]
y = df["Readiness Score"]

# 2. Setup plot
fig, ax = plt.subplots(figsize=(10, 10))

# 3. Plot data
ax.scatter(x,y)

# 4. Customize plot
ax.set(title="Simple Plot", 
       xlabel="Activity Score",
       ylabel="Readiness Score")

# 5. Save and show (you save the whole figure)
fig.savefig("C:/Users/McLovin/miniconda3/envs/spyder_env/Capstone/Capstone/Figures/Figure_7.png")


# In[404]:


# 1. Prepare data
x = df["Resting Heart Rate Score"]
y = df["Readiness Score"]

# 2. Setup plot
fig, ax = plt.subplots(figsize=(10, 10))

# 3. Plot data
ax.scatter(x,y)

# 4. Customize plot
ax.set(title="Simple Plot", 
       xlabel="Resting Heart Rate Score",
       ylabel="Readiness Score")

# 5. Save and show (you save the whole figure)
fig.savefig("C:/Users/McLovin/miniconda3/envs/spyder_env/Capstone/Capstone/Figures/Figure_8.png")


# In[405]:


# 1. Prepare data
x = df["Temperature Score"]
y = df["Readiness Score"]

# 2. Setup plot
fig, ax = plt.subplots(figsize=(10, 10))

# 3. Plot data
ax.scatter(x,y)

# 4. Customize plot
ax.set(title="Simple Plot", 
       xlabel="Temperature Score",
       ylabel="Readiness Score")

# 5. Save and show (you save the whole figure)
fig.savefig("C:/Users/McLovin/miniconda3/envs/spyder_env/Capstone/Capstone/Figures/Figure_9.png")


# In[406]:


# 1. Prepare data
x = df["HRV Balance Score"]
y = df["Readiness Score"]

# 2. Setup plot
fig, ax = plt.subplots(figsize=(10, 10))

# 3. Plot data
ax.scatter(x,y)

# 4. Customize plot
ax.set(title="Simple Plot", 
       xlabel="HRV Balance Score",
       ylabel="Readiness Score")

# 5. Save and show (you save the whole figure)
fig.savefig("C:/Users/McLovin/miniconda3/envs/spyder_env/Capstone/Capstone/Figures/Figure_10.png")


# In[407]:


#st.image('Figures/Figure_1.png')
#st.image('Figures/Figure_8.png')
#st.image('Figures/Figure_10.png')


# ## Remove data columns that are lagging data fields or not necessary

# In[408]:


df.drop(df.columns[[0,1,2,3,4,5,6,7,8,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,46,47,48,49,53]], axis=1, inplace=True)


# In[409]:


df.info()


# In[410]:


df.describe()


# In[411]:


# Average Readiness Score
df["Readiness Score"].mean()


# In[412]:


len(df)


# ## Algorithm/Estimator

# In[413]:


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


# In[414]:


model.get_params()


# In[415]:


model.fit(X_train, y_train);


# In[416]:


y_preds = model.predict(X_test)


# In[417]:


model.score(X_test, y_test)


# In[418]:


# Try Ridge Regression

from sklearn.linear_model import Ridge

model = Ridge()
model.fit(X_train, y_train)

# Check the score of the model (on the test set)
model.score(X_test, y_test)


# In[419]:


model.get_params()


# In[420]:


from sklearn import linear_model
model = linear_model.LassoLars(alpha=1.0)
model.fit(X_train, y_train)

# Check the score of the model (on the test set)
model.score(X_test, y_test)


# ## Make Predictions Using Machine Language Model

# In[421]:


test_data = pd.read_csv("New Data/oura_2023-09-17_2023-09-17_trends.csv")


# In[422]:


test_data.drop(test_data.columns[[0,1,2,3,4,5,6,7,8,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,46,47,48,49,53]], axis=1, inplace=True)
test_data.info()


# In[423]:


## Remove Readiness Score
test_data.drop(test_data.columns[2], axis=1, inplace=True)
test_data.info()


# In[424]:


test_data.info()


# In[425]:


## Convert Total Sleep Duration to hours
test_data["Total Sleep Duration"] = test_data["Total Sleep Duration"] / 3600


# In[426]:


value = st.slider(
    'Select a estimated sleep',
    4.0, 12.0, 8.0)
st.write('Estimated Sleep:', value)
test_data["Total Sleep Duration"] = value


# In[427]:


st.write('Based on your estimated sleep, your readiness score is prediced to be: ', model.predict(test_data))


# In[428]:


sleep_hours_pred = float(input("How many hours of planned sleep? "))
print(sleep_hours_pred)


# In[429]:


test_data


# In[430]:


test_data["Total Sleep Duration"] = sleep_hours_pred
test_data


# In[431]:


model.predict(test_data)


# In[ ]:


