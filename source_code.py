#!/usr/bin/env python
# coding: utf-8

# # Linear Regression with Scikit Learn - Machine Learning with Python
# 

# ### How to run the code
# 
# This tutorial is an executable [Jupyter notebook](https://jupyter.org) hosted on [Jovian](https://www.jovian.ai). You can _run_ this tutorial and experiment with the code examples in a couple of ways: *using free online resources* (recommended) or *on 
# your computer*.
# 
# #### Option 1: Running using free online resources (1-click, recommended)
# 
# The easiest way to start executing the code is to click the **Run** button at the top of this page and select **Run on Binder**. You can also select "Run on Colab" or "Run on Kaggle", but you'll need to create an account on [Google Colab](https://colab.research.google.com) or [Kaggle](https://kaggle.com) to use these platforms.
# 
# 
# #### Option 2: Running on your computer locally
# 
# To run the code on your computer locally, you'll need to set up [Python](https://www.python.org), download the notebook and install the required libraries. We recommend using the [Conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/) distribution of Python. Click the **Run** button at the top of this page, select the **Run Locally** option, and follow the instructions.
# 
# >  **Jupyter Notebooks**: This tutorial is a [Jupyter notebook](https://jupyter.org) - a document made of _cells_. Each cell can contain code written in Python or explanations in plain English. You can execute code cells and view the results, e.g., numbers, messages, graphs, tables, files, etc., instantly within the notebook. Jupyter is a powerful platform for experimentation and analysis. Don't be afraid to mess around with the code & break things - you'll learn a lot by encountering and fixing errors. You can use the "Kernel > Restart & Clear Output" menu option to clear all outputs and start again from the top.

# ## Scope Of Work
# 
# #### Creating an automated system to estimate the annual medical expenditure for new customers**, using information such as their age, sex, BMI, children, smoking habits and region of residence. 
# >
# > Estimates from system can be used to determine the annual insurance premium (amount paid every month) offered to the customer.
# > 
# > Data Source [CSV file](https://raw.githubusercontent.com/JovianML/opendatasets/master/data/medical-charges.csv) containing verified historical data and information about the actual medical charges incurred by over 1300 customers. 
# > Dataset source: https://github.com/stedy/Machine-Learning-with-R-datasets

# ## Downloading the Data
# 
# To begin, let's download the data using the `urlretrieve` function from `urllib.request`.

# In[ ]:


#restart the kernel after installation
get_ipython().system('pip install pandas-profiling --quiet')
# pandas profiling is a adv lib used to profile data for the first look before diving into any analysis.
#-- quiet makes it silent.
import pandas_profiling


# In[ ]:


medical_charges_url = 'https://raw.githubusercontent.com/JovianML/opendatasets/master/data/medical-charges.csv'


# In[ ]:


from urllib.request import urlretrieve


# In[ ]:


urlretrieve(medical_charges_url, 'medical.csv')


# We can now create a Pandas dataframe using the downloaded file, to view and analyze the data.

# In[ ]:


import pandas as pd


# In[ ]:


from pandas_profiling import ProfileReport
profile = ProfileReport(medical_df, title="Pandas Profiling Report")
profile


# In[ ]:


medical_df = pd.read_csv('medical.csv')


# In[ ]:


medical_df


# The dataset contains 1338 rows and 7 columns. Each row of the dataset contains information about one customer. 
# 
# Our objective is to find a way to estimate the value in the "charges" column using the values in the other columns. If we can do so for the historical data, then we should able to estimate charges for new customers too, simply by asking for information like their age, sex, BMI, no. of children, smoking habits and region.
# 
# Let's check the data type for each column.

# In[ ]:


medical_df.info()


# Looks like "age", "children", "bmi" ([body mass index](https://en.wikipedia.org/wiki/Body_mass_index)) and "charges" are numbers, whereas "sex", "smoker" and "region" are strings (possibly categories). None of the columns contain any missing values, which saves us a fair bit of work!
# 
# Here are some statistics for the numerical columns:

# In[ ]:


medical_df.describe()


# The ranges of values in the numerical columns seem reasonable too (no negative ages!), so we may not have to do much data cleaning or correction. The "charges" column seems to be significantly skewed however, as the median (50 percentile) is much lower than the maximum value.
# 
# 
# Imp. inferences about the data->>
# There isn't much variation is No. of children column, It is mostly bounded between 1 and 2.
# 
# 

# In[ ]:


get_ipython().system('pip install jovian --quiet')


# In[ ]:


import jovian


# In[ ]:


jovian.commit() --quiet


# ## Exploratory Analysis and Visualization
# 
# Let's explore the data by visualizing the distribution of values in some columns of the dataset, and the relationships between "charges" and other columns.
# 
# We'll use libraries Matplotlib, Seaborn and Plotly for visualization.

# In[ ]:


get_ipython().system('pip install plotly matplotlib seaborn --quiet')


# In[ ]:


import plotly.express as px
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# The following settings will improve the default style and font sizes for our charts.

# In[ ]:


sns.set_style('darkgrid')
matplotlib.rcParams['font.size'] = 14
matplotlib.rcParams['figure.figsize'] = (10, 6)
matplotlib.rcParams['figure.facecolor'] = '#00000000'


# ### Age
# 
# Age is a numeric column. The minimum age in the dataset is 18 and the maximum age is 64. Thus, we can visualize the distribution of age using a histogram with 47 bins (one for each year) and a box plot. We'll use plotly to make the chart interactive, but you can create similar charts using Seaborn.

# In[ ]:


medical_df.age.describe()


# In[ ]:


fig = px.histogram(medical_df, 
                   x='age', 
                   marginal='box', 
                   nbins=47, 
                   title='Distribution of Age')
fig.update_layout(bargap=0.1)
fig.show()


# The distribution of ages in the dataset is almost uniform, with 20-30 customers at every age, except for the ages 18 and 19, which seem to have over twice as many customers as other ages. The uniform distribution might arise from the fact that there isn't a big variation in the number of people of any given age.
# 
# 
# 

# ### Body Mass Index
# 
# Let's look at the distribution of BMI (Body Mass Index) of customers, using a histogram and box plot.

# In[ ]:


fig = px.histogram(medical_df, 
                   x='bmi', 
                   marginal='box', 
                   color_discrete_sequence=['red'], 
                   title='Distribution of BMI (Body Mass Index)')
fig.update_layout(bargap=0.1)
fig.show()


# The measurements of body mass index seem to form a Gaussian distribution centered around the value 30, with a few outliers towards the right.

# ### Charges
# 
# Let's visualize the distribution of "charges" i.e. the annual medical charges for customers. This is the column we're trying to predict. Let's also use the categorical column "smoker" to distinguish the charges for smokers and non-smokers.

# In[ ]:


fig = px.histogram(medical_df, 
                   x='charges', 
                   marginal='box', 
                   color='smoker', 
                   color_discrete_sequence=['green', 'grey'], 
                   title='Annual Medical Charges')
fig.update_layout(bargap=0.1)
fig.show()


# We can make the following observations from the above graph:
# 
# * For most customers, the annual medical charges are under \\$10,000. Only a small fraction of customer have higher medical expenses, possibly due to accidents, major illnesses and genetic diseases. The distribution follows a "power law"
# * There is a significant difference in medical expenses between smokers and non-smokers. While the median for non-smokers is \\$7300, the median for smokers is close to \\$35,000.

# In[ ]:


medical_df[['sex','region','charges']]


# In[ ]:


fig = px.histogram(medical_df, 
                   x='charges', 
                   marginal='box', 
                   color='sex', 
                   color_discrete_sequence=['green', 'grey'], 
                   title='Annual Medical Charges')
fig.update_layout(bargap=0.1)
fig.show()


# In[ ]:


fig = px.histogram(medical_df, 
                   x='charges', 
                   marginal='box', 
                   color='region', 
                   color_discrete_sequence=['green', 'grey','yellow','red'], 
                   title='Annual Medical Charges')
fig.update_layout(bargap=0.1)
fig.show()


# ### Smoker
# 
# Let's visualize the distribution of the "smoker" column (containing values "yes" and "no") using a histogram.

# In[ ]:


medical_df.smoker.value_counts()


# In[ ]:


px.histogram(medical_df, x='smoker', color='sex', title='Smoker')


# It appears that 20% of customers have reported that they smoke. This matches the national average.
# 
# 

# In[ ]:


px.histogram(medical_df, x='region', color='smoker', title='Smoker')


# In[ ]:


fig = px.histogram(medical_df, 
                   x='sex', 
                   color='region', 
                   color_discrete_sequence=['green', 'grey','yellow','red'], 
                   title='Annual Medical Charges')
fig.update_layout(bargap=0.1)
fig.show()


# Having looked at individual columns, we can now visualize the relationship between "charges" (the value we wish to predict) and other columns.
# 
# ### Age and Charges
# 
# Let's visualize the relationship between "age" and "charges" using a scatter plot. Each point in the scatter plot represents one customer. We'll also use values in the "smoker" column to color the points.

# In[ ]:


fig = px.scatter(medical_df, 
                 x='age', 
                 y='charges', 
                 color='smoker', 
                 opacity=0.8, 
                 hover_data=['sex'], 
                 title='Age vs. Charges')
fig.update_traces(marker_size=5)
fig.show()


# We can make the following observations from the above chart:
# 
# * The general trend seems to be that medical charges increase with age, as we might expect. However, there is significant variation at every age, and it's clear that age alone cannot be used to accurately determine medical charges.
# 
# 
# * We can see three "clusters" of points, each of which seems to form a line with an increasing slope:
# 
#      1. The first and the largest cluster consists primary of presumably "healthy non-smokers" who have relatively low medical charges compared to others
#      
#      2. The second cluster contains a mix of smokers and non-smokers. It's possible that these are actually two distinct but overlapping clusters: "non-smokers with medical issues" and "smokers without major medical issues".
#      
#      3. The final cluster consists exclusively of smokers, presumably smokers with major medical issues that are possibly related to or worsened by smoking.

# ### BMI and Charges
# 
# Let's visualize the relationship between BMI (body mass index) and charges using another scatter plot. Once again, we'll use the values from the "smoker" column to color the points.

# In[ ]:


fig = px.scatter(medical_df, 
                 x='bmi', 
                 y='charges', 
                 color='smoker', 
                 opacity=0.8, 
                 hover_data=['sex'], 
                 title='BMI vs. Charges')
fig.update_traces(marker_size=5)
fig.show()


# It appears that for non-smokers, an increase in BMI doesn't seem to be related to an increase in medical charges. However, medical charges seem to be significantly higher for smokers with a BMI greater than 30.

# 
# 

# In[ ]:


df = px.data.tips()
fig = px.violin(medical_df, y="charges", x="region", color="smoker", box=True, points="all",
          hover_data=medical_df.columns)
fig.show()


# ### Correlation
# 
# As you can tell from the analysis, the values in some columns are more closely related to the values in "charges" compared to other columns. E.g. "age" and "charges" seem to grow together, whereas "bmi" and "charges" don't.

# In[ ]:


medical_df.charges.corr(medical_df.age)


# In[ ]:


medical_df.charges.corr(medical_df.bmi)


# To compute the correlation for categorical columns, they must first be converted into numeric columns.

# In[ ]:


smoker_values = {'no': 0, 'yes': 1}
smoker_numeric = medical_df.smoker.map(smoker_values)
medical_df.charges.corr(smoker_numeric)


# In[ ]:


medical_df.corr()


# In[ ]:


sns.heatmap(medical_df.corr(), cmap='Reds', annot=True)
plt.title('Correlation Matrix');


# In[ ]:


jovian.commit()


# ## Linear Regression using a Single Feature
# 
# We now know that the "smoker" and "age" columns have the strongest correlation with "charges". Let's try to find a way of estimating the value of "charges" using the value of "age" for non-smokers. First, let's create a data frame containing just the data for non-smokers.

# In[ ]:


non_smoker_df = medical_df[medical_df.smoker == 'no']


# Next, let's visualize the relationship between "age" and "charges"

# In[ ]:


plt.title('Age vs. Charges')
sns.scatterplot(data=non_smoker_df, x='age', y='charges', alpha=0.7, s=15);


# Apart from a few exceptions, the points seem to form a line. We'll try and "fit" a line using this points, and use the line to predict charges for a given age. A line on the X&Y coordinates has the following formula:
# 
# $y = wx + b$
# 

# 
# 
# We can use a library like `scikit-learn` perform liner regression.

# In[ ]:


get_ipython().system('pip install scikit-learn --quiet')


# Let's use the `LinearRegression` class from `scikit-learn` to find the best fit line for "age" vs. "charges" using the ordinary least squares optimization technique.

# In[ ]:


from sklearn.linear_model import LinearRegression


# In[ ]:


model = LinearRegression()


# Next, we can use the `fit` method of the model to find the best fit line for the inputs and targets.

# In[ ]:


inputs = non_smoker_df[['age']]
targets = non_smoker_df.charges
print('inputs.shape :', inputs.shape)
print('targes.shape :', targets.shape)


# In[ ]:


model.fit(inputs, targets)


# We can now make predictions using the model. Let's try predicting the charges for the ages 23, 37 and 61

# In[ ]:


model.predict(np.array([[23], 
                        [37], 
                        [61]]))


# In[ ]:


predictions = model.predict(inputs)


# In[ ]:


predictions


# Let's compute the RMSE(Root mean square error) loss to evaluate the model.

# In[ ]:


get_ipython().system('pip install numpy --quiet')
import numpy as np
def rmse(targets, predictions):
    return np.sqrt(np.mean(np.square(targets - predictions)))


# In[ ]:


rmse(targets, predictions)


# Seems like our prediction is off by $4000 on average, which is not too bad considering the fact that there are several outliers.

# In[ ]:


jovian.commit()


# ## Linear Regression using Multiple Features
# 
# So far, we've used on the "age" feature to estimate "charges". Adding another feature like "bmi" is fairly straightforward. We simply assume the following relationship:
# 
# $charges = w_1 \times age + w_2 \times bmi + b$
# 
# We need to change just one line of code to include the BMI.

# In[ ]:


# Create inputs and targets
inputs, targets = non_smoker_df[['age', 'bmi']], non_smoker_df['charges']

# Create and train the model
model = LinearRegression().fit(inputs, targets)

# Generate predictions
predictions = model.predict(inputs)

# Compute loss to evalute the model
loss = rmse(targets, predictions)
print('Loss:', loss)


# Adding the BMI doesn't seem to reduce the loss by much, as the BMI has a very weak correlation with charges, especially for non smokers.

# In[ ]:


non_smoker_df.charges.corr(non_smoker_df.bmi)


# In[ ]:


fig = px.scatter(non_smoker_df, x='bmi', y='charges', title='BMI vs. Charges')
fig.update_traces(marker_size=5)
fig.show()


# We can also visualize the relationship between all 3 variables "age", "bmi" and "charges" using a 3D scatter plot.

# In[ ]:


fig = px.scatter_3d(non_smoker_df, x='age', y='bmi', z='charges')
fig.update_traces(marker_size=3, marker_opacity=0.5)
fig.show()


# In[ ]:


model.coef_, model.intercept_


# Clearly, BMI has a much lower weightage, and you can see why. It has a tiny contribution, and even that is probably accidental.

# Let's go one step further, and add the final numeric column: "children", which seems to have some correlation with "charges".
# 
# $charges = w_1 \times age + w_2 \times bmi + w_3 \times children + b$

# In[ ]:


non_smoker_df.charges.corr(non_smoker_df.children)


# In[ ]:


fig = px.strip(non_smoker_df, x='children', y='charges', title= "Children vs. Charges")
fig.update_traces(marker_size=4, marker_opacity=0.7)
fig.show()


# In[ ]:


# Create inputs and targets
inputs, targets = non_smoker_df[['age', 'bmi', 'children']], non_smoker_df['charges']

# Create and train the model
model = LinearRegression().fit(inputs, targets)

# Generate predictions
predictions = model.predict(inputs)

# Compute loss to evalute the model
loss = rmse(targets, predictions)
print('Loss:', loss)


# Once again, we don't see a big reduction in the loss, even though it's greater than in the case of BMI.

# In[ ]:


jovian.commit()


# ## Binary Categories
# 
# The "smoker" category has just two values "yes" and "no". Let's create a new column "smoker_code" containing 0 for "no" and 1 for "yes".
# 

# In[ ]:


sns.barplot(data=medical_df, x='smoker', y='charges');


# In[ ]:


smoker_codes = {'no': 0, 'yes': 1}
medical_df['smoker_code'] = medical_df.smoker.map(smoker_codes)


# In[ ]:


medical_df.charges.corr(medical_df.smoker_code)


# In[ ]:


medical_df


# We can now use the `smoker_df` column for linear regression.
# 
# $charges = w_1 \times age + w_2 \times bmi + w_3 \times children + w_4 \times smoker + b$

# In[ ]:


# Create inputs and targets
inputs, targets = medical_df[['age', 'bmi', 'children', 'smoker_code']], medical_df['charges']

# Create and train the model
model = LinearRegression().fit(inputs, targets)

# Generate predictions
predictions = model.predict(inputs)

# Compute loss to evalute the model
loss = rmse(targets, predictions)
print('Loss:', loss)


# The loss reduces from `11355` to `6056`, almost by 50%! This is an important lesson: never ignore categorical data.
# 
# 
# Let's try adding the "sex" column as well.
# 
# $charges = w_1 \times age + w_2 \times bmi + w_3 \times children + w_4 \times smoker + w_5 \times sex + b$

# In[ ]:


sns.barplot(data=medical_df, x='sex', y='charges')


# In[ ]:


sex_codes = {'female': 0, 'male': 1}


# In[ ]:


medical_df['sex_code'] = medical_df.sex.map(sex_codes)


# In[ ]:


medical_df.charges.corr(medical_df.sex_code)


# In[ ]:


# Create inputs and targets
inputs, targets = medical_df[['age', 'bmi', 'children', 'smoker_code', 'sex_code']], medical_df['charges']

# Create and train the model
model = LinearRegression().fit(inputs, targets)

# Generate predictions
predictions = model.predict(inputs)

# Compute loss to evalute the model
loss = rmse(targets, predictions)
print('Loss:', loss)


# As you might expect, this does have a significant impact on the loss.

# In[ ]:


sns.barplot(data=medical_df, x='region', y='charges');


# In[ ]:


from sklearn import preprocessing
enc = preprocessing.OneHotEncoder()
enc.fit(medical_df[['region']])
enc.categories_


# In[ ]:


one_hot = enc.transform(medical_df[['region']]).toarray()
one_hot


# In[ ]:


medical_df[['northeast', 'northwest', 'southeast', 'southwest']] = one_hot


# In[ ]:


medical_df


# Let's include the region columns into our linear regression model.
# 
# $charges = w_1 \times age + w_2 \times bmi + w_3 \times children + w_4 \times smoker + w_5 \times sex + w_6 \times region + b$

# In[ ]:


# Create inputs and targets
input_cols = ['age', 'bmi', 'children', 'smoker_code', 'sex_code', 'northeast', 'northwest', 'southeast', 'southwest']
inputs, targets = medical_df[input_cols], medical_df['charges']

# Create and train the model
model = LinearRegression().fit(inputs, targets)

# Generate predictions
predictions = model.predict(inputs)

# Compute loss to evalute the model
loss = rmse(targets, predictions)
print('Loss:', loss)


# In[ ]:


jovian.commit()


# ## Model Improvements
# 
# Let's discuss and apply some more improvements to our model.
# 
# ### Feature Scaling
# 
# $charges = w_1 \times age + w_2 \times bmi + w_3 \times children + w_4 \times smoker + w_5 \times sex + w_6 \times region + b$
# 
# To compare the importance of each feature in the model, our first instinct might be to compare their weights.
# 
# Because different columns have different ranges, we run into two issues:
# 
# 1. We can't compare the weights of different column to identify which features are important
# 2. A column with a larger range of inputs may disproportionately affect the loss and dominate the optimization process.
# 
# For this reason, it's common practice to scale (or standardize) the values in numeric column by subtracting the mean and dividing by the standard deviation.
# 

# In[ ]:


model.coef_


# In[ ]:


model.intercept_


# In[ ]:


weights_df = pd.DataFrame({
    'feature': np.append(input_cols, 1),
    'weight': np.append(model.coef_, model.intercept_)
})
weights_df


# In[ ]:


###We can apply scaling using the StandardScaler class from scikit-learn.


# In[ ]:


medical_df
from sklearn.preprocessing import StandardScaler
numeric_cols = ['age', 'bmi', 'children'] 
scaler = StandardScaler()
scaler.fit(medical_df[numeric_cols])
scaler.mean_
scaler.var_


# In[ ]:


scaled_inputs = scaler.transform(medical_df[numeric_cols])
scaled_inputs


# In[ ]:


cat_cols = ['smoker_code', 'sex_code', 'northeast', 'northwest', 'southeast', 'southwest']
categorical_data = medical_df[cat_cols].values


# In[ ]:


inputs = np.concatenate((scaled_inputs, categorical_data), axis=1)
targets = medical_df.charges

# Create and train the model
model = LinearRegression().fit(inputs, targets)

# Generate predictions
predictions = model.predict(inputs)

# Compute loss to evalute the model
loss = rmse(targets, predictions)
print('Loss:', loss)


# ## We can now compare the weights in the formula:
# 
# ##### \(charges = w_1 \times age + w_2 \times bmi + w_3 \times children + w_4 \times smoker + w_5 \times sex + w_6 \times region + b\)

# In[ ]:


weights_df = pd.DataFrame({
    'feature': np.append(numeric_cols + cat_cols, 1),
    'weight': np.append(model.coef_, model.intercept_)
})
weights_df.sort_values('weight', ascending=False)


# ### As you can see now, the most important feature are:
# 
# 1.Smoker
# 2.Age
# 3.BMI

# In[ ]:




