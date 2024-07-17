#!/usr/bin/env python
# coding: utf-8

# # Decision Tree

# ##Importing Libraries

# In[30]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings(action = 'ignore')
#plt.style.use(['seaborn-bright', 'dark_background'])


# ## Importing dataset

# In[31]:


data = pd.read_csv('churn_prediction_simple.csv')
data.head()


# In[32]:


data.describe()


# In[33]:


data.info()


# In[34]:


#separating dependent and independent varibales
X = data.drop(columns = ['churn','customer_id'])
Y = data['churn']


# In[35]:


#Scaling the dataset
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaled_X = scaler.fit_transform(X)


# In[36]:


#splitting the dataset
from sklearn.model_selection import train_test_split as tts
x_train, x_test, y_train, y_test = tts(scaled_X, Y, train_size = 0.80, stratify = Y)
x_train.shape, x_test.shape, y_train.shape, y_test.shape


# ## Model Building, predictions

# In[37]:


y_train


# In[38]:


from sklearn.tree import DecisionTreeClassifier as DTC
classifier = DTC( class_weight = 'balanced')
classifier = DTC()


# In[39]:


classifier.fit(x_train, y_train)
predicted_values = classifier.predict(x_train)


# In[40]:


predicted_values[:30]


# ##Evaluation Metrics

# In[41]:


from sklearn.metrics import classification_report
print(classification_report(y_train, predicted_values))


# In[42]:


predicted_values = classifier.predict(x_test)
print(classification_report(y_test, predicted_values))


# # Visualising Decision Tree

# In[43]:


get_ipython().system('pip install graphviz')


# In[44]:


from sklearn.tree import export_graphviz
export_graphviz(decision_tree = classifier, out_file = 'tree_viz',
                max_depth=None, feature_names = X.columns ,
                label = None, impurity = False )


# In[45]:


from graphviz import render
render(  filepath='tree_viz', format = 'png', engine = 'neato')


# # Hyperparamter Tuning

# In[46]:


classifier = DTC()
classifier.fit(x_train, y_train)


# ## max_depth

# In[47]:


from sklearn.metrics import f1_score
def calc_score(model, x1, y1, x2, y2):

  model.fit(x1,y1)

  predict = model.predict(x1)
  f1 = f1_score(y1, predict)

  predict = model.predict(x2)
  f2 = f1_score(y2, predict)

  return f1, f2


# In[48]:


def effect(train_score, test_score, x_axis, title):
  plt.figure(figsize = (5,5), dpi = 120)
  plt.plot(x_axis, train_score, color = 'red', label = 'train_Score')
  plt.plot(x_axis, test_score, color = 'blue', label = 'test_Score')
  plt.title(title)
  plt.legend()
  plt.xlabel("parameter_value")
  plt.ylabel("f1 score")
  plt.show()


# In[49]:


maxdepth = [i for i in range(1,50)]
train = []
test = []

for i in maxdepth:  
  model = DTC(class_weight = 'balanced', max_depth = i, random_state = 42)
  f1, f2 = calc_score(model, x_train, y_train, x_test, y_test)
  train.append(f1)
  test.append(f2)


# In[50]:


effect( train, test, range(1,50) , 'max_depth')


# ## min_samples_split

# In[51]:


min_samples = [i for i in range(2,5000, 25)]
train = []
test = []

for i in min_samples:  
  model = DTC(class_weight = 'balanced', min_samples_split = i, random_state = 42)
  f1, f2 = calc_score(model, x_train, y_train, x_test, y_test)
  train.append(f1)
  test.append(f2)


# In[52]:


effect( train, test, range(2,5000, 25) , 'min_samples_split')


# ## max_leaf_nodes

# In[53]:


maxleafnodes = [i for i in range(2,200,10)]
train = []
test = []

for i in maxleafnodes:  
  model = DTC(class_weight = 'balanced', max_leaf_nodes = i, random_state = 42)
  f1, f2 = calc_score(model, x_train, y_train, x_test, y_test)
  train.append(f1)
  test.append(f2)


# In[54]:


effect( train, test, range(2,200,10) , 'max_leaf_nodes')


# ## min_samples_leaf

# In[55]:


minsamplesleaf = [i for i in range(2,4000,25)]
train = []
test = []

for i in minsamplesleaf:  
  model = DTC(class_weight = 'balanced', min_samples_leaf = i, random_state = 42)
  f1, f2 = calc_score(model, x_train, y_train, x_test, y_test)
  train.append(f1)
  test.append(f2)


# In[56]:


effect( train, test, range(2,4000,25) , 'min_samples_leaf')


# In[57]:


model = DTC(max_depth = 9)
model.fit(x_train, y_train)
feature_imp = pd.Series(model.feature_importances_, index = X.columns)
k = feature_imp.sort_values()


# In[58]:


plt.figure(figsize = (10,5), dpi = 120)
plt.barh(k.index, k)
plt.xlabel('Importance')
plt.ylabel('feature_name')
plt.title('feature importance')


# In[58]:




