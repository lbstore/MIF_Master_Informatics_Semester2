
# coding: utf-8

# In[49]:


import pandas as pd
from sklearn.preprocessing import MinMaxScaler
# get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
import numpy as np
from sklearn.manifold import MDS
from sklearn.metrics import euclidean_distances
from sklearn.metrics import pairwise_distances


# In[50]:


baseDir = "E:/University/MIF_Master_Informatics_Semester2/Data Visualization/"

dataSet = pd.read_csv(baseDir+"spamFileRemovedResampledComma.csv")
print(dataSet.info())
dataSet = dataSet.drop(columns=["maxRepeatedWordCount","maxWordLength","lineCount","averageWordLengh","alphaCount","alphaNumCount","alphaWordCount","numberCount"])
dataSet = dataSet.sample(frac=51/155)

dataSet['spam'] = dataSet['spam'].astype('category');

what = dataSet.select_dtypes(['category']).columns

print(dataSet.info())
print(what)


numerate = {
    'spam' : {"SPAM":0,"NOTSPAM":1}
}
print("try conv")
dataSet = dataSet.replace(numerate)
# dataSet[what] = dataSet[what].apply(lambda l: conv(l))
print("after conv")
print(dataSet.info())
# data = pd.read_csv('parkinsons.data')
data = dataSet

# In[51]:


y = data['spam'].values


# In[52]:


x = data.loc[:, :'spam'].values


# In[77]:


x = MinMaxScaler().fit_transform(x)
colorize = dict(c=x[:, 0], cmap=plt.cm.get_cmap('rainbow', 5))
D = pairwise_distances(x)


# In[79]:

print("Random state 1")
model = MDS(n_components=2, dissimilarity='precomputed', random_state=1)
out = model.fit_transform(D)
plt.scatter(out[:, 0], out[:, 1], **colorize)
plt.axis('equal');
plt.show();

# In[81]:

print("Random state 2")
model = MDS(n_components=2, dissimilarity='precomputed', random_state=2)
out = model.fit_transform(D)
plt.scatter(out[:, 0], out[:, 1], **colorize)
plt.axis('equal');
plt.show();

print("Random state 5")
model = MDS(n_components=2, dissimilarity='precomputed', random_state=5)
out = model.fit_transform(D)
plt.scatter(out[:, 0], out[:, 1], **colorize)
plt.axis('equal');
plt.show();

# In[75]:


print("Manhattan")
D3 = pairwise_distances(x, metric="manhattan")
out3 = model.fit_transform(D3)
plt.scatter(out3[:, 0], out3[:, 1], **colorize)
plt.axis('equal');
plt.show();

# In[76]:

print("braycurtis")
D3 = pairwise_distances(x, metric="braycurtis")
out3 = model.fit_transform(D3)
plt.scatter(out3[:, 0], out3[:, 1], **colorize)
plt.axis('equal');
plt.show();

# In[60]:


D3


# In[ ]:




