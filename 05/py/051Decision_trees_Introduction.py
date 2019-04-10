
# coding: utf-8

# # Decision trees
# 
# 
# 

# In this section we cover the decision tree methods. We divided it into three parts:
# - univariate methods,
# - multivariate methods,
# - quality metrics.
# 
# In the first two sections we show two different approaches on building decision trees. The last section is about methods that allow to measure the quality and improve the tree by reducing the number of leafs (pruning).

# ## Idea
# 
# Decision trees are one of the methods that is transparent on the decision it took while prediction. It build a set of leafs at each level based on a rule. It is easy to understand, because it can be also written as a set of rules that are human readable. The tree can be a binary as below or can divide on each level and generate more than two leafs.
# 
# ![](images/tree.png)

# The customer segmentation shown in the lectures looks as following:
# 
# 
# | Location|Category   | Gender | Product review  | Decision |
# |---------|-----------|--------|-----------------|--------|
# | Berlin | Furniture  | Male   | Checked Reviews | True |
# | London | Furniture  | Male   | Checked Reviews | False |
# | Berlin | Furniture  | Female | Checked Reviews | False|
# | Berlin | Textile    | Female | Checked Reviews | True |
# | London | Electronics| Male   | Checked Reviews | False |
# | London | Textile    | Female | Checked Reviews | False |
# | Paris  | Textile    | Male   | Did not checked | True |
# | Berlin | Electronics| Male   | Checked Reviews | True |
# | Paris  | Electronics| Male   | Did not checked | True |
# | London | Electronics| Female | Checked Reviews | True |
# | Paris  | Furniture  | Female | Did not checked | True |
# | Berlin | Textile    | Female | Did not checked | True |
# | London | Electronics| Female | Did not checked | False |
# | London | Furniture  | Female | Checked Reviews | False |
# | London | Textile    | Female | Did not checked | False |
# 
# 
# The data above can be represented as two variables ``labels`` that is the last column and ``data_set`` that represents the other columns.

# In[1]:


import numpy

labels = [1,1,-1,1,-1,-1,1,1,1,1,1,1,-1,-1,-1]
data_set = [[1,1,2,2],[2,1,2,2],[1,1,1,2],[1,2,1,2],[2,3,2,2],
                [2,2,1,2],[3,2,2,1],[1,3,2,2],[3,3,2,1],[2,3,1,2],
                [3,1,1,1],[1,2,1,1],[2,3,1,1],[2,1,1,2],[2,2,1,1]]


# The data set can be visualized using pandas as shown below.

# In[2]:


import pandas as pd

pd.DataFrame(data_set, columns=['Location','Category','Gender','Product review'])


# In[3]:


get_ipython().run_line_magic('store', 'data_set')
get_ipython().run_line_magic('store', 'labels')

