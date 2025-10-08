#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


# In[3]:


df = pd.read_csv('wine.csv')
df


# #### Separate features and target variable

# In[4]:


X = df.drop(columns=['Wine'])
y = df['Wine']


# #### Split dataset

# In[5]:


X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.80, random_state=42)


# #### Feature selection using correlation

# In[6]:


import seaborn as sns

fig, ax = plt.subplots( figsize = ( 12 , 10 ) )
sns.heatmap(df.corr(method='pearson'), annot = True)


# #### Feature selection using chi-squared test

# In[7]:


from sklearn.feature_selection import chi2


# In[8]:


chi2_values, p_values = chi2(X_train, y_train)

chi2_results = pd.DataFrame({
    'Feature': X_train.columns,
    'Chi2 Value': chi2_values,
    'P-Value': p_values
}).sort_values(by='Chi2 Value', ascending=False)

# Step 5: Visualize the results with a barplot
plt.figure(figsize=(10, 6))
sns.barplot(x='Chi2 Value', y='Feature', data=chi2_results)
plt.title('Chi-Squared Test Results for Feature Importance')
plt.xlabel('Chi2 Value')
plt.ylabel('Features')
plt.show()

# Optional: Display the chi-squared and p-values
print(chi2_results)


# ##### Fairer Comparison of Variance With Feature Normalization
# Often, it is not fair to compare the variance of a feature to another. The reason is that as the values in the distribution get bigger, the variance grows exponentially. In other words, the variances will not be on the same scale.

# In[9]:


X_train.describe()


# #### Feature selection using correlation

# In[10]:


X_train.boxplot()


# The above features all have different medians, quartiles, and ranges — completely different distributions. We cannot compare these features to each other.
# 
# One method we can use is scale all features using the Robust Scaler which is not highly affected by outliers:

# In[11]:


from sklearn.preprocessing import RobustScaler

transformer = RobustScaler()
# scale all features
scaled = transformer.fit_transform(X_train)
# covert scaled array to Pandas DataFrame
X_train_scaled = pd.DataFrame(scaled, columns=X_train.columns, index=X_train.index)


# This method ensures that all variances are on the same scale:

# In[12]:


X_train_scaled.var()


# In[13]:


from sklearn.feature_selection import VarianceThreshold

selector = VarianceThreshold(threshold=0.35)
# Learn variances from X_train_scaled
_ = selector.fit(X_train_scaled)
# Get a mask (or integer index if indices=True is set) of the features selected
mask = selector.get_support()
print(mask)
# get the subset of features selected
X_train_transformed = X_train_scaled.loc[:,mask]
X_train_transformed


# #### 

# #### Feature selection using feature importance

# In[14]:


# Feature Importance 1
# Use ensemble method: The goal of ensemble methods is to combine the 
# predictions of several base estimators built with a given learning algorithm 
# in order to improve generalizability / robustness over a single estimator.
# http://scikit-learn.org/stable/modules/ensemble.html
from sklearn.ensemble import ExtraTreesClassifier
# Build an estimator (forest of trees) and compute the feature importances
# n_estimators = number of trees in forest
estimator = ExtraTreesClassifier(n_estimators=100, max_features=13, random_state=0)
estimator.fit(X_train,y_train)
# Lets get the feature importances. Features with high importance score higher.
importances = estimator.feature_importances_
std = np.std([tree.feature_importances_ for tree in estimator.estimators_],
             axis=0)
indices = np.argsort(importances)[::-1]

# Print the feature ranking
print("Feature ranking:")
for f in range(X_train.shape[1]):
    print("%d. feature %d - %s (%f)" % (f + 1, indices[f], X_train.columns[indices[f]], importances[indices[f]]))

# Plot the feature importances of the forest
plt.figure()
plt.title("Feature importances")
plt.bar(range(X_train.shape[1]), importances[indices],
       color="r", yerr=std[indices], align="center")
plt.xticks(range(X_train.shape[1]), indices)
plt.xlim([-1, X_train.shape[1]])
plt.show()


# In[14]:


# Feature Importance 2
from sklearn.feature_selection import RFE
estimator = ExtraTreesClassifier(n_estimators=100, random_state=0)
# keep the 5 most informative features
selector = RFE(estimator, n_features_to_select=5, step=1)
selector = selector.fit(X_train, y_train)
print(list(selector.support_))
print(list(selector.ranking_))


# #### Feature selection using Forward selection/Backward elimination

# In[15]:


# Example 1
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=4)

from mlxtend.feature_selection import SequentialFeatureSelector as SFS
# Sequential Forward Selection
sfs = SFS(knn, 
           k_features=5, 
           forward=True, 
           floating=False, 
           verbose=2,
           scoring='accuracy',
           cv=10,
           n_jobs=-1)

sfs = sfs.fit(X_train_scaled, y_train)

print('\nSequential Forward Selection (k=5):')
print('Selected features:',sfs.k_feature_idx_)
print('Prediction score:',sfs.k_score_)


# In[16]:


# Example 2
# Sequential Backward Selection
sbs = SFS(knn, 
          k_features=5, 
          forward=False, 
          floating=False, 
          scoring='accuracy',
          cv=10,
          n_jobs=-1)

sbs = sbs.fit(X_train_scaled, y_train)

print('\nSequential Backward Selection (k=5):')
print('Selected features:',sbs.k_feature_idx_)
print('Prediction (CV) score:',sbs.k_score_)


# In[17]:


# Example 3
print(pd.DataFrame.from_dict(sbs.get_metric_dict()).T)


# In[18]:


# Example 4
from mlxtend.plotting import plot_sequential_feature_selection as plot_sfs

sfs = SFS(knn, 
          k_features=5, 
          forward=True, 
          floating=False, 
          scoring='accuracy',
          verbose=2,
          cv=10,
          n_jobs=-1)

sfs = sfs.fit(X_train_scaled, y_train)

fig1 = plot_sfs(sfs.get_metric_dict(), kind='std_dev')

plt.ylim([0.65, 1])
plt.title('Sequential Forward Selection (w. StdDev)')
plt.grid()
plt.show()


# In[19]:


# Example 5
knn = KNeighborsClassifier(n_neighbors=4)

sfs_range = SFS(estimator=knn, 
           k_features=(2, 13),
           forward=True, 
           floating=False, 
           scoring='accuracy',
           cv=10,
           n_jobs=-1)

sfs_range = sfs_range.fit(X_train_scaled, y_train)

print('best combination (ACC: %.3f): %s\n' % (sfs_range.k_score_, sfs_range.k_feature_idx_))
print('all subsets:\n', sfs_range.subsets_)
plot_sfs(sfs_range.get_metric_dict(), kind='std_err');
plt.show()


# In[20]:


# export the selected set of features in a new dataset
X_train_scaled_selected = sfs_range.transform(X_train_scaled)
X_train_scaled_selected


# #### Feature Extraction

# In[21]:


def print_2d_scatter_plot(features, target, title, n_components):
    newDf=pd.DataFrame(features, columns=['Feature '+str(i) for i in range(features.shape[1])])
    newDf['target']=target
    sns.scatterplot(data=newDf, x='Feature 0', y='Feature 1', hue='target').set(title=title+', '+str(n_components)+' components')
    plt.show() 

def print_variance_explained_plot(obj, n_components):
    cum_var_exp = np.cumsum(obj.explained_variance_ratio_)
    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(range(n_components), obj.explained_variance_ratio_, alpha=0.5, align='center',
                label='individual explained variance')
    # show percentage of explained variance on top of each bar
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., 1.05*height, '%.2f%%' % (height*100), 
                ha='center', va='bottom')
    plt.step(range(n_components), cum_var_exp, where='mid',
                 label='cumulative explained variance')
    plt.ylabel('Explained variance ratio')
    plt.xlabel('Principal components')
    plt.xticks( range(n_components) )
    plt.grid(True)
    plt.legend(loc='best')
    plt.tight_layout()
    plt.show()

# PCA on the wine dataset
from sklearn.decomposition import PCA
components = 10
pca = PCA(n_components=components)
# train PCA
pca.fit(X_train_scaled)
# Percentage of variance explained for each components
print('PCA explained variance ratio (first '+str(components)+' components): %s'
      % str(pca.explained_variance_ratio_))
print_variance_explained_plot(pca,components)


# In[22]:


# As can be seen from the figure above, 7 principal components (new features created by PCA) cumulatively explain ~90% of the variance 
pca = PCA(n_components=7)
pca.fit(X_train_scaled)

plt.figure(figsize=(8, 5))
plt.plot(range(1, len(pca.explained_variance_ratio_) + 1), pca.explained_variance_ratio_.cumsum(), marker='o')
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance')
plt.title('Explained Variance by Components')
plt.grid()
plt.show()


# In[23]:


# new extracted feature set with the new 7 features
X_train_scaled_pca = pca.transform(X_train_scaled)
X_train_scaled_pca


# #### Feature Extraction example using PCA, SVD and LDA

# In[24]:


from sklearn.datasets import load_iris
iris = load_iris()
iris_data = iris.data
iris_target = iris.target
num_of_classes = len(set(iris_target))

# Truncated SVD
from sklearn.decomposition import TruncatedSVD
tsvd = TruncatedSVD(2)
iris_tsvd = tsvd.fit_transform(iris_data)
print_2d_scatter_plot(iris_tsvd, iris_target, 'Truncated SVD', 2)
# Percentage of variance explained for each components
print('TruncatedSVD explained variance ratio (first two components): %s'
      % str(tsvd.explained_variance_ratio_))
print_variance_explained_plot(tsvd,2)

# PCA (no need to scale features because all features on the same scale)
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
iris_pca = pca.fit_transform(iris_data)
print_2d_scatter_plot(iris_pca, iris_target, 'PCA', 2)
# Percentage of variance explained for each components
print('PCA explained variance ratio (first two components): %s'
      % str(pca.explained_variance_ratio_))
print_variance_explained_plot(pca,2)

# Linear Discriminant Analysis
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
lda = LinearDiscriminantAnalysis(n_components=2)
iris_lda = lda.fit(iris_data, iris_target).transform(iris_data)
print_2d_scatter_plot(iris_lda, iris_target, 'LDA', 2)
# Percentage of variance explained for each components
print('LDA explained variance ratio (first two components): %s'
      % str(lda.explained_variance_ratio_))
print_variance_explained_plot(lda,2)

