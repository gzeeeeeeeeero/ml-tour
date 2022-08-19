# Machine Learning Steps

---

## **<u>EDA \[Optional (kind of)\]</u>**

### *Visual Inspection*

- data.shape

- data.head()

- data.describe() (statistical description)

- data.info() (tells dtypes, total entries, non-null counts, mem-usage)

- data.value_counts() (for discrete column) judge what type of classification problem if discrete target, for eg if one class dominates then it is imbalanced classif.
  
  Example :
  
  ```python
  >>> DataFrame({'A':random.randn(10), 
                 'B':list("yes" if i%2 
                     else "no" for i in range(10))}
               )['B'].value_counts()
  no     5
  yes    5
  Name: B, dtype: int64
  
  # Somehow the dtype is int64
  ```

- \[Correlations\] Heatmap of `DataFrame.corr()`
  
  Note : it uses **Pearson Correlation** which works poorly on non-linear data use **Spearman’s rank coefficient** instead `df.corr(method='spearman')`
  
  - Example :
    
    ```python
    >>> plt.subplots(figsize=(12, 12))
    >>> sns.heatmap(data.corr(), annot=True, cbar=False, 
                  fmt=".2f", square=True, *optionally set vmin/max*)
    ```
  
  - Or just :
    
    ```python
    >>> data.corr()[target].sort_values(ascending=False)
    ```

- boxplots for discrete features

- sns.pairplot (pairwise scatterplots) to understand linearty of features

### *Four Assumptions of Hair et. al (2013)*

- Normality - Seeing how Normally distributed data is, we can see it univariately by scaling the data to have zero mean and SD of 1 using `StandardScaler()` or `PowerTransformer()` i have notebooks that demo them, or we can see it multivariately using a histogram and a normal probability plot
  
  Example :
  
  ```python
    """
    Histogram - Kurtosis and skewness.
    Normal probability plot - Data distribution should closely follow the diagonal that represents the normal distribution.
    """
  
    from scipy import stats
    from scipy.stats import norm
  
    sns.distplot(df_train['SalePrice'], fit=norm);
    fig = plt.figure()
    res = stats.probplot(df_train['SalePrice'], plot=plt)
  
    # To plot all continous columns :
    for i, col in enumerate(cols):
      plt.subplot(r, c, i+1)
      sns.histplot(...)
      plt.title(col)
    # Use barplot for discrete variables
  ```
  
    ![Histogram](/home/g0/Pictures/assets/download.png)
    ![Probability Plot](/home/g0/Pictures/assets/download%20(1).png)
    the skewness is eliminated by taking log, in case the data have zeros then those zeros are ignored in log and a binary indicator variable column is added.

- Homoscedasticity - Homoscedasticity refers to the "assumption that dependent variable(s) exhibit equal levels of variance across the range of predictor variable(s)" (Hair et al., 2013). Homoscedasticity is desirable because we want the error term to be the same across all values of the independent variables. we can check for Homoscedasticity with the Box's M statistic test which is implmeneted in [`scipy.stats.levene`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.levene.html) with a p value >0.05 we can assume Homoscedasticity is present

- Linearity- The most common way to assess linearity is to examine scatter plots and search for linear patterns. If patterns are not linear, it would be worthwhile to explore data transformations.

- Absence of correlated errors - When one error is correlated to another. For instance, if one positive error makes a negative error systematically, it means that there's a relationship between these variables. This occurs often in time series, where some patterns are time related.
  
  ```r
  For more information search 
  - EDA on google
  - EDA notebooks on kaggle
  - Hair et al., (2013) 
  ```

---

## <u>Data Cleaning</u>

### *Missing Values*

- Decide to Fill them somehow or just drop the column based on importance of feature or the amount of missing data or make a tradeoff, or drop the rows if the missing data isnt too much
  
  ```python
  # Drop Columns
  df.drop(columns=unimportant_cols)
  
  # Drop rows when X_y
  X = train_df.dropna()
  y = train_df.loc[X.index, "target"]
  ```

- To get count (since data.info() only gives non-null count) :
  
  1. Quick :
     
     ```python
     df.replace('', np.NaN)
     df.isna().sum().sort_values(ascending=False)
     ```
  
  2. Plot :
     
     ```python
     import missingno as msno
     
     msno.bar(df)
     # or
     msno.matrix(df)
     # or
     msno.heatmap(df)
     ```

- Filling :
  
  1. Simple Imputing \[Univariate\] : Fill with Mean/Median for continous or Mode for categorical of that column. If the missing values are a lot in case of Categorical, we make it a new category.
      Demo :
     
     ```python
     >>> import pandas as pd
     >>> df = pd.DataFrame([["a", "x"],
     ...                    [np.nan, "y"],
     ...                    ["a", np.nan],
     ...                    ["b", "y"]], dtype="category")
     ...
     >>> imp = SimpleImputer(strategy="most_frequent")
     >>> print(imp.fit_transform(df))
     [['a' 'x']
      ['a' 'y']
      ['a' 'y']
      ['b' 'y']]
     ```
     
      Or fill with a constant value, `strategy='constant'` then `fill_value=fi_val`
  
  2. KNN Imputing : Impute with N neighbours
     
     ```python
     >>> import numpy as np
     >>> from sklearn.impute import KNNImputer
     >>> nan = np.nan
     >>> X = [[1, 2, nan], [3, 4, 3], [nan, 6, 5], [8, 8, 7]]
     >>> imputer = KNNImputer(n_neighbors=2, weights="uniform")
     >>> imputer.fit_transform(X)
     array([[1. , 2. , 4. ],
            [3. , 4. , 3. ],
            [5.5, 6. , 5. ],
            [8. , 8. , 7. ]])
     ```
     
      or optionally `weights="distance"` or `callable` : a user-defined function which accepts an array of distances, and returns an array of the same shape containing the weights.
  
  3. Predict Missing Values :
      We can use other features to predict a feature (select features that have strong corr with the feature with missing vals), in sklearn we have [Iterative Imputer](https://scikit-learn.org/stable/modules/generated/sklearn.impute.IterativeImputer.html) for that or we can do it ourselves by first initializing the missing values with some imputer and then using the non-missing values for training/testing and then predict the missing values or use a DL imputing library from AWS namely [Datawig](https://datawig.readthedocs.io/en/latest/)
      Demo :
     
     ```python
     import datawig
     df_train, df_test = datawig.utils.random_split(data)
     
     imputer = datawig.SimpleImputer(
         input_columns=in_cols,
         output_column=out_col,
         output_path =path_to_model_dir # stores model data and metrics
         )
     imputer.fit(train_df=df_train, num_epochs=50)
     imputed = imputer.predict(df_test)
     ```

### *Novelty and Outlier Detection*

#### Outlier detection

The training data contains outliers which are defined as observations that are far from the others. Outlier detection estimators thus try fit the regions where the training data is the most concentrated, ignoring the deviant observations.

#### Novelty detection

The training data is not polluted by outliers and we are interested in detecting whether a new observation is an outlier. In this context outlier is also called a novelty.

Generally a datapoint at a distance of 3+= Standard Deviations is considered an outlier or a datapoint
For a Gaussian Distribution :

- Standard Deviation from the Mean: 68% data

- Standard Deviations from the Mean: 95% data

- Standard Deviations from the Mean: 99.7% data

Use [`sklearn.neighbors`](https://scikit-learn.org/stable/modules/classes.html#module-sklearn.neighbors "sklearn.neighbors").[`LocalOutlierFactor`]([sklearn.neighbors.LocalOutlierFactor — scikit-learn 1.1.2 documentation](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.LocalOutlierFactor.html#sklearn.neighbors.LocalOutlierFactor))

Demo :

```python
 >>> import numpy as np
 >>> from sklearn.neighbors import LocalOutlierFactor
 >>> X = [[-1.1], [0.2], [101.1], [0.3]]
 >>> clf = LocalOutlierFactor(n_neighbors=2)
 >>> clf.fit_predict(X)
 array([ 1, 1, -1, 1]) # -1 = outlier
 >>> clf.negative_outlier_factor_
 array([ -0.9821..., -1.0370..., -73.3697..., -0.9821...])
```

### *Dealing with Categorical Data*

OneHotEncoding :

```python
pd.get_dummies(df, cols=cat_cols)

# OR

from sklearn.preprocessing import OneHotEncoder

df = OneHotEncoder(categories=cat_cols|'auto'[default]).fit_transform(df)
```

LabelEncoding (not suitable for some ML models)

### *Dealing with Imbalances*

to create sample imbalanced data :

```python
!pip install imblearn # a scikit-learn lib for imbalanced learning

from imblearn.datasets import make_imbalance

df_res, y_res = make_imbalance(df, y, sampling_strategy={class_label: class_count ....},)
```

- collect more data of minority (if u can)

- pending ...

### *Scaling the Data*

Required when data is on different scales, smthn is 1-5 smthn is 1-200000. A [`StandardScaler`](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html#sklearn.preprocessing.StandardScaler "sklearn.preprocessing.StandardScaler") can be used generally but a [`MaxAbsScaler`](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MaxAbsScaler.html#sklearn.preprocessing.MaxAbsScaler "sklearn.preprocessing.MaxAbsScaler") is used in case of sparse data, [`RobustScaler`](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.RobustScaler.html#sklearn.preprocessing.RobustScaler "sklearn.preprocessing.RobustScaler") is used in case of Outliers; [`MinMaxScaler`](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MinMaxScaler.html#sklearn.preprocessing.MinMaxScaler "sklearn.preprocessing.MinMaxScaler") and [`MaxAbsScaler`](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MaxAbsScaler.html#sklearn.preprocessing.MaxAbsScaler "sklearn.preprocessing.MaxAbsScaler") can fit data into a range as well, with `feature_range=(min, max)`

Some Non-linear Transformers :

                                                    <img src="https://scikit-learn.org/stable/_images/sphx_glr_plot_map_data_to_normal_001.png" title="" alt="" width="378">

    See also : [Demo-Notebook](/home/g0/Programming/ML/SL/Dis/Untitled.ipynb)

---

## [<u>Feature Engineering</u>](https://elitedatascience.com/feature-engineering-best-practices)

### *Discretization/Binning*

- Breaking continous data into ranges (for eg house_price, ages) : [`KBinsDiscretizer`](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.KBinsDiscretizer.html#) , [`Binarizer`](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.Binarizer.html#sklearn.preprocessing.Binarizer "sklearn.preprocessing.Binarizer")

### *Combining 2+= Features*

- For example, bin the age and combine with gender to make a feature like "old man"

### *Decomposing a Feature into 2+= features*

- Decompose a Date for example :
  
  ```python
  dates = array([[f'2022-{j}-{i}' for i in range(1, 21)] for j in range(1, 13)]).flatten()
  df = DataFrame(dates, columns=["Date"])
  df["Date"] = to_datetime(df["Date"], format="%Y-%m-%d", errors='coerce') # format should match
  df.dt.day # For instance
  
  # Extracting dates from a csv with parse_dates argument
  
  parse_dates : bool or list of int or names or list of lists or dict, default False
      The behavior is as follows:
  
      * boolean. If True -> try parsing the index.
      * list of int or names. e.g. If [1, 2, 3] -> try parsing columns 1, 2, 3
        each as a separate date column.
      * list of lists. e.g.  If [[1, 3]] -> combine columns 1 and 3 and parse as
        a single date column.
      * dict, e.g. {'foo' : [1, 3]} -> parse columns 1, 3 as date and call
        result 'foo' 
  ```

doing this enables :

![](https://miro.medium.com/max/460/1*S5vHn10C5T_0PqIwZhTf1g.png)

### *Indicator Features*

Using some features to make an indicator such as:

- create X_is_missing

- car_travelled_under_100KM

---

## <u>Feature Selection (Dimensionality Reduction)</u>

**Filter Method**: In this method, features are dropped based on their relation to the output, or how they are **correlating** to the output

**Wrapper Method**: We split our data into subsets and train a model using this. Based on the output of the model, we add and subtract features and train the model again. In the core a machine learning algorithm is used to give weights to features unlike filter methods. This is achieved by fitting the given machine learning algorithm used in the core of the model, ranking features by importance, discarding the least important features, and re-fitting the model. This process is repeated until a specified number of features remains

**Intrinsic Method**: This method combines the qualities of both the Filter and Wrapper method to create the best subset

*for finding correlations*

![](https://machinelearningmastery.com/wp-content/uploads/2019/11/How-to-Choose-Feature-Selection-Methods-For-Machine-Learning.png)

- Pearson’s correlation coefficient (linear)

- Spearman’s rank coefficient (nonlinear)

- ANOVA correlation coefficient (linear)

- Kendall’s rank coefficient (nonlinear)

### *Select Top X features [Univariate]*

```python
from sklearn.feature_selection import SelectKBest, SelectPercentile, SelectFromModel, mutual_info_classif # mutual_info_regression for regr prob

# Selecting with a correlation ranker

selector = SelectKBest(mutual_info_classif, k=20)
X_reduced = selector.fit_transform(X, y) # X is feature space

selector = SelectPercentile(mutual_info_classif, percentile=25)
X_reduced = selector.fit_transform(X, y)

# selector.get_support returns the feature names

# Selection using Models
# Utilizes the coef_ or feature_importance_ attributes of models

clf = Pipeline([
('feature_selection', SelectFromModel(LinearSVC(penalty="l1"))),
('classification', RandomForestClassifier())
])
clf.fit(X, y)
```

### *An [ExhaustiveFeatureSelector](http://rasbt.github.io/mlxtend/user_guide/feature_selection/ExhaustiveFeatureSelector/)*

```python
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import load_iris
from mlxtend.feature_selection import ExhaustiveFeatureSelector as EFS

iris = load_iris()
X = iris.data
y = iris.target

knn = KNeighborsClassifier(n_neighbors=3)

efs1 = EFS(knn, 
           min_features=1,
           max_features=4,
           scoring='accuracy',
           print_progress=True,
           cv=5)

efs1 = efs1.fit(X, y)

print('Best accuracy score: %.2f' % efs1.best_score_)
print('Best subset (indices):', efs1.best_idx_)
print('Best subset (corresponding names):', efs1.best_feature_names_)
```

```python
Features: 15/15

Best accuracy score: 0.97
Best subset (indices): (0, 2, 3)
Best subset (corresponding names): ('0', '2', '3')
```

### *A [SequentialFeatureSelector]([SequentialFeatureSelector: The popular forward and backward feature selection approaches incl. floating variants - mlxtend](http://rasbt.github.io/mlxtend/user_guide/feature_selection/SequentialFeatureSelector/)) [Greedy]*

**Sequential Forward Selection (SFS)**

1. Initialize with empty set and set a P-Value (5% for eg)

2. Add a new feature; keep the feature if it maximized criterion function

3. Repeat Until Convergeance (when reached max_features)

```python
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import load_iris

iris = load_iris()
X = iris.data
y = iris.target
knn = KNeighborsClassifier(n_neighbors=4)


from mlxtend.feature_selection import SequentialFeatureSelector as SFS

sfs1 = SFS(knn, 
           k_features=3, # max features
           forward=True, 
           floating=False, 
           verbose=2,
           scoring='accuracy',
           cv=0)

sfs1 = sfs1.fit(X, y)
```

**Sequential Backward Selection (SBS)** performs exclusions on full set of features

### *Dimentionality Reduction Techniques*

#### PCA

PCA works by projecting data into a lower dimensional space that captures the most variance using principal components which are orthogonal vectors, the 1<sup>st</sup> PC have most variance compared to any vector in the vector space of features, the second PC have 2<sup>nd</sup> highest variance.

<img src="https://assets.website-files.com/5e6f9b297ef3941db2593ba1/5f76ef7799e20652be0d79f6_Screenshot%202020-10-02%20at%2011.12.32.png" title="" alt="" width="251">

There are many mathematical ways to calculate/implement PCA, [`sklearn.decomposition`](https://scikit-learn.org/stable/modules/classes.html#module-sklearn.decomposition "sklearn.decomposition").[`PCA`](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html#) automatically selects the svd solver depending upon data

Demo :

```python
from sklearn.decomposition import PCA
# PCA is sensitive to scaling (perform Standard Scaling first)
X = StandardScaler().fit_transform(X)
pca = PCA().fit(X) # by default n_components=None which makes n_components = n_features

>>> with printoptions(precision=4, suppress=True):
...:     print(pca.explained_variance_ratio_.cumsum()*100)
[ 43.3817  62.9563  72.4801  79.1868  84.4887  88.5793  90.8581  92.5663
  93.9582  95.1609  96.1642  97.016   97.8511  98.3379  98.6483  98.8999
  99.1016  99.2811  99.4458  99.5569  99.6585  99.7498  99.8324  99.8892
  99.9417  99.9686  99.9917  99.997   99.9996 100.    ]
# this shows that first 10 components capture 95% Variance in Data
covariance
>>> pca.components_.shape
(30, 30)

X_scaled = StandardScaler().fit_transform(X)
pca = PCA(n_components=0.95).fit(X_scaled)
>>> pca.n_components_
10

X_scaled_pca = pca.transform(X_scaled)
xt, xv, yt, yv = train_test_split(X_scaled_pca, y)
>>> print(classification_report(SVC().fit(xt, yt).predict(xv), yv)
              precision    recall  f1-score   support

           0       0.96      0.96      0.96        53
           1       0.97      0.97      0.97        65

    accuracy                           0.97       118
   macro avg       0.97      0.97      0.97       118
weighted avg       0.97      0.97      0.97       118
```

to deduce which are the most important features in the components. since **each component is some linear combination of all the features** it doesnt matter how many components we initialize `PCA` with each component still shows importance of all the features

![](/home/g0/.config/marktext/images/2022-08-12-23-04-34-image.png)

printing top 4 features in 6 components

```python
pca = PCA(n_components=6).fit(X_scaled)
>>> pca.explained_variance_ratio_.cumsum()[-1]*100
88.57927184093991

>>> cols = load_breast_cancer(as_frame=True).data.columns
       ...: all_feat = []
       ...: for i, comp in enumerate(pca.components_):
       ...:     print(f"===== Component - {i+1} ======")
       ...:     top_4 = cols[argpartition(comp, 4)[:4]].to_list()
       ...:     print(top_4)
       ...:     all_feat+=top_4
       ...: print("\n===== Unique Features =====")
       ...: print(set(all_feat), f"\nTotal - {len(set(all_feat))}")

===== Component - 1 ======
['smoothness error', 'texture error', 'symmetry error', 'mean fractal dimension']
===== Component - 2 ======
['mean radius', 'mean area', 'worst area', 'worst radius']
===== Component - 3 ======
['worst symmetry', 'worst smoothness', 'worst compactness', 'worst fractal dimension']
===== Component - 4 ======
['mean smoothness', 'area error', 'radius error', 'perimeter error']
===== Component - 5 ======
['mean smoothness', 'worst smoothness', 'mean symmetry', 'symmetry error']
===== Component - 6 ======
['mean smoothness', 'smoothness error', 'worst smoothness', 'mean fractal dimension']

===== Unique Features =====
{'symmetry error', 'worst smoothness', 'worst fractal dimension', 'perimeter error', 
'mean radius', 'smoothness error', 'mean smoothness', 'worst compactness', 
'mean fractal dimension', 'worst symmetry', 'area error', 'mean symmetry', 
'radius error', 'worst area', 'texture error', 'worst radius', 'mean area'} 
Total - 17
```

#### LDA

LDA works by finding vectors that give the most seperability in the datapoints it is a **supervised** classification algorithm, it does so by maximizing the seperation of means of data clusters and It minimizes the variation or scatter within each category represented by s², it can be used for dimensionality reduction by projecting the data along vectors that are normal to the seperating vectors

<img src="file:///home/g0/Pictures/assets/lda.png" title="" alt="" width="1072">

Assumptions of LDA for best performance :

1. Normality

2. Homoscedasticity : If this fails then we use Quadratic Discriminant Analysis aka Gaussian Discriminant Analysis

3. Multicollinearity : The performance of prediction can decrease with the increased correlation between the independent variables.

LDA tends to overfit rapidly especially when these assumptions are broken, sometimes it makes sense to apply PCA before LDA as a regularizer

![](https://i.stack.imgur.com/Gv4n7.png)

Although this is a possible strategy still it is adviced to use rLDA

#### Kernel PCA (Non-Linear)

Kernel PCA is an extension of [PCA](https://ml-explained.com/blog/principal-component-analysis-explained) that allows for the separability of nonlinear data by making use of kernels. The basic idea behind it is to project the linearly inseparable data onto a higher dimensional space where it becomes linearly separable. RBF is a general purpose kernel, cross-validation currently is the only good way to figure which kernel is best, for eg if data looks like it follows a polynomial function use poly kernel

#### t-SNE

t-SNE (tees-knee) (t-Distributed Stochastic Neighbor Embedding) is a dim red tech used for visualizing very high dimensional data in 2d or 3d, like images, audio, words etc generally when d > 50. It is an iterative unsupervised algorithm that makes clusters of similar datapoints. [more information](https://towardsdatascience.com/t-sne-clearly-explained-d84c537f53a), [visualisation, experiments and tips to use it efficiently]([How to Use t-SNE Effectively](https://distill.pub/2016/misread-tsne/)), [visualization of t-SNE on clustering similar words together](http://projector.tensorflow.org/)

> t-SNE [1] is a tool to visualize high-dimensional data. It converts similarities between data points to joint probabilities and tries to minimize the Kullback-Leibler divergence between the joint probabilities of the low-dimensional embedding and the high-dimensional data.

which also means that t-SNE does not retain/care-about the size, density and distance of clusters, it retains probabilities/neighbours which means running a distance/density based clustering algorithm such as [DBSCAN](https://www.kdnuggets.com/2020/04/dbscan-clustering-algorithm-machine-learning.html) on its output might not be a good idea; the perplexity hyperparameter of t-SNE is comparable to the nearest neighbours of KNN

> In contrast to, e.g., PCA, t-SNE has a non-convex objective function. The objective function is minimized using a gradient descent optimization that is initiated randomly. As a result, it is possible that different runs give you different solutions. Notice that it is perfectly fine to run t-SNE a number of times (with the same data and parameters), and to select the visualization with the lowest value of the objective function as your final visualization.

> finds patterns in the data by identifying observed clusters based on similarity of data points with multiple features. But it is not a clustering algorithm it is a dimensionality reduction algorithm. This is because it maps the multi-dimensional data to a lower dimensional space, the input features are no longer identifiable. Thus you cannot make any inference based only on the output of t-SNE. So essentially it is mainly a data exploration and visualization technique.
> 
> But t-SNE can be used in the process of classification and clustering by using its output as the input feature for other classification algorithms.

Cluster sizes in any t-SNE plot must not be evaluated for standard deviation, dispersion or any other similar measures. This is because t-SNE expands denser clusters and contracts sparser clusters to even out cluster sizes. This is one of the reasons for the crisp and clear plots it produces.

[video walkthrough](https://www.youtube.com/watch?v=NEaUSP4YerM).

[why clustering on t-SNE's output is not always a good idea]((https://stats.stackexchange.com/questions/263539/clustering-on-the-output-of-t-sne).

### *Removing Multicolinearity*

Colinearity is the state where two variables are highly correlated and contain similiar information about the variance within a given dataset. To detect colinearity among variables, simply create a correlation matrix and find variables with large absolute values. In R use the [`corr`](http://www.sthda.com/english/wiki/correlation-matrix-a-quick-start-guide-to-analyze-format-and-visualize-a-correlation-matrix-using-r-software) function and in python this can by accomplished by using numpy's [`corrcoef`](https://docs.scipy.org/doc/numpy/reference/generated/numpy.corrcoef.html) function.

[Multicolinearity](https://en.wikipedia.org/wiki/Multicollinearity) on the other hand is more troublesome to detect because it emerges when three or more variables, which are highly correlated, are included within a model. To make matters worst multicolinearity can emerge even when isolated pairs of variables are not colinear.

**Steps for Implementing VIF**

1. Run a multiple regression.
2. Calculate the VIF factors.
3. Inspect the factors for each predictor variable, if the VIF is between 5-10, multicolinearity is likely present and you should consider dropping the variable.

```python
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant

df = pd.DataFrame(
    {'a': [1, 1, 2, 3, 4],
     'b': [2, 2, 3, 2, 1],
     'c': [4, 6, 7, 8, 9],
     'd': [4, 3, 4, 5, 4]}
)

X = add_constant(df)
>>> pd.Series([variance_inflation_factor(X.values, i) 
               for i in range(X.shape[1])], 
              index=X.columns)
const    136.875
a         22.950
b          3.000
c         12.950
d          3.000
dtype: float64
```

### *Removing features with low variance*

cheap way when u know the data strictly follows a distribution and u can use the variance formula

```python
>>> from sklearn.feature_selection import VarianceThreshold
>>> X = [[0, 0, 1], [0, 1, 0], [1, 0, 0], [0, 1, 1], [0, 1, 0], [0, 1, 1]]
>>> sel = VarianceThreshold(threshold=(.8 * (1 - .8))) # this is variance of bernoulli dist
>>> sel.fit_transform(X)
array([[0, 1],
       [1, 0],
       [0, 0],
       [1, 1],
       [1, 0],
       [1, 1]])
```

## Data Splitting

Train Set : 70-80% data, model trained on this

Val Set : 10-15% data, hyperparameters tuned on this

Test Set : 10-15% data, model test on this

More on [Cross Validation](Cross Validation Techniques.md)

## Model Selection

Bias-Variance Trade-Off

difference between bias and variance

![](/home/g0/Pictures/assets/selection.png)

3. ### Fitting

4. ### Hyperparameter Tuning : Can be Implemented with `GridSearchCV` of `sklearn.model_selection` as follows tuning HPs in Pipeline
   
   Metrics - Model Eval, roc_auc, classif_report
