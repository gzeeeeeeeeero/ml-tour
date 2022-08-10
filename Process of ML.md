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

- Homoscedasticity - Homoscedasticity refers to the "assumption that dependent variable(s) exhibit equal levels of variance across the range of predictor variable(s)" (Hair et al., 2013). Homoscedasticity is desirable because we want the error term to be the same across all values of the independent variables.

- Linearity- The most common way to assess linearity is to examine scatter plots and search for linear patterns. If patterns are not linear, it would be worthwhile to explore data transformations.

- Absence of correlated errors - When one error is correlated to another. For instance, if one positive error makes a negative error systematically, it means that there's a relationship between these variables. This occurs often in time series, where some patterns are time related.
  
  ```r
  For more information search 
  - EDA on google
  - EDA notebooks on kaggle
  - Hair et al., 2013) 
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

*for finding correlations &| FS*

![](https://machinelearningmastery.com/wp-content/uploads/2019/11/How-to-Choose-Feature-Selection-Methods-For-Machine-Learning.png)

- Pearson’s correlation coefficient (linear)

- Spearman’s rank coefficient (nonlinear)

- ANOVA correlation coefficient (linear)

- Kendall’s rank coefficient (nonlinear)

**Filter Method**: In this method, features are dropped based on their relation to the output

**Wrapper Method**: We split our data into subsets and train a model using this. Based on the output of the model, we add and subtract features and train the model again

Examples :

- Forward Selection : We start with empty set and add feature one by one

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

### *Dimentionality Reduction Techniques*



### *Removing features with low variance*

```python
>>> from sklearn.feature_selection import VarianceThreshold
>>> X = [[0, 0, 1], [0, 1, 0], [1, 0, 0], [0, 1, 1], [0, 1, 0], [0, 1, 1]]
>>> sel = VarianceThreshold(threshold=(.8 * (1 - .8)))
>>> sel.fit_transform(X)
array([[0, 1],
       [1, 0],
       [0, 0],
       [1, 1],
       [1, 0],
       [1, 1]])
```

3. ### Model Selection

4. ### Fitting

5. ### Hyperparameter Tuning : Can be Implemented with `GridSearchCV` of `sklearn.model_selection` as follows
   
   Metrics - Model Eval, roc_auc
