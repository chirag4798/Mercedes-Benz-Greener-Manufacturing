# Mercedes-Benz Greener Manufacturing
Can you cut the time a Mercedes-Benz spends on the test bench?

<a href="https://imgur.com/Pl6iv1X"><img src="https://i.imgur.com/Pl6iv1X.jpg" title="source: imgur.com" /></a>

## Problem Statement
<p align="justify">An automobile has various components like, engine, chassis, steering, suspension, transmission etc which work together in tandem to provide the driving experience. These systems can be thought of as building blocks of an automobile that has to be fit together to form a vehicle. These building blocks can have variations in them in accordance with the purpose they’re being used for. While building a vehicle the manufacturer has to take into account the different ways these components fit and interact with each other, for which the vehicle has to be tested rigorously before its sent on road. The time it takes to test the performance, comfort, reliability and safety of a vehicle is highly correlated with the type of component it’s made of. Testing cars is expensive hence an entry level passenger car does not undergo the same criteria for testing as a high end luxury car, depending on the components and the type of car the test time may vary. To accurately estimate the test time required one has to account for each and every configuration of the components and their interaction, which would be a complex and time consuming process if done non-algorithmically, hence the need for a machine learning solution.</p>

## Dataset
<p align="justify">A dataset has been prepared by the engineers in Daimler to tackle this problem. The dataset contains permutations of car features, configurations and different testing methods applied with their respective time required for testing, the features are anonymized i.e. the name of the features are not interpretable. The dataset mainly consists of binary and categorical data, where the binary data represents the type of test being carried out and the categorical data represents the features of the vehicle. The goal of this case study is to provide a model that can estimate the time with the least amount of error.</p>
    
[Data Source](https://www.kaggle.com/c/mercedes-benz-greener-manufacturing)

- This dataset contains an anonymized set of variables, each representing a custom feature in a Mercedes car. For example, a variable could be 4WD, added air suspension, or a head-up display.

- The ground truth is labeled ‘y’ and represents the time (in seconds) that the car took to pass testing for each variable.

**File descriptions**
- Variables with letters are categorical. Variables with 0/1 are binary values.

    - **train.csv** - the training set
    - **test.csv** - the test set, one has to predict the 'y' variable for the 'ID's in this file
    - **sample_submission.csv** - a sample submission file in the correct format


## Key Performance Metric
<p align="justify">The primary metric for evaluation of the model is the Coefficient of Determination also called the R-squared. It can be interpreted as a rescaled version of the mean-squared error. In essence it compares the performance of model with mean-model i.e a model that outputs the mean value of the target for all inputs.</p>
<p align="center"><a href="https://www.codecogs.com/eqnedit.php?latex=R-squared&space;=&space;1&space;-&space;\frac{\sum_{i}^N(y_{i}&space;-&space;\hat{y})^2}{\sum_{i}^N(y_{i}&space;-&space;\bar{y})^2}\\&space;\\&space;\text{Where,&space;}&space;\\&space;\hat{y}&space;=&space;\text{&space;Predicted&space;target}\\&space;\bar{y}&space;=&space;\text{&space;Mean&space;target}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?R-squared&space;=&space;1&space;-&space;\frac{\sum_{i}^N(y_{i}&space;-&space;\hat{y})^2}{\sum_{i}^N(y_{i}&space;-&space;\bar{y})^2}\\&space;\\&space;\text{Where,&space;}&space;\\&space;\hat{y}&space;=&space;\text{&space;Predicted&space;target}\\&space;\bar{y}&space;=&space;\text{&space;Mean&space;target}" title="R-squared = 1 - \frac{\sum_{i}^N(y_{i} - \hat{y})^2}{\sum_{i}^N(y_{i} - \bar{y})^2}\\ \\ \text{Where, } \\ \hat{y} = \text{ Predicted target}\\ \bar{y} = \text{ Mean target}" /></a></p>

## Existing Solutions
[Mercedes Benz Greener Manufacturing - Kaggle - Winner's Solution](https://www.kaggle.com/c/mercedes-benz-greener-manufacturing/discussion/37700)
<p align="justify">This blog was written by the person that won the first place solution for this problem 3 years ago. The blog emphasizes that the variations in test - time are attributed to a small set of variables or short paths of a few subprocesses. Interesting findings that were mentioned was that while doing cross-validation for hyper-parameter tuning it was observed that the max_depth parameter for gradient boosted trees to be 2 consistently which led to a conclusion that a few 2 way or 3 way interactions between features was relevant for solving the problem effectively. The final submission was done by ensembling 2 XGBoost models and the predictions were averaged among them. Other improvements suggested included, trying more 2 way or 3 way interactions and applying much more rigorous feature selection methods.</p>

[Mercedes-Benz Greener Manufacturing - Runner Up's Solution](https://www.kaggle.com/c/mercedes-benz-greener-manufacturing/discussion/36390)
<p align="justify">This blog belongs to the second place winner of the problem. It focused more on outliers and noise present in the data and how to deal with them. The author used two approaches to solve the problem, one was a stacked ensemble including sklearn’s GBDTRegressor, RandomForest Regressor and Support Vector Regressor with a reduced set of features. The other approach was to use features obtained from dimensionality reduction techniques like PCA and SVD. The feature selection method used here is also quite innovative. The author called it the Feature Learner. The features were selected based on the CV scores from 5-Fold validation, during which features were randomly bagged from models with a likelihood proportional to the model’s CV score, the feature weights were bagged and if the weights exceeded a certain threshold they were selected as a top feature. The final list of features selected were the top features list for the epoch that had the highest CV score. For removing outliers the samples with ‘y’ > 150 were clipped i.e. removing samples that have a test time greater than 150 seconds.</p>

[Top Voted Public Kernel on Kaggle](https://www.kaggle.com/hakeem/stacked-then-averaged-models-0-5697?scriptVersionId=1252368)
<p align="justify">The main focus of the kernel was applying dimensionality reduction techniques like PCA, Truncated SVD, ICA and Gaussian Random Projections. Only the first 12 components were selected from these methods and were concatenated with the rest of the features. These features were used for XGBoost models along with a Stacked Gradient Boosted Regressor and Cross Validated Linear Regression Model with L1 regularization. The categorical features were encoded with Label encoding instead of One-Hot Encoding. The results from the stacked model and the XGBoost model were later re-weighed and averaged to get the final prediction. This kernel also inspired the 2nd place winner’s solution.</p>
    
[Will Koehrsen's Solution](https://williamkoehrsen.medium.com/capstone-project-mercedes-benz-greener-manufacturing-competition-4798153e2476)
<p align="justify">The blog has experimented with various features and benchmarked the performance, however the final model closely resembles the 2nd place winner’s solution. Here is the architecture of the final model presented in the blog.</p> 
<p align="center"><a href="https://imgur.com/NozG4eD"><img src="https://i.imgur.com/NozG4eD.png" title="source: imgur.com" /></a></p>    
<p align="justify">It also consists of weighted averaging results from a Stacked model made up of Extremely randomized versions of Random Forest Regressor and a Linear Regression Model with both L1 and L2 regularization. The other model being an Extremely Randomized version of Gradient Boosted Regression Tree. The author has also experimented with PCA for dimensionality reduction and selected principal components based on the explained variance score calculated by using the eigenvalues.</p>

[Aditya Pandey's Solution](https://medium.com/analytics-vidhya/mercedes-benz-greener-manufacturing-74a932ae0693)
<p align="justify">The author has carefully researched existing methods of solving the problem and crafted a unique solution. The 2 way and 3 way interaction between features has been adopted from the Winner’s solution along with applying PCA for dimensionality reduction, inspired from the Runner up’s solution. The final model architecture is similar to the one discussed in the earlier section but with only the stacked model consisting of the XGBoost model, RandomForest Regressor model and a Linear Regression model. The meta classifier is a Linear Regression model with L2 regularization. The categorical features are label encoded instead of one hot encoding.


## First Cut Approach & Improvements
- All of the features are either binary or categorical which are not ideal for Linear methods. They need to be handled appropriately by applying One-Hot Encoding or other approaches along with the use of techniques like target encoding or mean encoding, since none of the other solutions implemented it.
- The previous solutions also showed that many features are redundant or have very low variance, hence to reduce the complexity of the final models, experiment with various forms of feature selection methods like Recursive feature selection and the Feature Learner method proposed by the 2nd place Winner’s solution
- The data also contains outliers which have to be taken care of. The key performance metric R-squared can be easily affected by the presence of outliers in train as well as test data. Experiment with simple as well as advanced methods of outlier detection.
- Use of cross validation to find hyperparameters and for feature selection, to make the final solution more robust.
- The correlation and interactions between features also needs a closer look. Will have to experiment to get 2-way and 3-way interactions between features and try to engineer new features from that.
- Since this is a regression problem, this problem would make a good candidate for simple Linear Regression Models. These linear models may not perform well but can act as good base lines for other more advanced methods like tree based models, and ensemble models.</p>

## Exploratory Data Analysis
<p align="justify">The training data shown below has high dimensionality and the features are anonymized hence its not very interpretable</p>
<p align="center"><a href="https://imgur.com/yIOYnVk"><img src="https://i.imgur.com/yIOYnVk.png" title="source: imgur.com" /></a></p>
<p align="justify">The summary stats for the training data can be seen below. The number of categorical variables are 8 and the number of binary variables are 356 along with the 'ID' and target variable 'y'. Around 13 features have zero variance in the training data that can be ignored for the rest of the analysis.</p>    
<p align="center"></a><a href="https://imgur.com/G9yONjd"><img src="https://i.imgur.com/G9yONjd.png" title="source: imgur.com" /></a></p>
<p align="justify">Lets have a closer look at the target variable 'y'. The target variable represents the time (in seconds) spent by a given car sample on bench. The average bench time is around 100 s, the minimum and maximum bench time is around 72 s and 265 s respectively. The meadian and mean value for bench time have a difference of about 1 s which could be due to a few outliers in the higher side that are pulling the mean higher, but the difference is not very significant. However the difference between the 75th percentile and the 100th percentile is significant and needs a closer look.</p>  
<p align="center"><a href="https://imgur.com/JzqEzp4"><img src="https://i.imgur.com/JzqEzp4.png" title="source: imgur.com" /></a></p>
<p align="justify">Lets have a look at the distribution of the target variable in the training data.</p>
<p align="center"><a href="https://imgur.com/M61CO3A"><img src="https://i.imgur.com/M61CO3A.png" title="source: imgur.com" /></a></p>
<p align="justify">- The Distribution is left skewed and has a long tail on the right side, indicated by the high values for Skewness and Kurtosis. The PDF has a long tail on the right which means there are few samples that have extremely large test time. These samples could affect the model training. It can also be seen that there is a huge gap between 99th percentile and 100th percentile value, which also confirms the presence of few outliers in the data. Also peaks can be seen in the distribution at various values of 'y' which indicate multimodal distribution of test time. This is an indication of presence of gropus of similar configurations and features that result in similar test-time.</p>
<p align="center"><a href="https://imgur.com/tDMvQHc"><img src="https://i.imgur.com/tDMvQHc.png" title="source: imgur.com" /></a></p>
<p align="justify">This shows the long tail is mainly due to a single observation with very high bench time of 265 s. There are various methods of handling skewed data as listed below, before that we need to understand skewness and kurtosis</p>

[Top 3 Methods for Handling Skewed Data](https://towardsdatascience.com/top-3-methods-for-handling-skewed-data-1334e0debf45)
<p align="justify">The first plot shows the distribution of our target variable 'y' along with the Skewness and Kurtosis values. Skewness is the measure of distortion from the symmetrical Bell Curve, used to differentiate extreme values on either side of the PDF. Consider a random Variable X with N samples, the Skewness for X is defined as</p>


<p align="center"><a href="https://www.codecogs.com/eqnedit.php?latex=Skewness&space;=&space;\frac{1}{N}&space;\sum_{i}^N&space;[\frac{&space;(x_{i}&space;-&space;\bar{x})}{\sigma_{x}}]^3" target="_blank"><img src="https://latex.codecogs.com/gif.latex?Skewness&space;=&space;\frac{1}{N}&space;\sum_{i}^N&space;[\frac{&space;(x_{i}&space;-&space;\bar{x})}{\sigma_{x}}]^3" title="Skewness = \frac{1}{N} \sum_{i}^N [\frac{ (x_{i} - \bar{x})}{\sigma_{x}}]^3"></a></p>

</p align="justify">Kurtosis is the measure of the tailedness of the distribution, it can be used to measure the outliers present in the data. Kurtosis is defined as,</p>

<p align="center"><a href="https://www.codecogs.com/eqnedit.php?latex=Kurtosis&space;=&space;\frac{1}{N}&space;\sum_{i}^N&space;[\frac{&space;(x_{i}&space;-&space;\bar{x})}{\sigma_{x}}]^4" target="_blank"><img src="https://latex.codecogs.com/gif.latex?Kurtosis&space;=&space;\frac{1}{N}&space;\sum_{i}^N&space;[\frac{&space;(x_{i}&space;-&space;\bar{x})}{\sigma_{x}}]^4" title="Kurtosis = \frac{1}{N} \sum_{i}^N [\frac{ (x_{i} - \bar{x})}{\sigma_{x}}]^4"></a></p>

[Four moments of distribution: Mean, Variance, Skewness, and Kurtosis](http://learningeconometrics.blogspot.com/2016/09/four-moments-of-distribution-mean.html)

- The Skewness for Normal Distribution is 0
- The Kurtosis (Excess Kurtosis) for Normal Distribution is 0
- We applied 3 different types of transforms to reduce the effect of outliers on the model
- The first one was a simple **Square-root Transform** <a href="https://www.codecogs.com/eqnedit.php?latex=(\sqrt{y})" target="_blank"><img src="https://latex.codecogs.com/gif.latex?(\sqrt{y})" title="\sqrt{y}" /></a>
    - This the simplest transform available to reduce the impact of outliers by scaling down the magnitude
    - The reverse transform will be quiet simple as well i.e. <a href="https://www.codecogs.com/eqnedit.php?latex=(\sqrt{y})^2" target="_blank"><img src="https://latex.codecogs.com/gif.latex?(\sqrt{y})^2" title="((\sqrt{y})^2)" /></a>
    - We see a drastic improvement in skewness and kurtosis, however the values are still not ideal
    - **Skewness = 0.7277 | Kurtosis = 3.1711**
    
- The next transformation we applied was the **Log Transform** <a href="https://www.codecogs.com/eqnedit.php?latex=(\log{y})" target="_blank"><img src="https://latex.codecogs.com/gif.latex?(\log{y})" title="(\log{y})" /></a>
    - This is also a monotonous transformation used to reduce the impact of outliers
    - The reverse transform is also straight forward i.e. <a href="https://www.codecogs.com/eqnedit.php?latex=(\text{Antilog[}{\log{y}}\text{]})" target="_blank"><img src="https://latex.codecogs.com/gif.latex?(\text{Antilog[}{\log{y}}\text{]})" title="(\text{Antilog[}{\log{y}}\text{]})" /></a>
    - The skewness and Kurtosis has also imporved drastically compared to the square-root transformation
    - **Skewness = 0.39 | Kurtosis = 1.3095**
    
- The next transformation we applied was the **BoxCox Transform** <a href="https://www.codecogs.com/eqnedit.php?latex=(y=\frac{X^\lambda&space;-&space;1}{\lambda})" target="_blank"><img src="https://latex.codecogs.com/gif.latex?(y=\frac{X^\lambda&space;-&space;1}{\lambda})" title="(y=\frac{X^\lambda - 1}{\lambda})" /></a>
    - The value of lambda is found such that it maximizes the Log-Likelihood function
    - The reverse transform is <a href="https://www.codecogs.com/eqnedit.php?latex=(X&space;=&space;(y&space;*&space;\lambda&space;&plus;&space;1)^\frac{1}{\lambda})" target="_blank"><img src="https://latex.codecogs.com/gif.latex?(X&space;=&space;(y&space;*&space;\lambda&space;&plus;&space;1)^\frac{1}{\lambda})" title="(X = (y * \lambda + 1)^\frac{1}{\lambda})"/></a>
    - **Skewness = -0.0155 | Kurtosis = 0.4019**
 
<p align="center"><a href="https://imgur.com/nHw5JSm"><img src="https://i.imgur.com/nHw5JSm.png" title="source: imgur.com" /></a>
<a href="https://imgur.com/q4o5Q7i"><img src="https://i.imgur.com/q4o5Q7i.png" title="source: imgur.com" /></a></p>

<p align="justify">We're only considering columns which have atleast 2 states i.e. Binary features and categorical features. There were around 10-12 features with only one state throughout the train data, the test data may or may not contain the other states hence it is good to remove such columns. The only continuous variables present are 'ID' and the target variable 'y'. We'll be applying the log transform on the data to correct for skewness and kurtosis. The following features have zero variance in the training data and hence will be dropped.</p>

**X11, X93, X107, X233, X235, X268, X289, X290, X293', X297, X330, X347**
<p align="justify">The dataset contains samples which have the same value for the input variables like X0, X1, .. X385, but with different test time. This could be due to varius reasons, but it is hard to interpret those reasons due to the anonymous nature of the features. These samples could result in contradictory conclusions by our model and hence need to be corrected.The test time for the duplicates can be replaced by mean of all the duplicates, i.e taking the mean of test time for samples with duplicate inputs.</p>

```python
train = pd.read_csv("train.csv")

categorical_columns = list()
binary_columns      = list()
drop_columns        = list()

for col in train.columns:
    if   2 < train[col].nunique() < 50:
        categorical_columns.append(col)
    elif train[col].nunique() == 2:
        binary_columns.append(col)
    elif train[col].nunique() == 1:
        drop_columns.append(col)
        
train = train.groupby(categorical_columns + binary_columns).mean().reset_index()
X = train[categorical_columns + binary_columns]
y = train['y'].apply(lambda x: np.log(x))
```
## Target Encoding
<p align="justify">Target encoding is when the categorical features are encoded with the mean of the target variable. This converts categorical columns to numeric and decreases the cardinality of the data.This helps the model as the categories are numeric, ordinal and hence more interpretable.</p>

```python
import category_encoders as ce
target_encoder = ce.target_encoder.TargetEncoder(cols=categorical_columns)
target_encoder.fit(X, y)

X = target_encoder.transform(X)
```

<p align="justify">Lets have a look at the co-realtion between the categorical features and the target variable 'y'. The Pearson's co-relation coefficient is defined below. A Pearson Coefficient of 1 indicates perfect co-relation between variables whereas a coefficient of 0 indicates no co-relation between the variables X and Y.</p>
<p align="center"><a href="https://www.codecogs.com/eqnedit.php?latex=\text{Pearson's&space;Correlation&space;Coefficient}&space;=&space;\frac{\sum_{i}^N&space;(x_{i}&space;-&space;\bar{x})(y_{i}&space;-&space;\bar{y})}{\sqrt{\sum_{i}^N&space;(x_{i}&space;-&space;\bar{x})^2&space;\sum_{i}^N&space;(y_{i}&space;-&space;\bar{y})^2}}&space;=&space;\frac{\text{Cov(X,Y)}}{\sigma_{x}&space;*&space;\sigma_{y}}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\text{Pearson's&space;Correlation&space;Coefficient}&space;=&space;\frac{\sum_{i}^N&space;(x_{i}&space;-&space;\bar{x})(y_{i}&space;-&space;\bar{y})}{\sqrt{\sum_{i}^N&space;(x_{i}&space;-&space;\bar{x})^2&space;\sum_{i}^N&space;(y_{i}&space;-&space;\bar{y})^2}}&space;=&space;\frac{\text{Cov(X,Y)}}{\sigma_{x}&space;*&space;\sigma_{y}}" title="\text{Pearson's Correlation Coefficient} = \frac{\sum_{i}^N (x_{i} - \bar{x})(y_{i} - \bar{y})}{\sqrt{\sum_{i}^N (x_{i} - \bar{x})^2 \sum_{i}^N (y_{i} - \bar{y})^2}} = \frac{\text{Cov(X,Y)}}{\sigma_{x} * \sigma_{y}}" /></a></p>
<p align="center"><a href="https://imgur.com/Rc2KEa3"><img src="https://i.imgur.com/Rc2KEa3.png" title="source: imgur.com" /></a></p>

<p align="justify">The figure shows the correlation coefficient between the Mean Encoded Categorical Features with the Target Variable 'y'. The feature X0 has the highest correlation with 'y' with a Person's Coefficient of 0.8, followed by the feature X2.</p>

## Recursive Feature Selection

<p align="justify">We'll use Recursive Feature Elimination to extract important features from the dataset in order to understand it better since we can't look at all the 350+ features. According to Dr. Json Brownlee from Machine Learning Mastery, RFE can be best described as,</p>

>Recursive Feature Elimination (RFE) is a wrapper-type feature selection algorithm. This means that a different machine learning algorithm is given and used in the core of the method, is wrapped by RFE, and used to help select features. This is in contrast to filter-based feature selections that score each feature and select those features with the largest (or smallest) score. Technically, RFE is a wrapper-style feature selection algorithm that also uses filter-based feature selection internally. RFE works by searching for a subset of features by starting with all features in the training dataset and successfully removing features until the desired number remains. This is achieved by fitting the given machine learning algorithm used in the core of the model, ranking features by importance, discarding the least important features, and re-fitting the model. This process is repeated until a specified number of features remains.

Dr. Json Brownlee, [Recursive Feature Elimination (RFE) for Feature Selection in Python](https://machinelearningmastery.com/rfe-feature-selection-in-python/) & [Machine Learning Mastery](https://machinelearningmastery.com/) 
- We'll use XGBoost regressor as the model for RFE

[**Pseudo Huber Loss**](https://www.kaggle.com/c/mercedes-benz-greener-manufacturing/discussion/34826)
- Due to the presence of outliers in the data, it would be better to use a more robust loss function in the Recursive Feature Elimination process performed using the XGBoost Regressor Model
- Huber Loss is a robust loss function used in regression that is less sensitive to outliers than Squared Loss.
- The Huber loss is defined as

<p align="center">
    <a href="https://www.codecogs.com/eqnedit.php?latex=\text{Huber&space;Loss}&space;=&space;\left\{\begin{array}{lr}&space;\frac{1}{2}(y&space;-&space;f(x))^2&space;&&space;\text{for&space;}&space;|y&space;-&space;f(x)|\leq&space;\delta,\\&space;\delta&space;(y&space;-&space;f(x))&space;-&space;\frac{1}{2}\delta^2&space;&&space;\text{otherwise&space;}\\&space;\end{array}\right\}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\text{Huber&space;Loss}&space;=&space;\left\{\begin{array}{lr}&space;\frac{1}{2}(y&space;-&space;f(x))^2&space;&&space;\text{for&space;}&space;|y&space;-&space;f(x)|\leq&space;\delta,\\&space;\delta&space;(y&space;-&space;f(x))&space;-&space;\frac{1}{2}\delta^2&space;&&space;\text{otherwise&space;}\\&space;\end{array}\right\}" title="\text{Huber Loss} = \left\{\begin{array}{lr} \frac{1}{2}(y - f(x))^2 & \text{for } |y - f(x)|\leq \delta,\\ \delta (y - f(x)) - \frac{1}{2}\delta^2 & \text{otherwise }\\ \end{array}\right\}" /></a>
</p>

- Delta is a hyperparameter for this loss, it defines the region or the threshold inside which the loss is quadratic outside which the loss is linear.
- However this function is discontinuous at delta and hence not differentiable at delta.
- Instead a smooth approximation of it is used generally called as the Pseudo Huber Loss
- Pseudo Huber Loss is defined as

<p align="center">
    <a href="https://www.codecogs.com/eqnedit.php?latex=\text{Pseudo&space;Huber&space;Loss}&space;=&space;\delta&space;\sqrt{\delta^2&space;&plus;&space;(y&space;-&space;f(x))^2}&space;-&space;\delta^2" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\text{Pseudo&space;Huber&space;Loss}&space;=&space;\delta&space;\sqrt{\delta^2&space;&plus;&space;(y&space;-&space;f(x))^2}&space;-&space;\delta^2" title="\text{Pseudo Huber Loss} = \delta \sqrt{\delta^2 + (y - f(x))^2} - \delta^2" /></a>
</p>

- The derivatives for Huber Loss
<p align="center">
    <a href="https://www.codecogs.com/eqnedit.php?latex=\text{Gradient}&space;=&space;\frac{\delta&space;*&space;(y&space;-&space;f(x))}{(\delta^2&space;&plus;&space;(y&space;-&space;f(x))^2)^{1/2}}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\text{Gradient}&space;=&space;\frac{\delta&space;*&space;(y&space;-&space;f(x))}{(\delta^2&space;&plus;&space;(y&space;-&space;f(x))^2)^{1/2}}" title="\text{Gradient} = \frac{\delta * (y - f(x))}{(\delta^2 + (y - f(x))^2)^{1/2}}" /></a>
</p>

[Reference](https://socratic.org/questions/how-do-you-differentiate-y-sqrt-1-x-2)

<p align="center">
    <a href="https://www.codecogs.com/eqnedit.php?latex=\text{Hessian}&space;=&space;\frac{\delta^3}{(\delta^2&space;&plus;&space;(y&space;-&space;f(x))^2)^{3/2}}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\text{Hessian}&space;=&space;\frac{\delta^3}{(\delta^2&space;&plus;&space;(y&space;-&space;f(x))^2)^{3/2}}" title="\text{Hessian} = \frac{\delta^3}{(\delta^2 + (y - f(x))^2)^{3/2}}" /></a>
</p>

[Reference](https://socratic.org/questions/how-do-you-find-the-derivative-of-x-sqrt-x-2-1)

**Note:** The optimizer for XGBoost Regressor doesn't actually require the Loss, only the gradients for calculating the updates.

```python
import numpy as np
from sklearn.metrics import r2_score, make_scorer
from sklearn.feature_selection import RFECV
from xgboost import XGBRegressor

def gradient(diff: np.ndarray, delta: int = 1) -> np.ndarray:
    '''
    Gradient for the Pseudo Huber Loss
    Args:
        diff  : Vector consisting of difference between Ground Truth of Target and the Predicted Target
        delta : Hyperparameter for the PseudoHuber Loss function. 
                The threshold within which the loss is quadratic in nature,
                outside the thresold the loss is linear
    Returns:
        g     : Gradients for the PsedoHuber Loss
    '''
    g      = delta*diff*(np.power((delta**2 + diff**2),-1/2))
    return g

def hessian(diff: np.ndarray, delta: int =1) -> np.ndarray:
    '''
    Hessian for the Pseudo Huber Loss
    Args:
        diff  : Vector consisting of difference between Ground Truth of Target and the Predicted Target
        delta : Hyperparameter for the PseudoHuber Loss function. 
                The threshold within which the loss is quadratic in nature,
                outside the thresold the loss is linear
    Returns:
        h     : Hessian for the PsedoHuber Loss
    '''
    h      = delta**3*(np.power((delta**2 + diff**2),-3/2))
    return h

def PseudoHuberLoss(y_true: np.ndarray , y_pred: np.ndarray, delta: int = 1) -> (np.ndarray, np.ndarray):
    '''
    Gradient & Hessian for the Pseudo Huber Loss
    Args:
        y_true: Ground truth for target
        y_pred: Predicted target
        delta : Hyperparameter for the PseudoHuber Loss function. 
                The threshold within which the loss is quadratic in nature,
                outside the thresold the loss is linear
    Returns:
        g     : Gradients for the PsedoHuber Loss
        h     : Hessian for the PsedoHuber Loss
    '''
    y_pred = y_pred.ravel()
    diff   = y_pred - y_true
    g      = gradient(diff, delta=delta)
    h      = hessian(diff, delta=delta)
    return g, h

R2_score = make_scorer(r2_score, greater_is_better=True)
est = XGBRegressor(max_depth=3, n_estimators=1000, base_score=y.mean(), objective=PseudoHuberLoss, seed=100, random_state=0)
rfecv = RFECV(estimator=est, step=1, cv=2, verbose=0, n_jobs=2, scoring=R2_score)
rfecv.fit(X, y)
```
[Reference](https://www.kaggle.com/c/mercedes-benz-greener-manufacturing/discussion/34826)

<p align="center"><a href="https://imgur.com/GR1aCpZ"><img src="https://i.imgur.com/GR1aCpZ.png" title="source: imgur.com" /></a></p>

<p align="justify">The optimum number of features selected was 3, **X0, X265 and X47** with the corresponding importance scores as shown in the plot below. There were around 364 features to select from, the Recursive Feature Elimination through Cross Validation has resulted in selection of just 3 important features out of the 364 features. As seen previously from the correlation plot, the feature 'X0' has the most amount of interaction with the target variable 'y'. The Feature X0 has 47 uniques categories in the training data, a further look into these categories will help extract more information. The importance score for 'X0' is significantly higher than the score for feature 'X265' and 'X47'. Features X265 and X47 are Binary Features, Binary Features also need a closer look for extracting more information.</p>

<p align="center"><a href="https://imgur.com/zHI8leu"><img src="https://i.imgur.com/zHI8leu.png" title="source: imgur.com" /></a></p>

### Feature - X0
<p align="justify">The categories aren't distributed uniformly for feature X0. Categories with high count have higher number of outliers.The Categories seem to be ordinal i.e. they seem to have some sort of inherent order in them. The categories from feature 'X0' have the most amount of interaction with the target 'y'When the categories are mean encoded, a monotonously increasing correlation can be seen between the categories of X0 and our target 'y'. The samples belonging to the categories of 'bc' and 'az' are responsible for the peaks at the lower end of the distribution for 'y'. The test time for categories 'bc' and 'az' belong to the lower end of the range for 'y' between 4.3 - 4.5. Whereas the test time for category 'aa' belongs to the higher end of the range for 'y' around 4.8 - 5.0Other categories are tightly bound in the range of 4.5 - 4.8 for log(y).</p>

<p align="center"><a href="https://imgur.com/c8LrRk9"><img src="https://i.imgur.com/c8LrRk9.png" title="source: imgur.com"/></a><a href="https://imgur.com/HRX1wte"><img src="https://i.imgur.com/HRX1wte.png" title="source: imgur.com"/></a><a href="https://imgur.com/5W6X3LE"><img src="https://i.imgur.com/5W6X3LE.png" title="source: imgur.com"/></a></p>

### Binary Features
<p align="justify">There are around 356 Binary features in the datasetOn the left there is the mean test time for 1 and 0 labels respectively for each featureOn the right  there is the count for 1 and 0 labels respectively for each featureThe mean test time for each binary feature is close to the overal average test time, but the features which have very high or very low mean test time usually have highly unbalanced distribution for classes 0 and 1. Majority of the features have a mean close to 100 secFeatures X127 and X314 have quiet a bit of deviation from the overall avegrage with a balance in both 0 and 1 class.</p>

<p align="center"><a href="https://imgur.com/UaRK9Yb"><img src="https://i.imgur.com/UaRK9Yb.png" title="source: imgur.com" /></a></p>

<p align="justify">The plots below shows distribution for Binary features with respect to the target variable "y". Features X47 and X265 have multiple peaks in the distribution.When X47 = 1, the test time is greater than the mean test time. This feature is a good indicator for samples with test time in the upper range of the data.The two peaks in the distribution indicate a bimodal distribution for feature X265. The multimodal nature of the plots indicate groups of similar features or configurations that result in similar test time. The peaks for features X127 and X314 are much more clearly seperable by the mean line.These features will be a good for seperating the bimodal distribution of the test time.</p>

<p align="center"><a href="https://imgur.com/iN0xlFn"><img src="https://i.imgur.com/iN0xlFn.png" title="source: imgur.com" /></a>
<a href="https://imgur.com/rQ8Z1fu"><img src="https://i.imgur.com/rQ8Z1fu.png" title="source: imgur.com" /></a></p>

### Bi-variate Analysis
**X47, X127, X265, X314**
<p align="justify"> The distribution plots for feature interaction between various selected features. The seperation between the peaks of the distribution allows for interesting feature interactions. When X47 = 0, the value for feature X127 has a correlation to the test time, i.e if X127 = 0, the test time is generally higher than the mean, whereas if X127 = 1, the test time is generally lower than the mean test time 'y'. A similar trend is observed when X47 = 1, however the test time for both X127 = 0 & X127 = 1 are generally higher than the mean test time 'y'. Similarly for feature X314 = 0, the value for feature X47 has a correlation to the test time, i.e if X47 = 0, the test time is generally lower than the mean, whereas if X47 = 1, the test time is generally higher than the mean. Similarly for feature X265 = 0, the value for feature X47 has a correlation to the test time, i.e if X314 = 0, the test time is generally lower than the mean, whereas if X314 = 1, the test time is generally higher than the mean. For feature X265 the difference between the two distributions is a lot more pronounced. Even when X265 = 1, the same trend is observed.</p>

<p align="center">
    <a href="https://imgur.com/Fp5Flt7"><img src="https://i.imgur.com/Fp5Flt7.png" title="source: imgur.com" /></a>
    <a href="https://imgur.com/qEVz3T7"><img src="https://i.imgur.com/qEVz3T7.png" title="source: imgur.com" /></a>
    <a href="https://imgur.com/vQaBTif"><img src="https://i.imgur.com/vQaBTif.png" title="source: imgur.com" /></a>
</p>

**X0, X127**
<p align="justify">The most interesting feature interaction is between the Features X0 and X127. The feature X127 creates two seperate groups of configurations for the categories in X0 The seperation can be clearly seen in the plot above. The categories 'ab', 'ac', and 'g' have been merged together since they have the same exact mean. This could carry some inherent meaning about the type of test or the combination of configurations in the testing used. Also it would allow us to bin certain categories to ceratin ranges, this could help in reducing the cardinality of the data. However we could loose information while binning, hence Mean Encoding would be a better approach to reduce cardinality of the data.</p>
<p align="center"><a href="https://imgur.com/vu9Hl7d"><img src="https://i.imgur.com/vu9Hl7d.png" title="source: imgur.com" /></a></p>

**X314 + X315, X127**
<p align="justify">The combination or rather the sum of features X314 & X315 have a similar interaction with feature X127, however the seperations are even more pronounced in this case as seen below.</p>
<p align="center"><a href="https://imgur.com/te0j3Pc"><img src="https://i.imgur.com/te0j3Pc.png" title="source: imgur.com" /></a></p>

## EDA Conclusion 
<p align="justify">The Dataset contains 7 Categorical Features, 356 Binary Features, 1 Continuous Feature and 1 continuous Target variable The Dataset has no missing or Nan values The distribution of y showed a heavily tailed one, presence of outliers was suspected. A log transform has been applied to the target variable to correct the skewness and the tailedness in its distribution. Recursive feature Elimination, resulted in 3 important features -> X0, X265 and X47. Other forms of visualization the Binary features do not have very good variance as a whole but few of them could be useful and important features like X127 and X314. Feature X0 is the most important feature in the task, it is highly correlated with the target variable The X0 categories can roughly be binned into these ranges, if we ignore the per category outlier samples.</p>
<a href="https://imgur.com/8LG6JJb"><img src="https://i.imgur.com/8LG6JJb.png" title="source: imgur.com" /></a>

## Dimensionality Reduction
- There are a lot of features in the dataset, however many features don't have any influence over the target variable.
- Its difficult to know of all the interactions possible between the features that would have an impact on the target variable.
- One way to overcome this problem is to apply dimensionality reduction techniques to the features.
- Dimensionality reduction is different from Feature selection, since the reduced dimensions are a combination of different existing features instead of being a single feature.
- Differernt methods of Dimensionality Reduction used in this case study:  
    - **Principal Component Analysis (PCA):** Aims to find the directions of maximum variance in high-dimensional data and projects it onto a new subspace with equal or fewer dimensions than the original one.  
    - **Independent Component Analysis (ICA):** is a computational method for separating a multivariate signal into additive subcomponents. This is done by assuming that the subcomponents are non-Gaussian signals and that they are statistically independent from each other.   
    - **Truncated Singular Value Decomposition (TSVD):** This transformer performs linear dimensionality reduction by means of truncated singular value decomposition (SVD). Contrary to PCA, this estimator does not center the data before computing the singular value decomposition. This means it can work with sparse matrices efficiently.    
    - **Gaussian Random Projection (GRP):** In random projection, the original d-dimensional data is projected to a k-dimensional (k << d) subspace, using a random *(k x dk x k)*- dimensional matrix R whose columns have unit lengths. Using matrix notation: If X is the original set of N, d-dimensional observations, then X' = R * X is the projection of the data onto a lower k-dimensional subspace. Random projection is computationally simple: form the random matrix "R" and project the *(d x Nd x N)*  data matrix X onto K dimensions of order *(O(dkN))*. If the data matrix X is sparse with about c nonzero entries per column, then the complexity of this operation is of order *(O(ckN))*. In Gaussian random projection, the random matrix R can be generated using a Gaussian distribution.  
    - **Sparse Random Projection (SRP):** Sparse random matrix is an alternative to dense random projection matrix that guarantees similar embedding quality while being much more memory efficient and allowing faster computation of the projected data.
    
```python
import pandas as pd
import numpy as np
import category_encoders as ce
from sklearn.random_projection import GaussianRandomProjection, SparseRandomProjection
from sklearn.decomposition import PCA, FastICA, TruncatedSVD

train = pd.read_csv('train.csv')
train = train.groupby(categorical_columns+binary_columns).mean().reset_index()
if ("X314 + X315" not in train.columns):
    train["X314 + X315"] = train["X314"] + train["X315"]
    binary_columns.append("X314 + X315")
X = train[categorical_columns + binary_columns]
y = train['y'].values

target_encoder = ce.target_encoder.TargetEncoder(cols=categorical_columns + binary_columns)
target_encoder.fit(X, y)

X = target_encoder.transform(X)

scaler = StandardScaler()
scaler.fit(X)

X = pd.DataFrame(scaler.transform(X), columns=categorical_columns + binary_columns)

n_comp = len(categorical_columns + binary_columns) - 1

# PCA
pca = PCA(n_components=n_comp, random_state=420)
pca_train = pca.fit_transform(X)

# ICA
ica = FastICA(n_components=n_comp, random_state=420)
ica_train = ica.fit_transform(X)

# tSVD
tsvd = TruncatedSVD(n_components=n_comp, random_state=420)
tsvd_train = tsvd.fit_transform(X)

# GRP
grp = GaussianRandomProjection(n_components=n_comp, eps=0.1, random_state=420)
grp_train = grp.fit_transform(X)

# SRP
srp = SparseRandomProjection(n_components=n_comp, dense_output=True, random_state=420)
srp_train = srp.fit_transform(X)
```

Next, we'll determine the optimum number of componenets to include with the help of 10-Fold Cross Validation. The n_comp value of 13 was revealed to have the highest R2 Score in the cross Validation.
<p align="center"><a href="https://imgur.com/AV31aXU"><img src="https://i.imgur.com/AV31aXU.png" title="source: imgur.com"/></a></p>

```python
n_comp = 13
# Append decomposition components to datasets
for i in range(1, n_comp + 1):
    X['pca_' + str(i)]  = pca_train[:, i - 1]
    X['ica_' + str(i)]  = ica_train[:, i - 1]
    X['tsvd_' + str(i)] = tsvd_train[:, i - 1]
    X['grp_' + str(i)]  = grp_train[:, i - 1]
    X['srp_' + str(i)]  = srp_train[:, i - 1]

final_features = list(X.columns)
X = X.values
```

## Regression Models
**Linear Regression**
<p align="justify">Now its time we apply some models to the data and evaluate the models. The idea is to start with simple models and try and get the best performance out of it before moving to more complex models. We'll use Cross Validation to get the best hyper-parameters. We'll start with the most well known regression model, Linear Regression.</p>

```python
from sklearn.linear_model import SGDRegressor
sgd = SGDRegressor(tol=1e-5, max_iter=100, random_state=100)

# define search space
space = dict()
space['loss']     = ['squared_loss', 'huber']
space['penalty']  = ['l1', 'l2', 'elasticnet']
space['l1_ratio'] = [x/10 for x in range(1, 11)]
space['alpha']    = [10**x for x in range(-5, 3)]

# define search
sgd = GridSearchCV(sgd, space, scoring=R2_score, cv=5, n_jobs=-1, verbose=1)

# execute search
result = sgd.fit(X, np.log(y))

# summarize result
sgd_params = result.best_params_

y_pred = np.exp(sgd.predict(X))
sgd_score = r2_score(y, y_pred)
```
The following plots show the feature importance score calculated by SGDRegressor and the plot of "ID" vs "y".  
<p align="center"><a href="https://imgur.com/ncp1ie0"><img src="https://i.imgur.com/ncp1ie0.png" title="source: imgur.com" /></a></p>
The Kaggle score for Linear Regression.  
<p align="center"><a href="https://imgur.com/d5Sapgw"><img src="https://i.imgur.com/d5Sapgw.png" title="source: imgur.com" /></a></p>

**XGB Regression**
<p align="justify">Next we'll try some tree based model, the most popular one being XGBoost.</p>

```python
from xgboost import XGBRegressor

xgb = XGBRegressor(tol=1e-5, base_score=np.mean(np.log(y)), random_state=21)

# define search space
space = dict()
space['n_estimators']     = [x*100 for x in range(1, 20)]
space['max_depth']        = [x for x in range(1, 10)] 
space['objective']        = ['reg:squarederror', PseudoHuberLoss]
space['learning_rate']    = loguniform(1e-5, 1)
space['gamma']            = loguniform(1e-5, 1)
space['reg_alpha']        = loguniform(1e-5, 100)
space['reg_lambda']       = loguniform(1e-5, 100)
space['colsample_bytree'] = uniform(0, 1)
space['subsample']        = uniform(0, 1)


# define search
xgb = RandomizedSearchCV(xgb, space, scoring=R2_score, cv=2, n_jobs=-1, verbose=1, n_iter=200, random_state=21)

# execute search
result = xgb.fit(X, np.log(y))

# summarize result
xgb_params = result.best_params_

y_pred = np.exp(xgb.predict(X))
xgb_score = r2_score(y, y_pred)
```
The following plots show the feature importance score calculated by XGBRegressor and the plot of "ID" vs "y".
<p align="center"><a href="https://imgur.com/VgLvx2y"><img src="https://i.imgur.com/VgLvx2y.png" title="source: imgur.com" /></a></p>
The Kaggle score for XGB Regression.
<p align="center"><a href="https://imgur.com/Xt71rW9"><img src="https://i.imgur.com/Xt71rW9.png" title="source: imgur.com" /></a></p>

**Gradient Boosted Regression**
<p align="justify">Next up is the most popular Gradient Boosting Model from scikit-learn, Gradient boosting Regressor.</p>

```python
from sklearn.ensemble import GradientBoostingRegressor

gbr = GradientBoostingRegressor(tol=1e-5,  min_samples_leaf=18, min_samples_split=14, random_state=21)

# define search space
space = dict()
space['loss']          = ['ls', 'lad', 'huber', 'quantile']
space['learning_rate'] = loguniform(1e-5, 1)
space['max_depth']     = [x for x in range(1, 10)] 
space['subsample']     = uniform(0, 1)
space['criterion']     = ['mse', 'mae', "friedman_mse"]
space['n_estimators']  = [x*100 for x in range(2, 11)]
space['max_samples']   = uniform(0,1)
space['ccp_alpha']     = uniform(0,1)
# define search
gbr = RandomizedSearchCV(gbr, space, scoring=R2_score, cv=2, n_iter=40, n_jobs=-1, verbose=1, random_state=21)

# execute search
result = gbr.fit(X, np.log(y))

# summarize result
gbr_params = result.best_params_

y_pred = np.exp(gbr.predict(X))
gbr_score = r2_score(y, y_pred)
```
The following plots show the feature importance score calculated by Gradient Boosting Regressor and the plot of "ID" vs "y".
<p align="center"><a href="https://imgur.com/YVlQlEE"><img src="https://i.imgur.com/YVlQlEE.png" title="source: imgur.com" /></a></p>
The Kaggle score for Gradient Boosting Regression.
<p align="center"><a href="https://imgur.com/XE2WJga"><img src="https://i.imgur.com/XE2WJga.png" title="source: imgur.com" /></a></p>

**Random Forest Regression**
<p align="justify">Next up is the most popular Bagging Model from scikit-learn, Random Forest Regressor.</p>

```python
from sklearn.ensemble import RandomForestRegressor

rfr = RandomForestRegressor(max_depth=None, criterion="mae", min_samples_leaf=18, min_samples_split=14, random_state=999)
rfr.fit(X, np.log(y))

y_pred = np.exp(rfr.predict(X))
rfr_score = r2_score(y, y_pred)
```
The following plots show the feature importance score calculated by Random Forest Regressor and the plot of "ID" vs "y".
<p align="center"><a href="https://imgur.com/1czSvji"><img src="https://i.imgur.com/1czSvji.png" title="source: imgur.com" /></a></p>
The Kaggle score for Random Forest Regression.
<p align="center"><a href="https://imgur.com/n1l8kaj"><img src="https://i.imgur.com/n1l8kaj.png" title="source: imgur.com" /></a></p>

**Stacked Model**
<p align="justify">Finally we'll use a stacked model by utilizing the best parameters from previous models and corss-validation results. We'll use the Ridge Regressor with 10 Fold Cross-Validation as our meta-model. The meta model will become</p>

```python
models = list()
models.append(('xgb', XGBRegressor(tol=1e-7, importance_type="total_cover", base_score=np.mean(np.log(y)), random_state=999,**xgb.best_params_)))
models.append(('gbr', GradientBoostingRegressor(tol=1e-7,  min_samples_leaf=120, min_samples_split=100, random_state=999, **gbr.best_params_)))
models.append(('sgd', SGDRegressor(tol=1e-7, max_iter=100, random_state=100, **sgd.best_params_)))
models.append(('rfr', RandomForestRegressor(max_depth=None, criterion="mae", min_samples_leaf=18, min_samples_split=14, random_state=999)))

# define meta learner model
meta_classifier = RidgeCV(cv=10, scoring=R2_score)

# define the stacking ensemble
model = StackingRegressor(estimators=models, final_estimator=meta_classifier, cv=10)

# fit the model on all available data
model.fit(X, np.log(y))

# make a prediction for one example
y_pred = np.exp(model.predict(X))
```
The following plots show the feature importance score for different models calculated by the meta-model and the plot of "ID" vs "y".
<p align="center"><a href="https://imgur.com/uYVcTgj"><img src="https://i.imgur.com/uYVcTgj.png" title="source: imgur.com" /></a></p>
The Kaggle score for Stacked Model.
<p align="center"><a href="https://imgur.com/CMBiJFb"><img src="https://i.imgur.com/CMBiJFb.png" title="source: imgur.com" /></a></p>

The stacked model could achieve a score similar to the 99th position on the Kaggle Private Leaderboard.
<p align="center"><a href="https://imgur.com/SNZCf5w"><img src="https://i.imgur.com/SNZCf5w.png" title="source: imgur.com" /></a></p>

## Model Comparision
<p align="justify">The stacked model with Target Encoded Categorical Features along with the 13 componenets of various dimensionality reduction techniques like PCA, ICA, TSVD, SRP and GRP had the best performance.</p>
<p align="center"><a href="https://imgur.com/YoBt4TN"><img src="https://i.imgur.com/YoBt4TN.png" title="source: imgur.com" /></a></p>

## Future Work
- We can use other methods of encoding the taret variable like Helmert Encoding, James-Stein Encoding, Count Encoding, M-estimate etc, these are all available on the category_encoder python library
- We tried to get the best performance on this problem out of machine learning solutions, we can improve the performance using much bigger, deep learning based models.

## References
- [Applied AI Course's Machine Learning Course](https://www.appliedaicourse.com/course/11/Applied-Machine-learning-course)
- [Mercedes Benz Greener Manufacturing, Kaggle](https://www.kaggle.com/c/mercedes-benz-greener-manufacturing)
- [Mercedes Benz Greener Manufacturing - Kaggle - Winner's Solution](https://www.kaggle.com/c/mercedes-benz-greener-manufacturing/discussion/37700)
- [Mercedes-Benz Greener Manufacturing - Runner Up's Solution](https://www.kaggle.com/c/mercedes-benz-greener-manufacturing/discussion/36390)
- [Top Voted Public Kernel on Kaggle](https://www.kaggle.com/hakeem/stacked-then-averaged-models-0-5697?scriptVersionId=1252368)
- [Will Koehrsen's Solution](https://williamkoehrsen.medium.com/capstone-project-mercedes-benz-greener-manufacturing-competition-4798153e2476)
- [Aditya Pandey's Solution](https://medium.com/analytics-vidhya/mercedes-benz-greener-manufacturing-74a932ae0693)
- [Top 3 Methods for Handling Skewed Data](https://towardsdatascience.com/top-3-methods-for-handling-skewed-data-1334e0debf45)
- [Four moments of distribution: Mean, Variance, Skewness, and Kurtosis](http://learningeconometrics.blogspot.com/2016/09/four-moments-of-distribution-mean.html)
- [Machine Learning Mastery](https://machinelearningmastery.com/)
- [Recursive Feature Elimination (RFE) for Feature Selection in Python](https://machinelearningmastery.com/rfe-feature-selection-in-python/)  
- [Pseudo Huber Loss](https://www.kaggle.com/c/mercedes-benz-greener-manufacturing/discussion/34826)
- [Pseudo Huber Loss, sklearn implementation]( https://www.kaggle.com/c/mercedes-benz-greener-manufacturing/discussion/34826)
- [Target Encoding vs One-Hot Encoding](https://medium.com/analytics-vidhya/target-encoding-vs-one-hot-encoding-with-simple-examples-276a7e7b3e64#:~:text=From%20the%20documentation%20linked%20above,over%20all%20the%20training%20data.%E2%80%9D)

## Contact
- [Github](https://github.com/chirag4798)
- [Linkedin](https://www.linkedin.com/in/chirag-shetty-85250913a/)
- [Website](https://chirag4798.github.io/)
