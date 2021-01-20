# Mercedes-Benz Greener Manufacturing
Can you cut the time a Mercedes-Benz spends on the test bench?

<a href="https://imgur.com/Pl6iv1X"><img src="https://i.imgur.com/Pl6iv1X.jpg" title="source: imgur.com" /></a>

## Problem Statement
<p align="justify">An automobile has various components like, engine, chassis, steering, suspension, transmission etc which work together in tandem to provide the driving experience. These systems can be thought of as building blocks of an automobile that has to be fit together to form a vehicle. These building blocks can have variations in them in accordance with the purpose they’re being used for. While building a vehicle the manufacturer has to take into account the different ways these components fit and interact with each other, for which the vehicle has to be tested rigorously before its sent on road. The time it takes to test the performance, comfort, reliability and safety of a vehicle is highly correlated with the type of component it’s made of. Testing cars is expensive hence an entry level passenger car does not undergo the same criteria for testing as a high end luxury car, depending on the components and the type of car the test time may vary. To accurately estimate the test time required one has to account for each and every configuration of the components and their interaction, which would be a complex and time consuming process if done non-algorithmically, hence the need for a machine learning solution.</p>

## Dataset
<p align="justify">A dataset has been prepared by the engineers in Daimler to tackle this problem. The dataset contains permutations of car features, configurations and different testing methods applied with their respective time required for testing, the features are anonymized i.e. the name of the features are not interpretable. The dataset mainly consists of binary and categorical data, where the binary data represents the type of test being carried out and the categorical data represents the features of the vehicle. The goal of this case study is to provide a model that can estimate the time with the least amount of error. 
    
[Data Source](https://www.kaggle.com/c/mercedes-benz-greener-manufacturing)

- This dataset contains an anonymized set of variables, each representing a custom feature in a Mercedes car. For example, a variable could be 4WD, added air suspension, or a head-up display.

- The ground truth is labeled ‘y’ and represents the time (in seconds) that the car took to pass testing for each variable.

**File descriptions**
- Variables with letters are categorical. Variables with 0/1 are binary values.

    - **train.csv** - the training set
    - **test.csv** - the test set, one has to predict the 'y' variable for the 'ID's in this file
    - **sample_submission.csv** - a sample submission file in the correct format
</p>

## Key Performance Metric
<p align="justify">The primary metric for evaluation of the model is the Coefficient of Determination also called the R-squared. It can be interpreted as a rescaled version of the mean-squared error. In essence it compares the performance of model with mean-model i.e a model that outputs the mean value of the target for all inputs.</p>
<p align="center"><a href="https://www.codecogs.com/eqnedit.php?latex=R-squared&space;=&space;1&space;-&space;\frac{\sum_{i}^N(y_{i}&space;-&space;\hat{y})^2}{\sum_{i}^N(y_{i}&space;-&space;\bar{y})^2}\\&space;\\&space;\text{Where,&space;}&space;\\&space;\hat{y}&space;=&space;\text{&space;Predicted&space;target}\\&space;\bar{y}&space;=&space;\text{&space;Mean&space;target}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?R-squared&space;=&space;1&space;-&space;\frac{\sum_{i}^N(y_{i}&space;-&space;\hat{y})^2}{\sum_{i}^N(y_{i}&space;-&space;\bar{y})^2}\\&space;\\&space;\text{Where,&space;}&space;\\&space;\hat{y}&space;=&space;\text{&space;Predicted&space;target}\\&space;\bar{y}&space;=&space;\text{&space;Mean&space;target}" title="R-squared = 1 - \frac{\sum_{i}^N(y_{i} - \hat{y})^2}{\sum_{i}^N(y_{i} - \bar{y})^2}\\ \\ \text{Where, } \\ \hat{y} = \text{ Predicted target}\\ \bar{y} = \text{ Mean target}" /></a></p>

<a href="https://imgur.com/Xqg0HB5"><img src="https://i.imgur.com/Xqg0HB5.png" title="source: imgur.com" /></a>

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


## First Cut Approach
- All of the features are either binary or categorical which are not ideal for Linear methods. They need to be handled appropriately by applying One-Hot Encoding or other approaches.
- The previous solutions also showed that many features are redundant or have very low variance, hence to reduce the complexity of the final models, experiment with various forms of feature selection methods like Recursive feature selection and the Feature Learner method proposed by the 2nd place Winner’s solution
- The data also contains outliers which have to be taken care of. The key performance metric R-squared can be easily affected by the presence of outliers in train as well as test data. Experiment with simple as well as advanced methods of outlier detection.
- Use of cross validation to find hyperparameters and for feature selection, to make the final solution more robust.
- The correlation and interactions between features also needs a closer look. Will have to experiment to get 2-way and 3-way interactions between features and try to engineer new features from that.
- Since this is a regression problem, this problem would make a good candidate for simple Linear Regression Models. These linear models may not perform well but can act as good base lines for other more advanced methods like tree based models, and ensemble models.</p>

