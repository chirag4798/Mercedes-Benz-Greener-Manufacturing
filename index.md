# Mercedes-Benz Greener Manufacturing
Can you cut the time a Mercedes-Benz spends on the test bench?

<a href="https://imgur.com/Pl6iv1X"><img src="https://i.imgur.com/Pl6iv1X.jpg" title="source: imgur.com" /></a>

## Problem Statement
<p align="justify">An automobile has various components like, engine, chassis, steering, suspension, transmission etc which work together in tandem to provide the driving experience. These systems can be thought of as building blocks of an automobile that has to be fit together to form a vehicle. These building blocks can have variations in them in accordance with the purpose they’re being used for. While building a vehicle the manufacturer has to take into account the different ways these components fit and interact with each other, for which the vehicle has to be tested rigorously before its sent on road. The time it takes to test the performance, comfort, reliability and safety of a vehicle is highly correlated with the type of component it’s made of. Testing cars is expensive hence an entry level passenger car does not undergo the same criteria for testing as a high end luxury car, depending on the components and the type of car the test time may vary. To accurately estimate the test time required one has to account for each and every configuration of the components and their interaction, which would be a complex and time consuming process if done non-algorithmically, hence the need for a machine learning solution.</p>

## Dataset
<p align="justify">A dataset has been prepared by the engineers in Daimler to tackle this problem. The dataset contains permutations of car features, configurations and different testing methods applied with their respective time required for testing, the features are anonymized i.e. the name of the features are not interpretable. The dataset mainly consists of binary and categorical data, where the binary data represents the type of test being carried out and the categorical data represents the features of the vehicle. The goal of this case study is to provide a model that can estimate the time with the least amount of error. The primary metric for evaluation of the model is the Coefficient of Determination also called the R-squared. It can be interpreted as a rescaled version of the mean-squared error. In essence it compares the performance of model with mean-model i.e a model that outputs the mean value of the target for all inputs.

- This dataset contains an anonymized set of variables, each representing a custom feature in a Mercedes car. For example, a variable could be 4WD, added air suspension, or a head-up display.

- The ground truth is labeled ‘y’ and represents the time (in seconds) that the car took to pass testing for each variable.

**File descriptions**
- Variables with letters are categorical. Variables with 0/1 are binary values.

    - **train.csv** - the training set
    - **test.csv** - the test set, you must predict the 'y' variable for the 'ID's in this file
    - **sample_submission.csv** - a sample submission file in the correct format
</p>

## Key Performance Metric
<p align="center"><a href="https://www.codecogs.com/eqnedit.php?latex=R-squared&space;=&space;1&space;-&space;\frac{\sum_{i}^N(y_{i}&space;-&space;\hat{y})^2}{\sum_{i}^N(y_{i}&space;-&space;\bar{y})^2}\\&space;\\&space;\text{Where,&space;}&space;\\&space;\hat{y}&space;=&space;\text{&space;Predicted&space;target}\\&space;\bar{y}&space;=&space;\text{&space;Mean&space;target}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?R-squared&space;=&space;1&space;-&space;\frac{\sum_{i}^N(y_{i}&space;-&space;\hat{y})^2}{\sum_{i}^N(y_{i}&space;-&space;\bar{y})^2}\\&space;\\&space;\text{Where,&space;}&space;\\&space;\hat{y}&space;=&space;\text{&space;Predicted&space;target}\\&space;\bar{y}&space;=&space;\text{&space;Mean&space;target}" title="R-squared = 1 - \frac{\sum_{i}^N(y_{i} - \hat{y})^2}{\sum_{i}^N(y_{i} - \bar{y})^2}\\ \\ \text{Where, } \\ \hat{y} = \text{ Predicted target}\\ \bar{y} = \text{ Mean target}" /></a></p>

<a href="https://imgur.com/Xqg0HB5"><img src="https://i.imgur.com/Xqg0HB5.png" title="source: imgur.com" /></a>
