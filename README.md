<div align="center">  <h1>
     DSC232R Spring 2024 Group Project<br>
    Exploring Demographic and Socioeconomic Influences on Educational Attainment
  </h1>
</div>

## Table of Contents
- [Introduction](#introduction)
- [Submissions](#submissions)
  - [Milestone 2](#milestone-2)
- [Code](#code)
- [Written Report](#written-report)
- [Final Model, Results Summary, and Github Repository](#final-model-results-summary-and-gibhub-repository)


## Introduction

### Description

This project aims to explore how various demographic and socioeconomic factors, such as geography, income, and age, shape individuals' educational attainment levels. By analyzing a simulated dataset representing diverse populations worldwide and their various characteristics, we aim to identify patterns and classifications using a variety of analytical methods. Our goal is to develop a predictive model that accurately explains the factors contributing to low educational attainment. The findings of this research could inform policy changes and raise awareness about the significant obstacles preventing access to education. Moreover, this study highlights the potential impact on regional economies and societies as education levels rise among the population.

## Submissions

### Milestone 2

#### Data Exploration

##### Getting Started

In order to gain understanding of the data set, we performed various data exploration tasks including visualizations of data distributions and relationships, determining data types, and exploring the size and quality of the data. We found that we have a total of six numerical type variables and nine categorical variables. Below we have a description of the data and the observations that were made. 

### Describe the Distributions 

The data were collected from the 1994 census database.  The data were collected using the following conditions: age is between 16 and 100, the number of hours worked in a week is more than 0, and the final weight is more than 1.

#### Age
The majority of the participants in the dataset are under 50 years old.  The distribution appears to be trimodal with a prominent peak in ages from 16-25, followed by another peak in ages from 40-50.  The third peak is in ages from 80+.  This distribution seems to generally match the distribution of age in the general population. In the general population, 8.7% of people are 19-25 years old, 12.3% of people are 26-34 years old, 25.7% of people are 35-54 years old, 12.9% of people are 55-64 years old, and 17.4% of people are over 65.   

#### Capital Gains
The distribution of capital gains is heavily right skewed.  The majority of people had $0 income from capital gains.  If one had income from capital gains it was most likely under $5,000.  There are a few people that had income from capital gains in excess of $5,000 with a maximum value of approximately $25,000.

#### Capital Loss
Similar to capital gains, most people did not have any capital loss either.  This distribution is bimodal with a peak at 0, indicating most people did not lose any money by the sale of assets.  There is another peak in the data around $2,000 dollars indicating if one did lose money from the sale of assets, it was typically around $2,000.

#### Hours worked per week
The distribution of the number of hours worked is mostly symmetric with the peak at 35 - 39 hours.  In the U.S. women work an average of 36.6 hours per week and men work an average of 40.5 hours per week.  This is likely due to the increase of women that work part-time compared to men.  

#### Work Class
The majority of people are employed by private companies.  After private companies, it is a fairly even distribution of employers between the remaining groups.

#### Education
The majority of people earned a high school education.  The lowest education level achieved was a Preschool education, which occurred more frequently than a doctorate degree.  After a high school degree, the most common highest level of education is some college, followed by 11th grade.   

#### Marital Status
There are several categories that refer to being married, such as married to a civilian, married to a spouse in the armed forces, or married to a spouse that lives a long distance away.  When these categories are combined, marriage is the most common status in this dataset. 

#### Race and Gender
71% of the population in the U.S. identifies as white, 12.2% identify as black, and 5.6% identify as Asian.  This demographic breakdown in this data is similar, as seen  in the bar chart on race.  Females comprise 51.1% of the population in the U.S.  The distribution of gender in this data also shows more females than males. 

#### Income
While income is typically a quantitative variable, in this data income has been categorized into 2 classes, less than or equal to $50,000 or more than $50,000.  Given that the median income in the U.S. in 2022 is $37,585 it is not surprising that the majority of participants earn less than or equal to $50,000 in our dataset as well.  

### Heat Map and Correlation of Quantitative Data

When examining how the quantitative data are correlated, some interesting patterns were observed.  One of the strongest correlations present in the data is as age increases one is more likely to incur capital loss.  As expected, capital loss and capital gains are positively correlated.  In addition, the number of hours worked each week is positively correlated with an increase in capital gains and capital loss.  Indicating that those who work more hours are more likely to have assets to gain and lose money from.  Surprisingly, the number of hours worked per week and level of education are negatively correlated, but it is important to note the correlation is not strong.   


## Preprocessing 

### Preprocessing Plan 
During the data exploration phase, we found that the data collection was already relatively clean. We found no null or missing values for any of the variables, so no imputation or handling of nulls will be needed. During the data exploration we did observe some skewing of distributions, but little evidence of outliers. The issue of skewed distributions can be handled during the preprocessing stage with normalization of the skewed variables. 

We have varying types of data, including a mix of numerical and categorical variables. To properly handle these variables during the modeling process we will perform both scaling of the numerical variables and encoding of the categorical variables. Scaling will ensure that our varying numerical values, like age and capital gains, can be properly compared on an even field. Encoding of the categorical variables will allow us to perform various classification tasks. We have both ordinal categorical variables like education (High School < Associates < Bachelors < Masters), as well as nominal categorical variables like occupation. Ordinal categorical variables can be encoded via mapping or Label Encoding, while the nominal categorical variables can be encoded with one-hot encoding. The varying techniques of encoding the ordinal and nominal categorical variables ensure that the model will not create unsubstantiated relationships between variables. 

### Preprocessing Execution
For this project, we decided to condense the education categories from 16 possible response variables to 7 response variables. This included mapping any education level between pre-school and 12th grade to "Less than High School", and mapping both "Masters" and "Prof-school" to the variable "Master's Degree". In the end, the resulting 7 possible response variables from this mapping were "Less than High School", "High School or GED", "Some College", "Associates Degree", "Bachelor's Degree", "Master's Degree", and "Doctorate".  

Numerical variables for the dataset were ultimately transformed in two ways. First, all numerical variables outlined above (Age, Fnlwgt, EducationNum, CaptialGain, CapitalLoss, and HoursPerWeek) were compiled into a vector and added to the "NumericalDataType" column. Second, numerical variables were scaled using StandardScaler to ensure that all numerical variables were appropriately represented by a normal distribution, which resolves the issue of skewing addressed above. 

Categorical variables for the dataset were ultimately transformed by the deployment of the StringIndexer function. Categorical variables were transformed from string type to numeric type, where indices were assigned according to the label frequencies. In this case, the most frequent labels were given an index of 0.0. This transformation of the categorical variables will allow for proper use and representation in the following model stage of the project. The transformed categorical variables included ['WorkClassIndexed', 'MaritalStatusIndexed', 'OccupationIndexed', 'RelationshipIndexed', 'RaceIndexed', 'SexIndexed', 'NativeCountryIndexed', 'IncomeIndexed']. 

After the preprocessing of the numerical and categorical variables, a final data frame containing the transformed variables was created. In addition, a 60/20/20 train, test, and validation split was performed on the data for use in the model-building stage. The randomSplit function was employed with seed=13. Ultimately, the features of the model-building stage were the categorical variables outlined above in addition to the numerical variable vector in 'NumericalDataTypeTransformed'. The target variable was determined to be 'EducationGroupIndexed', which was also indexed using the StringIndex function to assign numerical index labels to the varying education levels. 

## Creation and Conclusion of First Model
The first model selected was a logistic regression model.  Logistic regression models are classification models used to predict outcomes based on the features of the data. Logistic regression uses the sigmoid function to map a linear combination of the features to a probability score for each class and the class with the highest probability is the predicted class for the sample.  Logistic regression assumes there is a linear relationship between the features and label and independence between the features.  While the correlation analysis on the data did not show a high correlation between the features, some of the features such as age and capital loss, and capital loss and capital gain did show moderate correlation.  

The overall accuracy of the model was low, at 46%.  Some possible reasons for the low accuracy of the model are a non-linear relationship between the features and outcome exists; the data are highly imbalanced, with the majority of samples having a high school education; overfitting, too complex of a model for the data available; missing data; and outliers.  Some possible steps to improve this model would be to reduce the number of features, remove data with outliers, and balance the data by removing some of the samples with a high school education.  However, since it is possible a non-linear relationship exists, we hope to have better success with a random forest model.

## Additional Models
Given the goal to predict education level based on various demographic and socioeconomic factors, such as geography, income, and age, shape individuals' educational attainment levels, a classification model is needed.  Some options for classification models are K-nearest neighbors, SVM, and random forests.  We have selected to use a random forest as the next model to predict education level.  

Random forests are an ensemble method that fits multiple decision tree classifiers to predict a more accurate outcome.  This recursive splitting allows decision trees to capture complex non-linear relationships and the ensemble approach reduces overfitting and improves performance.  Each decision tree is trained on a random subset of features at each split. This randomness helps to decorrelate the trees within the ensemble and can be beneficial when dealing with dependent features. By considering only a subset of features at each split, random forests can mitigate the impact of multicollinearity or dependencies between features.  

The advantages of random forests are there are more options for hyperparameter tuning which could help us to improve the fit of our model, the model is robust to missing or noisy data, and the model is highly scalable. Since random forests do not assume a linear relationship in the data they can model more complex data.  


## Jupyter Notebook

You can find the analysis in the [Jupyter Notebook](https://github.com/allison-conrey/DSC232R-GroupProject/blob/main/Milestone3.ipynb).

## Written Report 
### Introduction 
Socioeconomic status (SES) encompasses an individual's economic and social standing, typically measured by factors like education, income, and occupation.4 Higher SES correlates with better health outcomes, attributed to factors like access to healthcare and healthier lifestyles.3 While SES doesn't directly cause chronic diseases, it reflects environmental and lifestyle influences that contribute to their likelihood.4  Access to quality education throughout life, including college education, is a key component in increasing SES.2  Increased SES plays a pivotal role in improving health by enhancing access to resources and opportunities.  Steady employment decreases the likelihood of poverty and the associated health risks of living in poverty.1  People with college educations are half as likely to be unemployed as their peers who only have a high school degree.5  Those with a bachelor’s degree earn on average 86% more than those with a high school education.5  In fact, college graduates on average make 1.2 million more over their lifetime.5  

Education is essential to the economic growth and development of a nation.7  The economic market is changing everyday and education helps to prepare students for the current and changing job market.7  “Educational attainment is a defining factor in people’s future success, and helps reduce inequality.”8  However, it has been shown that children from low SES families do not have the same access to high-quality schools, technology, extracurricular activities, and tutors.8  Furthermore, children from low SES tend to have poor cognitive development, language, memory, and socio-emotional processing.  These factors contribute to higher dropout rates that perpetuate the low SES intergenerational trap.6,8  

This study aims to explore economic and social factors influencing educational attainment, aiming to inform policies to elevate SES, reduce poverty, and enhance health and well-being.  By analyzing a simulated dataset representing diverse populations worldwide and their various characteristics, we aim to identify patterns and classifications using a variety of analytical methods. Our goal is to develop a predictive model that accurately explains the factors contributing to low educational attainment. The findings of this research could inform policy changes and raise awareness about the significant obstacles preventing access to education. Moreover, this study highlights the potential impact on regional economies and societies as education levels rise among the population.
### Figures
### Methods
### Results
### Discussion Section 
### Conclusion
### Statement of Collaboration 

## Authors

Allison Conrey 
alconrey@ucsd.edu

Konrad Kaim 
kkaim@ucsd.edu 

Filina Nurcahya 
Fnurcahyatjoa@ucsd.edu

Caroline Hutchings
Chutchings@ucsd.edu

Camryn Curtis 
cscurtis@ucsd.edu 



### Data Scheme

root
 |-- Age: float (nullable = true)
 |-- WorkClass: string (nullable = true)
 |-- Fnlwgt: float (nullable = true)
 |-- Education: string (nullable = true)
 |-- EducationNum: float (nullable = true)
 |-- MaritalStatus: string (nullable = true)
 |-- Occupation: string (nullable = true)
 |-- Relationship: string (nullable = true)
 |-- Race: string (nullable = true)
 |-- Sex: string (nullable = true)
 |-- CapitalGain: float (nullable = true)
 |-- CapitalLoss: float (nullable = true)
 |-- HoursPerWeek: float (nullable = true)
 |-- NativeCountry: string (nullable = true)
 |-- Income: string (nullable = true)
