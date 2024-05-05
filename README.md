# DSC232R Spring 2024 Group Project 

# Exploring Demographic and Socioeconomic Influences on Educational Attainment

## Description

This project aims to explore how various demographic and socioeconomic factors, such as geography, income, and age, shape individuals' educational attainment levels. By analyzing a simulated dataset representing diverse populations worldwide and their various characteristics, we aim to identify patterns and classifications using a variety of analytical methods. Our goal is to develop a predictive model that accurately explains the factors contributing to low educational attainment. The findings of this research could inform policy changes and raise awareness about the significant obstacles preventing access to education. Moreover, this study highlights the potential impact on regional economies and societies as education levels rise among the population.

## Getting Started

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
When examining how the quantitative data are correlated, some interesting patterns were observed.  One of the strongest correlations present in the data is as age increases one is more likely to incur capital loss.  As expected, capital loss and capital gains are positively correlated.  In addition, the number of hours worked each week is positively correlated with an increase in capital gains and capital loss.  Indicating that those that work more hours are more likely to have assets to gain and lose money from.  Surprisingly, the number of hours worked per week and level of education are negatively correlated, but it is important to note the correlation is not strong.   

## Preprocessing 
During the data exploration phase we found that the data collection is already relatively clean. We found no null or missing values for any of the variables, so no imputation or handling of nulls will be needed during the preprocessing. During the data exploration we did observe some skewing of distributions, but little evidence of outliers. The issue of skewed distributions can be handled during the preprocessing stage with normalization of the skewed variables. 

We have varying types of data, including a mix of numerical and categorical variables. In order to properly handle these variables during the modeling process we will perform both scaling of the numerical variables and encoding of the categorical variables. Scaling will ensure that our varying numerical values, like age and capital gains, can be properly compared on an even field. Encoding of the categorical variables will allow us to perform various classification tasks. We have both ordinal categorical variables like education (High School < Associates < Bachelors < Masters), as well as nominal categorical variables like occupation. Ordinal categorical variables can be encoded via mapping or Label Encoding, while the nominal categorical variables can be encoded with one-hot encoding. The varying techniques of encoding the ordinal and nominal categorical variables ensures that the model will not create unsubstantiated relationships between variables. 


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
