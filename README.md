# DSC232R Spring 2024 Group Project 

# Exploring Demographic and Socioeconomic Influences on Educational Attainment

## Description

This project aims to explore how various demographic and socioeconomic factors, such as geography, income, and age, shape individuals' educational attainment levels. By analyzing a simulated dataset representing diverse populations worldwide and their various characteristics, we aim to identify patterns and classifications using a variety of analytical methods. Our goal is to develop a predictive model that accurately explains the factors contributing to low educational attainment. The findings of this research could inform policy changes and raise awareness about the significant obstacles preventing access to education. Moreover, this study highlights the potential impact on regional economies and societies as education levels rise among the population.

## Getting Started

## Preprocessing 
In order to prepare data for further analysis, we will compose all numerical columns into a vector and then scale using Spark's StandardScaler. This will ensure their distributions are close to normal distribution and would allow for better analysis later. Additionally, we will encode all categorical variables represented currently as a strings using Spark's SparkLabelEncoder giving every string label appropriate numerical value that can be used easily later.

## Authors





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
