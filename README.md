# <center> Prediction of Water Point Functionality </center>



![image](https://user-images.githubusercontent.com/99384454/188149456-101445fe-6efa-4c36-bfe7-924fad606700.png)

## TABLE OF CONTENTS
  - [Business Settings](#business-settings)
  - [Explanatory Data Analysis (EDA)](#explanatory-data-analysis-eda)
  - [Preprocessing and Feature Engineering](#preprocessing-and-feature-engineering)
  - [Model Building and Optimization](#model-building-and-optimization)
  - [Model Evaluation](#model-evaluation)
  - [Conclusion](#conclusion)
  - [References](#references)

## Business Settings
The objective of this study is to build a predictive model to predict Non-Functional water points in Nigeria based on predictors such as water management records, population data, geographical location of the water points, etc. The prediction outcomes will help Nigerian government and private water supply companies in planning to rehabilitate Non-functional water points.

## Explanatory Data Analysis (EDA)
### 1. Univariate Distribution
#### 1.1. Continuous Variables
![image](https://user-images.githubusercontent.com/99384454/188153495-3001dc60-29bf-42be-af18-86d46395a1fb.png)
#### 1.2. Categorical Variables
![image](https://user-images.githubusercontent.com/99384454/188159488-a7d29fa2-ced9-4058-bf21-6f992aa4b602.png)

### 2. Multivariate Analysis
#### 2.1. Peason Correlation
![image](https://user-images.githubusercontent.com/99384454/188153586-782d9c42-b82c-44b7-838a-d10124c903b0.png) <br>
Based on multivariate analysis, **Served Population** and **Pressure** have high Pearson correlation, hence I excluded **Pressure** as this feature has 10% of missing data.

#### 2.2. VIF factor
VIF Factor is another method to validate the multicollinearity issue between variables. The advantage of VIF Factor is that it considers not just pairwise, but also a combination of more than 2 features collinearity issues.
![image](https://user-images.githubusercontent.com/99384454/188155830-af34c762-4946-4dd2-a08a-8a0bf2129599.png)
Based on the VIF factor, consistent with Pearson pairwise correlation, **Pressure** and **Served Population** have the highest VIF factor, exceeding the pre-defined threshold of **VIF = 8**. After removing **Pressure**, the VIF Factor of **Served Population** dropped to **2.4**.

#### 2.3. Chi-Square Statistics
![image](https://user-images.githubusercontent.com/99384454/188157777-1a90cd05-6954-40c9-820e-b92deb43ba13.png)
I dropped columns **"water_source_clean", "water_tech_clean", "facility_type", "install_year"** due to high degree of association (Cramer's V Coefficient > 0.8) with at least another feature inside the dataset.

## Preprocessing and Feature Engineering

## Model Building and Optimization

## Model Evaluation

## Conclusion

## References
- https://www.waterpointdata.org
- https://www.waterpointdata.org/wp-content/uploads/2021/04/WPDx_Data_Standard.pdf
