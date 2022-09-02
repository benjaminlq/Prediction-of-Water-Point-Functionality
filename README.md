# <center> Prediction of Water Point Functionality </center>



![image](https://user-images.githubusercontent.com/99384454/188149456-101445fe-6efa-4c36-bfe7-924fad606700.png)

## TABLE OF CONTENTS
  - [Business Settings](#business-settings)
  - [Explanatory Data Analysis (EDA)](#explanatory-data-analysis-eda)
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

## Model Building and Optimization
### 1. Logistic Regression
```
lr_base = linear_model.LogisticRegression(random_state = 2021,
                                          solver = 'lbfgs')
lr_base.fit(x_train, y_train)
y_pred = lr_base.predict(x_test)

lr_best_coef = pd.DataFrame(lr_base.coef_[0], index = column_labels, columns = ['Coefficient'])
lr_best_coef['magnitude'] = abs(lr_best_coef)
lr_best_coef.sort_values(by = 'magnitude', ascending = False)[:10]
```
![image](https://user-images.githubusercontent.com/99384454/188166577-207a5512-7be1-464d-ab08-52b614736fd4.png)

### 2. Decision Tree
Based on the base fully grown decision tree, I performed greedy Post-Pruning procedure using Cost Complexity Pruning, which assign a cost for each additional node inside a tree. The more nodes there are, the higher penalty that the tree receives while calculating cost function. The purpose of this procedure is to reduce the effect of overfitting while reducing the chance that we may potentially miss out important branch which may occur when we perform Pre-Pruning. This method, however, incur significantly longer training time, as we prune backward and considered all possible trees in a greedy manner.
![image](https://user-images.githubusercontent.com/99384454/188168071-d0776592-17e8-467d-adb2-7f60254b9a59.png) <br>
![image](https://user-images.githubusercontent.com/99384454/188168112-9d9bbbb4-4348-4e5e-821c-9c9582fc0c92.png) <br>
![image](https://user-images.githubusercontent.com/99384454/188168139-e18ff21f-ed78-47ce-939d-0d09a521e7cc.png) <br>

I find the alpha value (**0.000192**) giving the best CV score and use this alpha score to retrain the entire tree.

### 3. Naive Bayes Classifier
#### GaussianNB for Continuous Variables
```
from sklearn import naive_bayes as nb

x_cont = conti_data
x_cont_train, x_cont_test, y_train, y_test = model_selection.train_test_split(x_cont,y,test_size = 0.4,
                                                                    random_state = 2021)
# Continuous
gnb = nb.GaussianNB()
gnb.fit(x_cont_train, y_train)
cont_pred_log_prob = gnb.predict_log_proba(x_cont_test)
cont_train_log_prob = gnb.predict_log_proba(x_cont_train)
```
#### CategoricalNB for Categorical Variables
```
x_cat_train, x_cat_test, y_train, y_test = model_selection.train_test_split(dummies.values,y,test_size = 0.4,
                                                                    random_state = 2021)

# Categorical
cnb = nb.CategoricalNB(alpha = 1)
cnb.fit(x_cat_train,y_train)
cat_pred_log_prob = cnb.predict_log_proba(x_cat_test)
cat_train_log_prob = cnb.predict_log_proba(x_cat_train)
```
#### Combining CNB and GNB
```
# Combine
pred_prob = cat_pred_log_prob + cont_pred_log_prob
train_prob = cat_train_log_prob + cont_train_log_prob
y_pred = 1*(pred_prob[:,1] >= pred_prob[:,0])
y_train_nb = 1*(train_prob[:,1] >= train_prob[:,0])
```

### Ensemble Models
#### AdaBoost
```
ada = ensemble.AdaBoostClassifier(base_estimator = tree.DecisionTreeClassifier(criterion = 'gini', random_state = 2021),                                       
                                 n_estimators = 100, algorithm = 'SAMME.R', random_state = 2021)          
ada.fit(x_train, y_train)
```
#### XgBoost
```
xgb = xgboost.XGBClassifier(n_estimators = 100, learning_rate = 1.0, n_jobs = 4, random_state = 2021)
xgb.fit(x_train, y_train)
```

## Model Evaluation
|      | Naive Bayes | Log Reg (L1) | Log Reg (L2)| Log Reg (Elastic) | Tree (Pre-Pruned) | Tree (CCP-Pruned)| Tree Bagging | RandomForest| Adaboost| Xgboost|
|------|------|------|------|------|------|------|------|------|------|------|
|Accuracy|0.7429|0.7971|0.7968|0.7975|0.7985|0.8039|0.8218|0.8318|0.8094|0.8072|
|Precision|0.5891|0.7279|0.7257|0.7281|0.7012|0.7003|0.7838|0.7780|0.7252|0.7074|
|Recall|0.5857|0.5599|0.5619|0.5613|0.6189|0.6513|0.5937|0.6461|0.6277|0.6532|
|F1-Score|0.5874|0.6330|0.6334|0.6339|0.6575|0.6749|0.6756|0.7059|0.6729|0.6792|
|AUC|0.700|0.7324|0.7327|0.7330|0.7496|0.7623|0.7596|0.7812|0.7598|0.7652|

## Conclusion
In this exercise, we built several Machine Learning models to predict the water point functionality in Nigeria using Naive Bayes classifiers, Logistic Regression Models, Decision Tree Models and Ensemble Models. The result for the models can be summarized in the table below. RandomForest has the highest performance (**83.2%**) whereas Naive Bayes classifier yields the worst performance based on test dataset.

## References
- https://www.waterpointdata.org
- https://www.waterpointdata.org/wp-content/uploads/2021/04/WPDx_Data_Standard.pdf
