# %reset -f

import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
from sklearn import tree, model_selection, metrics, datasets, linear_model, preprocessing
import seaborn as sns
import re
from scipy.stats import chi2_contingency

#### Data Import and Cleaning ####
data = pd.read_csv('Nigeria.csv')
region_data = pd.read_csv('recode_region.csv')
region_dict = {}
for i in range(len(region_data)):
    region_dict[region_data['clean_adm1'][i]] = region_data['clean_adm1 recoded'][i]

## Remove unused attributes
useless_attributes = ['row_id','source','clean_country_name','clean_adm2','clean_adm3',
                      'activity_id','scheme_id', 'wpdx_id', 'notes', 'orig_lnk',
                      'photo_lnk','country_id', 'data_lnk', 'public_data_source',
                      'latest_record', 'location_id','lat_deg_original',
                      'lon_deg_original','country_name','adm2', 'adm3',
                      'clean_country_id','management','created_timestamp', 
                      'updated_timestamp', 'New Georeferenced Column ',
                      'lat_lon_deg', 'count', 'converted','water_source',
                      'water_tech', 'adm1','report_date','status','subjective_quality']

for i in useless_attributes:
    data.drop(columns = i, inplace = True)

## Remove columns with too many missing data
nulldata = data.isnull().sum()
missing_cols = nulldata[nulldata > len(data)/2].index

for i in missing_cols:
    data.drop(columns = i, inplace = True)
    
## Recategorize data
# Region
data['region'] = data['clean_adm1'].apply(lambda x: region_dict[x])
data.drop(columns = 'clean_adm1',inplace = True)

# Pay
data['pay'] = data['pay'].apply(lambda x: x if x == 'No' else 'Yes')

# Multivariate Analysis
conti_var = ['lat_deg', 'lon_deg','distance_to_primary_road',
             'distance_to_secondary_road', 'distance_to_tertiary_road',
             'distance_to_city', 'distance_to_town','served_population',
             'local_population_1km', 'crucialness', 'pressure']

data_sample = data.sample(n = 1000)
data_sample.dropna(inplace = True)
data_cont = data_sample[conti_var]
names = data_cont.columns

target = data_sample['status_id'].apply(lambda x: 1 if x == 'Yes' else 0)
colors = ['red','green']
y_color = [colors[i] for i in target]
r2_mat = np.zeros(len(conti_var)**2).reshape(len(conti_var),-1)
r2_mat = pd.DataFrame(data = r2_mat, index = conti_var, columns = conti_var)

# for i in range(len(names)):
#     for j in range(len(names)):
#         ind = i*len(names) + j + 1
#         plt.subplot(len(names),len(names),ind)
#         if i == j:
#             plt.xlim(0,1)
#             plt.ylim(0,1)
#             plt.text(0.2,0.2,names[i],fontsize = 10)
#         else:
#             lin_reg = linear_model.LinearRegression()
#             lin_reg.fit(data_cont[[names[j]]].values,data_cont[names[i]].values)
#             plt.scatter(data_cont[names[j]],data_cont[names[i]],color = 'blue')
#             y_pred = lin_reg.predict(data_cont[[names[j]]].values)
#             plt.plot(data_cont[names[j]],y_pred,color = 'red')
#             r_square = metrics.explained_variance_score(data_cont[names[i]].values,y_pred)
#             r2_mat.iloc[i,j] = r_square
#             plt.text(min(data_cont[names[j]]) + 0.2*(max(data_cont[names[j]])-min(data_cont[names[j]])),
#                      max(data_cont[names[i]]) - 0.2*(max(data_cont[names[i]])-min(data_cont[names[i]])),
#                      "R^2: " + str(r_square), fontsize = 20, c = 'red')

# High correlation between pressure and served_population - Drop pressure
data.drop(columns = 'pressure', inplace = True)
# data.dropna(subset = ['served_population','local_population_1km','crucialness'])

## Chi-Square Test
# Build Contingency Table

cat_var = ['water_source_clean','water_source_category', 'water_tech_clean',
           'water_tech_category','facility_type', 'install_year',
           'management_clean', 'pay','usage_capacity', 'is_urban',
           'cluster_size', 'region']

## Cramers'V Coefficient

# Drop highly correlated cat_var:
data['water_tech_category'] = data['water_tech_category'].apply(lambda x: x if x == 'Hand Pump' else 'Mechanized Pump & Tapstand')
def management_recode(x):
    if x == 'School Management' or x == 'Health Care Facility':
        return 'Community Management'
    elif x == 'Private Operator/Delegated Management' or x == 'Other Institutional Management':
        return 'Other'
    else:
        return x
data['management_clean'] = data['management_clean'].apply(lambda x: management_recode(x))
data['usage_capacity'] = data['usage_capacity'].apply(lambda x: "1000" if x == 1000 else "250&500")
data['cluster_size'] = data['cluster_size'].apply(lambda x: str(x) if x <= 2 else ">=3")
data.drop(columns = 'water_source_clean', inplace = True)
data.drop(columns = 'water_tech_clean', inplace = True)
data.drop(columns = 'facility_type', inplace = True)
data.drop(columns = 'install_year', inplace = True)
data.dropna(subset = ['served_population','local_population_1km','crucialness'],
            inplace = True)
data.fillna('Missing',inplace = True)

# Binarize Data
cat_var1 = ['water_source_category', 'water_tech_category','management_clean', 
            'pay','usage_capacity', 'is_urban','cluster_size', 'region']
conti_var1 = ['lat_deg', 'lon_deg','distance_to_primary_road',
             'distance_to_secondary_road', 'distance_to_tertiary_road',
             'distance_to_city', 'distance_to_town','served_population',
             'local_population_1km', 'crucialness']
dummies = pd.get_dummies(data[cat_var1],drop_first = True)

## Data for fitting model
x = pd.concat([dummies,data[conti_var1]],axis = 1).values
y = data['status_id'].apply(lambda x: 1 if x == 'No' else 0).values

# Scaling data
scl = preprocessing.StandardScaler()
x = scl.fit_transform(x)

# Split data
x_train, x_test, y_train, y_test = model_selection.train_test_split(x,y,test_size = 0.3,
                                                                    random_state = 2021)

#### Logistic Regression Model
## Base Model (l2 penalty)
lr_base = linear_model.LogisticRegression(random_state = 2021,
                                          solver = 'lbfgs')

lr_base.fit(x_train, y_train)

y_pred = lr_base.predict(x_test)

print(metrics.confusion_matrix(y_test,y_pred))
print(metrics.accuracy_score(y_test,y_pred))
print(metrics.precision_score(y_test,y_pred))
print(metrics.recall_score(y_test,y_pred))

## Log Model (No penalty)
lr_base = linear_model.LogisticRegression(random_state = 2021,
                                          penalty = 'none')

lr_base.fit(x_train, y_train)

y_pred = lr_base.predict(x_test)

print(metrics.confusion_matrix(y_test,y_pred))
print(metrics.accuracy_score(y_test,y_pred))
print(metrics.precision_score(y_test,y_pred))
print(metrics.recall_score(y_test,y_pred))

## Log Model (l1)
lr_base = linear_model.LogisticRegression(random_state = 2021,
                                          solver = 'saga',
                                          penalty = 'l1',
                                          max_iter=1000)

lr_base.fit(x_train, y_train)

y_pred = lr_base.predict(x_test)

print(metrics.confusion_matrix(y_test,y_pred))
print(metrics.accuracy_score(y_test,y_pred))
print(metrics.precision_score(y_test,y_pred))
print(metrics.recall_score(y_test,y_pred))

## Log Model (Elastic Net)
lr_base = linear_model.LogisticRegression(random_state = 2021,
                                          solver = 'saga',
                                          penalty = 'elasticnet',
                                          l1_ratio=0.5,
                                          max_iter=1000)

lr_base.fit(x_train, y_train)

y_pred = lr_base.predict(x_test)

print(metrics.confusion_matrix(y_test,y_pred))
print(metrics.accuracy_score(y_test,y_pred))
print(metrics.precision_score(y_test,y_pred))
print(metrics.recall_score(y_test,y_pred))