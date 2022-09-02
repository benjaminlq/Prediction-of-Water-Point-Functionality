# %reset -f

import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
from sklearn import tree, model_selection, metrics, datasets, linear_model, preprocessing
import seaborn as sns
import re
from scipy.stats import chi2_contingency

####### Data Import and Cleaning #######
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
x_train, x_test, y_train, y_test = model_selection.train_test_split(x,y,test_size = 0.4,
                                                                    random_state = 2021)

#### Decision Tree Model

# ####### Pre-Pruning (Minimum Leaves) #######
# tree_perf = []
# for i in range(10,101,10):
#     tree_clf = tree.DecisionTreeClassifier(criterion = 'gini',
#                                            min_samples_leaf = i,
#                                            random_state = 2021)
#     tree_clf.fit(x_train, y_train)
#     # plt.figure(figsize = (40,40))
#     # tree.plot_tree(tree_clf,filled = True)
    
#     y_pred = tree_clf.predict(x_test)
#     # print(metrics.confusion_matrix(y_test,y_pred))
#     # print(metrics.accuracy_score(y_test,y_pred))
#     # print(metrics.precision_score(y_test,y_pred))
#     # print(metrics.recall_score(y_test,y_pred))
#     tree_perf.append([metrics.accuracy_score(y_test,y_pred),
#                       metrics.precision_score(y_test,y_pred),
#                       metrics.recall_score(y_test,y_pred)])

# print(tree_perf)
# perf = np.array(tree_perf)
# plt.figure(figsize = (15,9))
# for i in range(3):
#     plt.subplot(1,3,i+1)
#     plt.plot(np.arange(10,101,10),perf[:,i])

# ####### Pre-Pruning (Minimum split) #######
# tree_perf2 = []
# for i in range(10,101,10):
#     tree_clf = tree.DecisionTreeClassifier(criterion = 'gini',
#                                            min_samples_split = i,
#                                            random_state = 2021)
#     tree_clf.fit(x_train, y_train)
#     # plt.figure(figsize = (40,40))
#     # tree.plot_tree(tree_clf,filled = True)
    
#     y_pred = tree_clf.predict(x_test)
#     # print(metrics.confusion_matrix(y_test,y_pred))
#     # print(metrics.accuracy_score(y_test,y_pred))
#     # print(metrics.precision_score(y_test,y_pred))
#     # print(metrics.recall_score(y_test,y_pred))
#     tree_perf2.append([metrics.accuracy_score(y_test,y_pred),
#                       metrics.precision_score(y_test,y_pred),
#                       metrics.recall_score(y_test,y_pred)])

# print(tree_perf2)
# perf2 = np.array(tree_perf2)
# plt.figure(figsize = (15,9))
# for i in range(3):
#     plt.subplot(1,3,i+1)
#     plt.plot(np.arange(10,101,10),perf2[:,i])

# ###### Pre-Pruning (Max Depth) #######
# tree_perf3 = []
# for i in range(5,31,5):
#     tree_clf = tree.DecisionTreeClassifier(criterion = 'gini',
#                                            max_depth= i,
#                                            random_state = 2021)
#     tree_clf.fit(x_train, y_train)
#     # plt.figure(figsize = (40,40))
#     # tree.plot_tree(tree_clf,filled = True)
    
#     y_pred = tree_clf.predict(x_test)
#     # print(metrics.confusion_matrix(y_test,y_pred))
#     # print(metrics.accuracy_score(y_test,y_pred))
#     # print(metrics.precision_score(y_test,y_pred))
#     # print(metrics.recall_score(y_test,y_pred))
#     tree_perf3.append([metrics.accuracy_score(y_test,y_pred),
#                       metrics.precision_score(y_test,y_pred),
#                       metrics.recall_score(y_test,y_pred)])

# print(tree_perf3)
# perf3 = np.array(tree_perf3)
# plt.figure(figsize = (15,9))
# for i in range(3):
#     plt.subplot(1,3,i+1)
#     plt.plot(np.arange(5,31,5),perf3[:,i])
    


# ###### Post Pruning (Cost Complexity Penalty) ###########
# ## Fit Model (Full Tree)
# tree_clf = tree.DecisionTreeClassifier(criterion = 'gini',
#                                         random_state = 2021)


# tree_clf.fit(x_train, y_train)
# # plt.figure(figsize = (40,40))
# # tree.plot_tree(tree_clf,filled = True)
# tree_obj = tree_clf.tree_

# y_pred = tree_clf.predict(x_test)
# print(metrics.confusion_matrix(y_test,y_pred))
# print(metrics.accuracy_score(y_test,y_pred))
# print(metrics.precision_score(y_test,y_pred))
# print(metrics.recall_score(y_test,y_pred))

# ## Post Pruning (Using test_data)
# path = tree_clf.cost_complexity_pruning_path(x_train, y_train)
# alphas = path['ccp_alphas']

# accuracy_train, accuracy_test = [], []
# precision_train, precision_test = [], []
# recall_train, recall_test = [], []
# for alpha in alphas[0:100]:
#     tree_ccp = tree.DecisionTreeClassifier(criterion = 'gini',
#                                            random_state = 2021,
#                                            ccp_alpha=alpha)
#     tree_ccp.fit(x_train,y_train)
#     y_train_pred = tree_ccp.predict(x_train)
#     y_pred = tree_ccp.predict(x_test)
    
#     accuracy_train.append(metrics.accuracy_score(y_train,y_train_pred))
#     recall_train.append(metrics.recall_score(y_train,y_train_pred))
    
#     accuracy_test.append(metrics.accuracy_score(y_test,y_pred))
#     recall_test.append(metrics.recall_score(y_test,y_pred))

# ## Accuracy vs Alpha
# plt.figure(figsize = (30,10))
# plt.plot(alphas,accuracy_train,c = 'orange',label = 'training')
# plt.plot(alphas,accuracy_test, c = 'blue', label = 'testing')
# plt.ylim(0.75,0.82)
# plt.xlim(0,0.005)
# plt.title('Accuracy vs Alpha')
# plt.xlabel('Alpha')
# plt.ylabel('Accuracy')
# plt.legend()

# ## Recall vs Alpha
# plt.figure(figsize = (30,10))
# plt.plot(alphas,recall_train,c = 'orange',label = 'training')
# plt.plot(alphas,recall_test, c = 'blue', label = 'testing')
# plt.title('Recall vs Alpha')
# plt.ylim(0.4,0.8)
# plt.xlim(0,0.005)
# plt.xlabel('Alpha')
# plt.ylabel('Recall')
# plt.legend()

# ## Alpha = 0.0005
# pruned_tree = tree.DecisionTreeClassifier(random_state = 2021,
#                                          ccp_alpha = 0.0005)
# pruned_tree.fit(x_train, y_train)
# y_pred = pruned_tree.predict(x_test)
# print(metrics.confusion_matrix(y_test,y_pred))
# print(metrics.accuracy_score(y_test,y_pred))
# print(metrics.precision_score(y_test,y_pred))
# print(metrics.recall_score(y_test,y_pred))

# ## Alpha = 0.001
# pruned_tree1 = tree.DecisionTreeClassifier(random_state = 2021,
#                                          ccp_alpha = 0.001)
# pruned_tree1.fit(x_train, y_train)
# y_pred = pruned_tree1.predict(x_test)
# print(metrics.confusion_matrix(y_test,y_pred))
# print(metrics.accuracy_score(y_test,y_pred))
# print(metrics.precision_score(y_test,y_pred))
# print(metrics.recall_score(y_test,y_pred))

####### Grid-Search CV (Pre-Pruning) #######
## Fit Model (Full Tree) ##
tree_clf = tree.DecisionTreeClassifier(criterion = 'gini',
                                        random_state = 2021)


tree_clf.fit(x_train, y_train)
# plt.figure(figsize = (40,40))
# tree.plot_tree(tree_clf,filled = True)
tree_obj = tree_clf.tree_

y_pred = tree_clf.predict(x_test)
print(metrics.confusion_matrix(y_test,y_pred))
print(metrics.accuracy_score(y_test,y_pred))
print(metrics.precision_score(y_test,y_pred))
print(metrics.recall_score(y_test,y_pred))

## Grid-Search CV (Pre-Pruning) ##
params = {'max_depth':list(range(5,31,5)),
          'min_samples_split':list(range(10,101,10)),
          'min_samples_leaf':list(range(10,101,10))}

gcv = model_selection.GridSearchCV(tree_clf, params,
                                   n_jobs = 4)
# gcv.fit(x_train, y_train)


####### Grid-Search CV (Post-Pruning) #######
## Fit Model (Full Tree) ##
tree_clf = tree.DecisionTreeClassifier(criterion = 'gini',
                                        random_state = 2021)

tree_clf.fit(x_train, y_train)

y_pred = tree_clf.predict(x_test)
y_pred_train = tree_clf.predict(x_train)

print('Training Set results:')
print(metrics.confusion_matrix(y_train,y_pred_train))
print('Accuracy Score:',metrics.accuracy_score(y_train,y_pred_train))
print('Precision Score:',metrics.precision_score(y_train,y_pred_train))
print('Recall Score:',metrics.recall_score(y_train,y_pred_train))

print('\nTesting Set results:')
print(metrics.confusion_matrix(y_test,y_pred))
print('Accuracy Score:',metrics.accuracy_score(y_test,y_pred))
print('Precision Score:',metrics.precision_score(y_test,y_pred))
print('Recall Score:',metrics.recall_score(y_test,y_pred))

alphas = tree_clf.cost_complexity_pruning_path(x_train,y_train)['ccp_alphas']
print('\n',len(alphas))

params = {'ccp_alpha':alphas}
gcv_ccp = model_selection.GridSearchCV(tree_clf,params,n_jobs = 4)
gcv_ccp.fit(x_train, y_train)

tree_postprune_cv = gcv_ccp.best_estimator_
tree_postprune_cv.fit(x_train, y_train)

y_pred = tree_postprune_cv.predict(x_test)
y_pred_train = tree_postprune_cv.predict(x_train)

print('Training Set results:')
print(metrics.confusion_matrix(y_train,y_pred_train))
print('Accuracy Score:',metrics.accuracy_score(y_train,y_pred_train))
print('Precision Score:',metrics.precision_score(y_train,y_pred_train))
print('Recall Score:',metrics.recall_score(y_train,y_pred_train))

print('\nTesting Set results:')
print(metrics.confusion_matrix(y_test,y_pred))
print('Accuracy Score:',metrics.accuracy_score(y_test,y_pred))
print('Precision Score:',metrics.precision_score(y_test,y_pred))
print('Recall Score:',metrics.recall_score(y_test,y_pred))