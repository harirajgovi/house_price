# -*- coding: utf-8 -*-
"""
Created on Thu Jan  2 23:57:47 2020

@author: hari4
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Dec 25 15:06:45 2019

@author: hari4
"""
print("Importing libraries and data.....")
# importing libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
from sklearn.model_selection import cross_validate#, GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import math

# importing data
df_train = pd.read_csv("train.csv")
df_test = pd.read_csv("test.csv")

print("Data preprocessing.....")
# splitting datas into categorical and numeric
columns= ["MSSubClass", "OverallQual", "OverallCond", "HalfBath", "KitchenAbvGr",
          "BsmtFullBath", "BsmtHalfBath", "FullBath", "BedroomAbvGr"]
df_train[columns] = df_train[columns].astype(object)
df_test[columns] = df_test[columns].astype(object)
df_train_catgry = df_train[[col for col in df_train.columns if df_train[col].dtype == object]]
df_train_numeric = df_train[[col for col in df_train.columns if df_train[col].dtype != object]]
df_test_catgry = df_test[[col for col in df_test.columns if df_test[col].dtype == object]]
df_test_numeric = df_test[[col for col in df_test.columns if df_test[col].dtype != object]]

# treating categorical datas 
tuple(map(lambda f: df_train_catgry[f].fillna(df_train_catgry[f].mode()[0], inplace=True), 
    df_train_catgry.columns))
tuple(map(lambda f: df_test_catgry[f].fillna(df_test_catgry[f].mode()[0], inplace=True), 
    df_test_catgry.columns))
to_drop = [col for col in df_train_catgry.columns 
           if df_train_catgry[col].value_counts().head(1).values[0]/len(df_train_catgry[col]) > 0.6]
df_train_catgry.drop(columns=to_drop , axis=1, inplace=True)
df_test_catgry.drop(columns=to_drop , axis=1, inplace=True)
to_drop1 = [col for col in df_train_catgry.columns 
            if len(df_test_catgry[col].value_counts()) != len(df_train_catgry[col].value_counts())]
df_train_catgry.drop(columns=to_drop1 , axis=1, inplace=True)
df_test_catgry.drop(columns=to_drop1 , axis=1, inplace=True)
label_encoders = {col:LabelEncoder() for col in df_train_catgry.columns}
for col in df_train_catgry.columns:
    df_train_catgry[col] = label_encoders[col].fit_transform(df_train_catgry[col])
    df_test_catgry[col] = label_encoders[col].transform(df_test_catgry[col])

# treating numeric datas
df_train_numeric.drop(columns="Id" , axis=1, inplace=True)
test_id = df_test_numeric[["Id"]]
df_test_numeric.drop(columns="Id" , axis=1, inplace=True)
tuple(map(lambda f: df_train_numeric[f].fillna(df_train_numeric[f].median(), inplace=True), 
    df_train_numeric.columns))
tuple(map(lambda f: df_test_numeric[f].fillna(df_test_numeric[f].median(), inplace=True), 
    df_test_numeric.columns))
to_drop2 = [col for col in df_train_numeric.columns if df_train_numeric[col].std() == 0
            or len(df_train_numeric[col][df_train_numeric[col] == 0])/len(df_train_numeric[col]) > 0.6]
df_train_numeric.drop(columns=to_drop2 , axis=1, inplace=True)
df_test_numeric.drop(columns=to_drop2 , axis=1, inplace=True)
        
# combining categorical and numerical datas
df_train = pd.concat([df_train_catgry, df_train_numeric], axis=1)
df_test = pd.concat([df_test_catgry, df_test_numeric], axis=1)
target = df_train[["SalePrice"]]
df_train.drop(columns="SalePrice", axis=1, inplace=True)
       
# Standardization 
std_scl = StandardScaler()
df_train_scl = pd.DataFrame(std_scl.fit_transform(df_train), columns = list(df_train.columns))
df_test_scl = pd.DataFrame(std_scl.transform(df_test), columns = list(df_test.columns))
scl_y = StandardScaler()
target_scl = pd.DataFrame(scl_y.fit_transform(target), columns=list(target.columns))

# backward elimination
def backward_elimination(predictors, trgt, sig_lvl):
    
    cols = list(predictors.columns)
    const = pd.DataFrame(np.ones((len(predictors), 1)), columns=["constant"])
    while True:
        x = predictors[cols]
        x = pd.concat([const, x], axis=1)
        model = sm.OLS(trgt, x).fit()
        p = pd.Series(model.pvalues.values[1:], index=cols)
        pmax = max(p)
        rsqr_adj_bfr = model.rsquared_adj.astype(float)
        if pmax > sig_lvl:
            pmax_col = p.idxmax()
            cols.remove(pmax_col)
            x = predictors[cols]
            x = pd.concat([const, x], axis=1)
            model = sm.OLS(trgt, x).fit()
            rsqr_adj_afr = model.rsquared_adj.astype(float)
            if rsqr_adj_afr <= rsqr_adj_bfr:
                cols.append(pmax_col)
                break
        else:
            break
    return cols
    
sl = 0.05
optimal_features = backward_elimination(df_train_scl, target_scl, sl)
df_train_scl = df_train_scl[optimal_features]
df_test_scl = df_test_scl[optimal_features]

# correlatin matrix
corr_matrix = pd.concat([df_train_scl, target_scl], axis=1).corr().abs()
full_corr = corr_matrix["SalePrice"].sort_values(ascending=False)
target_corr = corr_matrix["SalePrice"].sort_values(ascending=False)[corr_matrix["SalePrice"] > 0.5]
sns.heatmap(corr_matrix)
plt.show()

# converting data into arrays
x_train = df_train_scl.iloc[:, :].values
y_train = target_scl.iloc[:, 0].values
x_test = df_test_scl.iloc[:, :].values

# ML model
print("\nTraining XGBR model.....")
regressor = XGBRegressor(max_depth=3, learning_rate=0.088, n_estimators=300, min_child_weight=0, 
                         subsample=0.9, colsample_bytree=0.6, gamma=0, reg_alpha=0.0009,
                         objective="reg:squarederror", n_jobs=-1, random_state=0, verbosity=1, 
                         scale_pos_weight=1)
"""params = {"reg_alpha": [0.0009, 0.001, 0.0008]}
tuner = GridSearchCV(regressor, param_grid=params, scoring="neg_mean_squared_error", 
                     n_jobs=-1, cv=10)
tuned = tuner.fit(x_train, y_train)
best_params = tuned.best_params_
best_score = tuned.best_score_"""
regressor.fit(x_train, y_train)
imp = pd.Series(regressor.feature_importances_, 
                index=df_train_scl.columns).sort_values(ascending=False)
train_pred = regressor.predict(x_train)
test_pred = regressor.predict(x_test)

#estimation
estimators = {"abs_error":"neg_mean_absolute_error", 
              "squared_error":"neg_mean_squared_error", 
              "r2_score":"r2"}
cross_val = cross_validate(regressor, x_train, y_train, 
                           scoring=estimators, cv=10, n_jobs=-1, return_train_score=True)
print("\nmean_absolute_error:{}".format(round(abs(cross_val["test_abs_error"]).mean()), 3), 
      "\nmean_squared_error:{}".format(round(abs(cross_val["test_squared_error"]).mean(), 3)),
      "\nroot_mean_squared_error:{}".format(round(math.sqrt(abs(cross_val["test_squared_error"]).mean()), 3)),
      "\nr2_score:{}".format(round(cross_val["test_r2_score"].mean(), 3)))
mae = round(mean_absolute_error(y_train, train_pred), 3)
mse = round(mean_squared_error(y_train, train_pred), 3)
rmse = round(math.sqrt(mse), 3)
r2 = round(r2_score(y_train, train_pred), 3)
print("\nMAE_train:{}".format(mae),
      "\nMSE_train:{}".format(mse),
      "\nRMSE_train:{}".format(rmse),
      "\nr2_score_train:{}".format(r2))

y_pred_train = scl_y.inverse_transform(train_pred)
y_pred_test = pd.DataFrame(scl_y.inverse_transform(test_pred), columns=list(target.columns))
fin_result = pd.concat([test_id, y_pred_test], axis=1)
#fin_result.to_csv("submission46.csv", encoding="utf-8", index=False)
