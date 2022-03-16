import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import RandomizedSearchCV
#######################################################################################################################
dat=pd.read_csv('Soloff_IA.csv', sep=" " )

#######################################################################################################################
#######################################################################################################################
dat=np.array(dat)
X=dat[0:800,0:4]
Y=dat[0:800,4:7]

######################################################
###############1st meta-model#########################
model = RandomForestRegressor(n_estimators=1000, min_samples_leaf=1, min_samples_split=2, random_state=1, max_features='sqrt',max_depth=20,bootstrap='true')
model.fit(X,Y)

######################################################
############## TEST ##########################
X2=dat[800:1000,0:4]
Y2=dat[800:1000,4:7]

predictions = model.predict(X2)
print(mean_squared_error(predictions, Y2))
######################################################
###############hyperparameter tuning##################

# # Number of trees in random forest
# n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
# # Number of features to consider at every split
# max_features = ['auto', 'sqrt']
# # Maximum number of levels in tree
# max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
# max_depth.append(None)
# # Minimum number of samples required to split a node
# min_samples_split = [2, 5, 10]
# # Minimum number of samples required at each leaf node
# min_samples_leaf = [1, 2, 4]
# # Method of selecting samples for training each tree
# bootstrap = [True, False]# Create the random grid
# random_grid = {'n_estimators': n_estimators,
#                 'max_features': max_features,
#                 'max_depth': max_depth,
#                 'min_samples_split': min_samples_split,
#                 'min_samples_leaf': min_samples_leaf,
#                 'bootstrap': bootstrap}

# rf_random = RandomizedSearchCV(estimator = model, param_distributions = random_grid, n_iter = 100, cv = 3, verbose=2, random_state=42, n_jobs = -1)
# rf_random.fit(X, Y)

# predictions2= rf_random.predict(X2)
# print(mean_squared_error(predictions2, Y2))

#################################
def evaluate(model, test_features, test_labels):
    predictions = model.predict(test_features)
    errors = abs(predictions - test_labels)
    mape = 100 * np.mean(errors / np.max(test_labels))
    accuracy = 100 - mape
    print('Model Performance')
    print('Average Error: {:0.4f} degrees.'.format(np.mean(errors)))
    print('Accuracy = {:0.2f}%.'.format(accuracy))

    return accuracy
# best_random = rf_random.best_estimator_
random_accuracy = evaluate(model, X, Y)
# random_accuracy = evaluate(best_random, X, Y)

# Results of hyperparamtertuning
#rf_random.best_params_
#{'n_estimators': 1000,
# 'min_samples_split': 2,
# 'min_samples_leaf': 1,
# 'max_features': 'sqrt',
# 'max_depth': 20,
# 'bootstrap': True}