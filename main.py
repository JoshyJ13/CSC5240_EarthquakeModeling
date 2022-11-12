import matplotlib_inline

from pathlib import Path

import numpy as np
import pandas as pd
##
import matplotlib.pyplot as plt
import seaborn as sns
import seaborn.axisgrid

#preprocessing data
from sklearn.preprocessing import StandardScaler
#model
from sklearn.ensemble import RandomForestClassifier
#combining the preprocess with model training
from sklearn.pipeline import make_pipeline
#optimizing hyperparamters of pipeline
from sklearn.model_selection import GridSearchCV
#computing f1 score
from sklearn.metrics import f1_score

train_values = pd.read_csv('train_values.csv', index_col='building_id')
train_labels = pd.read_csv('train_labels.csv', index_col='building_id')

#(train_labels.damage_grade.value_counts().sort_index().plot.bar(title="Number of buildings with Each Damage Grade"))

selected_features = ['foundation_type',
                     'area_percentage',
                     'height_percentage',
                     'count_floors_pre_eq',
                     'land_surface_condition',
                     'has_superstructure_cement_mortar_stone']

train_values_subset = train_values[selected_features]

#sns.pairplot(train_values_subset.join(train_labels), hue='damage_grade')

train_values_subset = pd.get_dummies(train_values_subset)

pipe = make_pipeline(StandardScaler(), RandomForestClassifier(random_state=2018))

param_grid = {'randomforestclassifier__n_estimators': [50,100],
              'randomforestclassifier__min_samples_leaf': [1,5]}

gs = GridSearchCV(pipe, param_grid, cv = 5)

gs.fit(train_values_subset.head(500), train_labels.head(500).values.ravel())

print(gs.best_params_)

in_sample_preds = gs.predict(train_values_subset)

f1 = f1_score(train_labels, in_sample_preds, average='micro')
print("F1 Score: " + str(f1))



# f1 = (2 * P-micro * Rmicro) / P-micro + R-micro

# also referred as f1 = (2 * (Precision * Recall)) / (Precision + Recall)

# Precision = Correct positive predictions relative to total positive predictions
# P = TP / (TP + FP)

# Recalls = Correct positive predictions relative to total actual positives
# Recalls = TP / (TP + FN)
