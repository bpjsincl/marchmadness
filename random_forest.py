from get_data import *
from calc_stats import *
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import numpy as np


def classify(df):

	df['is_train'] = np.random.uniform(0, 1, len(df)) <= .75
	df.head()
	 
	train, test = df[df['is_train']==True], df[df['is_train']==False]
	 
	features = df.columns[:4]
	clf = RandomForestClassifier(n_jobs=2)
	y, _ = pd.factorize(train['winner'])
	clf.fit(train[features], y)
	 
	prob_preds = clf.predict(test[features])
	preds = clf.predict(test[features])
	pd.crosstab(test['winner'], preds, rownames=['actual'], colnames=['preds'])


	

