from get_data import *
from calc_stats import *
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import numpy as np
import random

def get_features(team):
    avg_score = random.randrange(60,90) # RANDOM
    avg_win_pct = random.randrange(5,9) #RANDOM
    return [avg_score, avg_win_pct/10]

def classify(matchup, training_data):
    
    feature_names = list(training_data.columns.values)
    
    SS_features = get_features(matchup['strongseed'])
    WS_features= get_features(matchup['weakseed'])
    
    test = pd.DataFrame([SS_features + WS_features], columns=feature_names[:4]) #4 features
    
    training_data.head()
    features = training_data.columns[:4] #4 features
    clf = RandomForestClassifier(n_jobs=2)
    y, _ = pd.factorize(training_data['winner'])
    clf.fit(training_data[features], y)
    
    prob_preds = clf.predict_proba(test[features])
    preds = clf.predict(test[features])[0]

    winner = matchup['strongseed'] if preds == 0 else matchup['weakseed']
    prob = prob_preds[0][preds]
    
    return winner, prob

def run_bracket(tourney_slots_s, RF_data):       
	tourney_slots_s['winner'] = 0
	tourney_slots_s['prob'] = 0
	for num in range(1,5):#len(tourney_slots_s)):
	    matchup = tourney_slots_s.ix[num]
	    round_1_flag = True
	    if list(matchup['strongseed'])[0] == 'R':
	        team_A = tourney_slots_s[(tourney_slots_s.slot == matchup['strongseed'])]['winner']
	        team_B = tourney_slots_s[(tourney_slots_s.slot == matchup['weakseed'])]['winner']

	        l_team_A = list(list(team_A)[0])
	        i_team_A = int(l_team_A[1]+l_team_A[2])
	        
	        l_team_B = list(list(team_B)[0])
	        i_team_B = int(l_team_A[1]+l_team_B[2])

	        if i_team_A < i_team_B:
	            matchup['strongseed'] = list(team_A)[0]
	            matchup['weakseed'] = list(team_B)[0]
	        else:
	            matchup['strongseed'] = list(team_B)[0]
	            matchup['weakseed'] = list(team_A)[0]
	    
	    winner, prob = classify(matchup, RF_data)
	    matchup['prob'] = prob

	    if (round_1_flag):
	        matchup['winner'] = winner
	    else:
	        matchup['winner'] = list(winner)[0]
	        
	    tourney_slots_s.ix[num] = matchup

def cross_validation(training_data):
    training_data['is_train'] = np.random.uniform(0, 1, len(RF_data)) <= .75
    training_data.head()
     
    train, test = training_data[training_data['is_train']==True], training_data[training_data['is_train']==False]
     
    features = training_data.columns[:4]
    clf = RandomForestClassifier(n_jobs=2)
    y, _ = pd.factorize(train['winner'])
    clf.fit(train[features], y)
    
    target_names = np.array(['strongseed', 'weakseed'])
    preds = target_names[clf.predict(test[features])]
    cross_tab = pd.crosstab(test['winner'], preds, rownames=['actual'], colnames=['preds'])
    return cross_tab