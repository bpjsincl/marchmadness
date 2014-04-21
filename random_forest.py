from get_data import *
from calc_stats import *
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import numpy as np
import random

def classify(matchup):
    # df['is_train'] = np.random.uniform(0, 1, len(df)) <= .75
    # df.head()
     
    # train, test = df[df['is_train']==True], df[df['is_train']==False]
     
    # features = df.columns[:4]
    # clf = RandomForestClassifier(n_jobs=2)
    # y, _ = pd.factorize(train['winner'])
    # clf.fit(train[features], y)
     
    # prob_preds = clf.predict(test[features])
    # preds = clf.predict(test[features])
    # pd.crosstab(test['winner'], preds, rownames=['actual'], colnames=['preds'])

    rand = random.randrange(0,2)
    if rand == 0:
        seed = matchup[2]
        prob = random.randrange(10,90)
    else:
        seed = matchup[3]
        prob = random.randrange(10,90)
#     print seed
    return seed, prob

def run_bracket(tourney_slots_s):       
	tourney_slots_s['winner'] = 0
	tourney_slots_s['prob'] = 0
	for num in range(1,len(tourney_slots_s)):
	    matchup = tourney_slots_s.ix[num]
	    round_1_flag = True
	    if list(matchup['strongseed'])[0] == 'R':
	        print 'in this loooop'
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
	    
	    winner, prob = classify(matchup)
	    matchup['prob'] = prob

	    if (round_1_flag):
	        matchup['winner'] = winner
	    else:
	        matchup['winner'] = list(winner)[0]
	        
	    tourney_slots_s.ix[num] = matchup
