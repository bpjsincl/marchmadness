from get_data import *
from calc_stats import *
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import numpy as np
import random

def get_features(team_A, team_B, year):
    ''' returns the features matchup of each team in a given year'''
    teamA = teams_map[teams_map['id']==Tseeds_map[str(team_A)]]['team_name']
    teamB = teams_map[teams_map['id']==Tseeds_map[str(team_B)]]['team_name']    
    fs = da.make_features_matchup(teams_mod, teamA.iloc[0], teamB.iloc[0], year)
    return fs

def classify(matchup, training_data, year):
    '''classifies a matchup against training data for a given year
        returns the predicted winning team and the probability of that team winning'''
    feature_names = list(training_data.columns.values)
    features = get_features(matchup['strongseed'], matchup['weakseed'], year)
#     SS_features = get_features(matchup['strongseed'])
#     WS_features= get_features(matchup['weakseed'])
    
    test = pd.DataFrame(features, columns=feature_names[:60]) #60 features
    training_data.head()
    features = training_data.columns[:60] #60 features
    clf = RandomForestClassifier(n_jobs=2, n_estimators=15, max_depth=5)
    y, _ = pd.factorize(training_data['value'])
    clf.fit(training_data[features], y)
    
    prob_preds = clf.predict_proba(test[features])
    preds = clf.predict(test[features])
    
    winner = matchup['strongseed'] if preds[0] == 0 else matchup['weakseed']
    prob = prob_preds[0][0] if preds[0] == 0 else prob_preds[0][1]
    return winner, prob
    
def run_bracket(tourney_slots_s, RF_data, year):
    ''' runs a bracket and based on classification and actual results builds each round
        modifies the tourney_slots_s data structure'''
    tourney_slots_s['winner'] = 0
    tourney_slots_s['prob'] = 0
    for num in range(0,len(tourney_slots_s)):
        matchup = tourney_slots_s.iloc[num]
        round_1_flag = True
        if list(matchup['strongseed'])[0] == 'R':
            team_A = tourney_slots_s[(tourney_slots_s.slot == matchup['strongseed'])]['winner']
            team_B = tourney_slots_s[(tourney_slots_s.slot == matchup['weakseed'])]['winner']
            matchup.loc['strongseed'] = list(team_B)[0]
            matchup.loc['weakseed'] = list(team_A)[0]
        
        winner, prob = classify(matchup, RF_data, year)
        matchup.loc['prob'] = prob
        matchup.loc['winner'] = winner

        tourney_slots_s.iloc[num] = matchup

def cross_validation(training_data):
    '''sets 70% of training data to be training and 30% to be testing
        returns a confusion matrix that can be used to calculate training accuracy'''
    training_data['is_train'] = np.random.uniform(0, 1, len(RF_data)) <= .70
    training_data.head()
     
    train, test = training_data[training_data['is_train']==True], training_data[training_data['is_train']==False]
     
    features = training_data.columns[:4]
    clf = RandomForestClassifier(n_jobs=2, n_estimators=15, max_depth=5)
    y, _ = pd.factorize(train['value'])
    clf.fit(train[features], y)
    
    target_names = np.array(['strongseed', 'weakseed'])
    preds = target_names[clf.predict(test[features])]
    cross_tab = pd.crosstab(test['value'], preds, rownames=['actual'], colnames=['preds'])
    return cross_tab

def tournament_accuracy():
    '''returns the accuracy of a tournament based on number of games predicted correctly
        divided by the total number of games in tourney_slots_s'''
    num_right = 0
    total = 0
    tourney_slots_s['actual'] =0
    for num in range(0,len(tourney_slots_s)):
        matchup = tourney_slots_s.iloc[num]
        
        actual_result = df_results[df_results[0]==matchup['slot']]
        matchup.loc['actual'] = actual_result.iloc[0][1]
        tourney_slots_s.iloc[num] = matchup
        if actual_result.iloc[0][1] == matchup['winner']:
            num_right += 1
        total += 1
    
    return num_right/total

def ll_classify(features, training_data, year):
    '''classifies an entire tournament for log loss'''
    test = pd.DataFrame(features, columns=feature_names[:60]) #60 features
    training_data.head()
    features = training_data.columns[:60] #60 features
    clf = RandomForestClassifier(n_jobs=2, n_estimators=15, max_depth=5)
    y, _ = pd.factorize(training_data['value'])
    clf.fit(training_data[features], y)
    
    prob_preds = clf.predict_proba(test[features])
#     preds = clf.predict(test[features])
#     winner = matchup['strongseed'] if preds[0] == 0 else matchup['weakseed']
#     prob = prob_preds[0][0] if preds[0] == 0 else prob_preds[0][1]
    return prob_preds

def log_loss_RF():
    '''calculates the log loss for the random forest'''
    tourney_data = pd.DataFrame(tournament, columns=feature_names[:60])
    ll_probs = []
    a = len(tourney_data)
    print a
    for num in range(0,len(tourney_data)):
        print num
        matchup = tourney_data.iloc[num]
        features = [[]]
        for i in range(0, len(matchup)):
           features[0].append(matchup[i])
        pd.DataFrame(features)
        probs = ll_classify(features, RF_data, year)
        ll_probs.append(probs[0].tolist())
    return ll_probs

# results is straight from create_tournmanet
# probs is the list of generated probabilities: [(p1, p2), (p1, p2), ...]
def log_loss(results, probs):
    res = results[results.res.notnull()]
    results['p1'],results['p2'] = np.array(probs)[:,0], np.array(probs)[:,1]
    probs = results.ix[results.res.notnull(), ['p1','p2']].values.tolist()
    ll = metrics.log_loss(res.res.tolist(), probs)
    
    return ll      

if __name__ == '__main__':
	# INITIALIZE ALL DATA FOR RANDOM FOREST

	year = 2010

	col = ['years' , 'letter']
	years = [[2013,'R'],[2012,'Q'],[2011,'P']]
	pd_years = pd.DataFrame(years, columns=col)
	letter_year = pd_years[pd_years['years']==2013]['letter'][0]

	main_folder = "/Users/matthewchong/Documents/SCHOOL/SYDE 4B/SYDE 522/marchmadness/data_files/"
	all_seasons = list(gd.get_seasons(main_folder)['season'])
	#     last_season = all_seasons[0]
	#     curr_season = all_seasons[0]
	teams_map = get_teams(main_folder)
	curr_season = letter_year

	tourney_seeds = gd.get_tourney_seeds(main_folder)
	tourney_slots = gd.get_tourney_slots(main_folder)
	tourney_results = gd.get_tourney_results(main_folder)

	tourney_results_s = tourney_results.ix[tourney_results['season'] == curr_season]
	T_matchup_results = tourney_results_s[['wteam', 'lteam']]
	tourney_slots_s = tourney_slots.ix[tourney_slots['season'] == curr_season]

	teams_in_tourney = single_tourney_teams(curr_season,tourney_seeds)
	tourney_stats_dict = teams_season_stats(teams_in_tourney, curr_season, main_folder) #(reg season avg_score, reg season win/loss %)
	Tseeds_map = tourney_seeds_dict(tourney_seeds, curr_season) #get mapping of teams to seed in tournament

	last_winners_dict, tourney_match_seeds, all_game_outcomes = construct_tourney_winners(tourney_slots_s, T_matchup_results, Tseeds_map)
	tournament_champion = Tseeds_map[last_winners_dict['R6CH']]

	NN_input_vec = create_features(all_game_outcomes, tourney_stats_dict, Tseeds_map)

	teams = da.get_teams()
	cols = ['year','name']
	cols += ['s'+str(i) for i in range(1,31)]
	teams_all = pd.DataFrame(teams, columns=cols)


	teams_mod = da.modify_teams(teams_all, da.mapper)
	tournament = da.create_tournament(teams_mod, da.kag_teams, da.kag_seeds, year, letter_year)

	games_all = pd.DataFrame(da.get_games_year(str(year)))
	games_all

	features_1 = da.make_features(games_all, teams_all, 'team1', 'team2', year, 1)
	features_2 = da.make_features(games_all, teams_all, 'team2', 'team1', year, 0)

	# combine remove unwanted columns
	features_total = pd.concat([features_1, features_2])
	features_total.drop(['team1','team2','year','name_1','name_2'], inplace=True, axis=1)

	# move value to end
	cols = features_total.columns.tolist()
	cols = cols[1:] + cols[:1]
	features_total = features_total[cols]
	# features_total
	feature_names = list(features_total.columns.values)
	RF_data = features_total


	actual_results = [('W16', 'W16b'),
	 ('Y11', 'Y11b'),
	 ('Y16', 'Y16b'),
	 ('Z13', 'Z13b'),
	 ('R1W1', 'W01'),
	 ('R1W2', 'W02'),
	 ('R1W3', 'W03'),
	 ('R1W4', 'W04'),
	 ('R1W5', 'W12'),
	 ('R1W6', 'W06'),
	 ('R1W7', 'W07'),
	 ('R1W8', 'W09'),
	 ('R1X1', 'X01'),
	 ('R1X2', 'X15'),
	 ('R1X3', 'X03'),
	 ('R1X4', 'X04'),
	 ('R1X5', 'X05'),
	 ('R1X6', 'X11'),
	 ('R1X7', 'X07'),
	 ('R1X8', 'X08'),
	 ('R1Y1', 'Y01'),
	 ('R1Y2', 'Y02'),
	 ('R1Y3', 'Y03'),
	 ('R1Y4', 'Y04'),
	 ('R1Y5', 'Y12'),
	 ('R1Y6', 'Y06'),
	 ('R1Y7', 'Y07'),
	 ('R1Y8', 'Y08'),
	 ('R1Z1', 'Z01'),
	 ('R1Z2', 'Z02'),
	 ('R1Z3', 'Z14'),
	 ('R1Z4', 'Z13'),
	 ('R1Z5', 'Z12'),
	 ('R1Z6', 'Z06'),
	 ('R1Z7', 'Z10'),
	 ('R1Z8', 'Z09'),
	 ('R2W1', 'W01'),
	 ('R2W2', 'W02'),
	 ('R2W3', 'W03'),
	 ('R2W4', 'W04'),
	 ('R2X1', 'X01'),
	 ('R2X2', 'X15'),
	 ('R2X3', 'X03'),
	 ('R2X4', 'X04'),
	 ('R2Y1', 'Y01'),
	 ('R2Y2', 'Y02'),
	 ('R2Y3', 'Y03'),
	 ('R2Y4', 'Y12'),
	 ('R2Z1', 'Z09'),
	 ('R2Z2', 'Z02'),
	 ('R2Z3', 'Z06'),
	 ('R2Z4', 'Z13'),
	 ('R3W1', 'W04'),
	 ('R3W2', 'W03'),
	 ('R3X1', 'X04'),
	 ('R3X2', 'X15'),
	 ('R3Y1', 'Y01'),
	 ('R3Y2', 'Y02'),
	 ('R3Z1', 'Z09'),
	 ('R3Z2', 'Z02'),
	 ('R4W1', 'W04'),
	 ('R4X1', 'X04'),
	 ('R4Y1', 'Y01'),
	 ('R4Z1', 'Z09'),
	 ('R5WX', 'X04'),
	 ('R5YZ', 'Y01'),
	 ('R6CH', 'Y01')]
	df_results = pd.DataFrame(actual_results)
	

	# ACTUALLY RUN BRACKET

	#2013 - R
	run_bracket(tourney_slots_s, RF_data, year)
	tournament_accuracy() #make sure you have the variable actual_results
	cross_validation(RF_data)
	ll_probs = log_loss_RF()
	ll = log_loss(results, ll_probs)