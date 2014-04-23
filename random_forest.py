from get_data import *
from calc_stats import *
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import numpy as np
import random

def get_features(team_A, team_B, year):
    print team_A
    print team_B
    teamA = teams_map[teams_map['id']==Tseeds_map[str(team_A)]]['team_name']
    teamB = teams_map[teams_map['id']==Tseeds_map[str(team_B)]]['team_name']    
    fs = da.make_features_matchup(teams_mod, teamA.iloc[0], teamB.iloc[0], year)
    return fs

def classify(matchup, training_data, year):
    
    feature_names = list(training_data.columns.values)
    features = get_features(matchup['strongseed'], matchup['weakseed'], year)
#     SS_features = get_features(matchup['strongseed'])
#     WS_features= get_features(matchup['weakseed'])
    
    test = pd.DataFrame(features, columns=feature_names[:30]) #30 features
    training_data.head()
    features = training_data.columns[:30] #30 features
    clf = RandomForestClassifier(n_jobs=1)
    y, _ = pd.factorize(training_data['value'])
    clf.fit(training_data[features], y)
    
    prob_preds = clf.predict_proba(test[features])
    preds = clf.predict(test[features])
    
    winner = matchup['strongseed'] if preds[0] == 1 else matchup['weakseed']
    prob = prob_preds[0][0] if preds[0] == 1 else prob_preds[0][1]
    return winner, prob

def run_bracket(tourney_slots_s, RF_data, year):
    tourney_slots_s['winner'] = 0
    tourney_slots_s['prob'] = 0
    for num in range(0,len(tourney_slots_s)):
        matchup = tourney_slots_s.iloc[num]
        round_1_flag = True
        if list(matchup['strongseed'])[0] == 'R':
            team_A = tourney_slots_s[(tourney_slots_s.slot == matchup['strongseed'])]['winner']
            team_B = tourney_slots_s[(tourney_slots_s.slot == matchup['weakseed'])]['winner']
            matchup.loc['strongseed'] = list(team_A)[0]
            matchup.loc['weakseed'] = list(team_B)[0]
            print matchup
        
        winner, prob = classify(matchup, RF_data, year)
        matchup.loc['prob'] = prob
        matchup.loc['winner'] = winner

        tourney_slots_s.iloc[num] = matchup

def cross_validation(training_data):
    training_data['is_train'] = np.random.uniform(0, 1, len(RF_data)) <= .75
    training_data.head()
     
    train, test = training_data[training_data['is_train']==True], training_data[training_data['is_train']==False]
     
    features = training_data.columns[:4]
    clf = RandomForestClassifier(n_jobs=2)
    y, _ = pd.factorize(train['value'])
    clf.fit(train[features], y)
    
    target_names = np.array(['strongseed', 'weakseed'])
    preds = target_names[clf.predict(test[features])]
    cross_tab = pd.crosstab(test['value'], preds, rownames=['actual'], colnames=['preds'])
    return cross_tab

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

	# ACTUALLY RUN BRACKET
	
	# run_bracket(tourney_slots_s, RF_data, year)
	# tourney_slots_s
	cross_validation(RF_data)