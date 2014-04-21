'''

Author: Brian Sinclair

Editted: March 2014

Version: Python 2.7

Purpose: Calculate statistics required to be used as features for ANN


'''
from __future__ import division
import numpy as np
import pandas as pd
from pprint import pprint
import get_data as gd
from get_data import *

def tourney_seeds_dict(tourney_seeds, curr_season): #create a mapping of teams to their ranking in a given season's tournament
    Tseeds = tourney_seeds.ix[tourney_seeds['season'] == curr_season]
    Tseeds_dict = Tseeds.ix[:,'seed':].set_index('seed').to_dict() #convert dataframe to dict
    # Tseeds_dict['seasonA'] = Tseeds_dict.pop('team') #this would be for if wanted to create dict of every tournament -- access by season
    Tseeds_map = Tseeds_dict['team']
    return Tseeds_map

def season_stats(season, team, main_folder):#calculate current season's stats
    reg_season_results = gd.get_reg_season_results(main_folder)

    reg_season_results_s = reg_season_results.ix[reg_season_results['season'] == season]
    #team_games = reg_season_results_sA.ix[(reg_season_results_sA['wteam'] == team) | (reg_season_results_sA['lteam'] == team)] #all games by specific team
    reg_season_results_s = reg_season_results_s.applymap(str)
    team_games_win = reg_season_results_s.ix[(reg_season_results_s['wteam'] == team)]
    team_games_loss = reg_season_results_s.ix[(reg_season_results_s['lteam'] == team)]

    win_scores = map(int,list(team_games_win.ix[:,'wscore'])) #map(int, results)
    games_won = len(win_scores)
    loss_scores = map(int,list(team_games_loss.ix[:,'lscore'])) #map(int, results)
    games_loss = len(loss_scores)

    season_record = games_won/(games_won + games_loss)
    
    avg_score = (sum(win_scores) + sum(loss_scores))/(games_won + games_loss)
    
    return avg_score, season_record

def last_tourney_stats(last_season):
    last_tourney_stats = [] #create vector of the last year's stats if team was in tournament otherwise populate with 0's
    
    return last_tourney_stats

def teams_season_stats(teams_in_tourney, curr_season, main_folder):
    teams_stats_dict = dict()
    for team in teams_in_tourney:
        curr_season_stats = season_stats(curr_season,team, main_folder) #regular season stats (avg_score, win/loss %)
        last_tourney_stats = [] #create vector of the last year's stats if team was in tournament otherwise populate with 0's
        teams_stats_dict.update({team: curr_season_stats})
        
    return teams_stats_dict

def bracket(tourney_match_seeds, T_matchup_results, Tseeds_map):  
    winners = []
    outcome = []
    games = []
    for matchup in tourney_match_seeds:
        opt1 = ((T_matchup_results['wteam'] == Tseeds_map[matchup[1][0]]) & (T_matchup_results['lteam'] == Tseeds_map[matchup[1][1]]))
        opt2 = ((T_matchup_results['lteam'] == Tseeds_map[matchup[1][0]]) & (T_matchup_results['wteam'] == Tseeds_map[matchup[1][1]]))
        if opt1.any() == True: #then winner is team A
            winner = matchup[1][0]
            home = 1
        else:
            winner = matchup[1][1]
            home = 0
            
        outcome = [matchup[0], winner]
        game = [matchup[1], home]
        games.append(game)
        winners.append(outcome)
    return winners, games

def construct_tourney_winners(tourney_slots_s, T_matchup_results, Tseeds_map):
	rounds = [1, 2, 3, 4, 5, 6]
	last_winners_dict = {}
	all_games = []

	tourney_slots_s_entry = tourney_slots_s.ix[~tourney_slots_s['slot'].str.contains('R')] #check for entry round for last place seed
	if not tourney_slots_s_entry.empty:
		seed_matchups = tourney_slots_s_entry[['slot', 'strongseed', 'weakseed']].values
		tourney_match_seeds = [[x[0], tuple(x[1:3])] for x in seed_matchups] #matchups by game label
		winners, games = bracket(tourney_match_seeds, T_matchup_results, Tseeds_map)
		all_games.append(games)

		last_winners = [Tseeds_map[s[1]] for s in winners] #teams that won first round
		last_winners_dict.update(winners)
		for x in seed_matchups:
			last_winners_dict[x[0]] = Tseeds_map[last_winners_dict[x[0]]] #winning team of entry round
		Tseeds_map.update(last_winners_dict) #update dict with new seed for 16th
		last_winners_dict = {}
		all_games = []

	for R in rounds:
		tourney_slots_s_R = tourney_slots_s.ix[tourney_slots_s['slot'].str.contains('R'+str(R))]
		seed_matchups = tourney_slots_s_R[['slot', 'strongseed', 'weakseed']].values
		tourney_match_seeds = [[x[0], tuple(x[1:3])] for x in seed_matchups] #matchups by game label

		if R!=1:
			tourney_match_seeds = [[x[0], (last_winners_dict[x[1][0]],last_winners_dict[x[1][1]])] for x in tourney_match_seeds]
			winners = bracket(tourney_match_seeds, T_matchup_results, Tseeds_map)

		winners, games = bracket(tourney_match_seeds, T_matchup_results, Tseeds_map)
		
		for x in games:
			all_games.append(x)

		last_winners = [Tseeds_map[s[1]] for s in winners] #teams that won first round
		last_winners_dict.update(winners)
	return last_winners_dict, tourney_match_seeds, all_games

def create_features(all_game_outcomes, tourney_stats_dict, Tseeds_map):
    #create vector for NN -- addition of two team's stats
    input_vec = []

    for game in all_game_outcomes:
        matchup = list(tourney_stats_dict[Tseeds_map[game[0][0]]]) + list(tourney_stats_dict[Tseeds_map[game[0][1]]]) + [game[1]]
        input_vec.append(matchup)
    return input_vec

if __name__ == '__main__':
    main_folder = "..//data_files//"
    all_seasons = list(gd.get_seasons(main_folder)['season'])
    last_season = all_seasons[0]
    curr_season = all_seasons[0]
    
    tourney_seeds = gd.get_tourney_seeds(main_folder)
    tourney_slots = gd.get_tourney_slots(main_folder)
    tourney_results = gd.get_tourney_results(main_folder)
    
    tourney_results_s = tourney_results.ix[tourney_results['season'] == curr_season]
    T_matchup_results = tourney_results_s[['wteam', 'lteam']]
    tourney_slots_s = tourney_slots.ix[tourney_slots['season'] == curr_season]
    
    teams_in_tourney = gd.single_tourney_teams(curr_season,tourney_seeds)
    tourney_stats_dict = teams_season_stats(teams_in_tourney, curr_season, main_folder) #(reg season avg_score, reg season win/loss %)
    Tseeds_map = tourney_seeds_dict(tourney_seeds, curr_season) #get mapping of teams to seed in tournament

    last_winners_dict, tourney_match_seeds, all_game_outcomes = construct_tourney_winners(tourney_slots_s, T_matchup_results, Tseeds_map)
    tournament_champion = Tseeds_map[last_winners_dict['R6CH']]
    
    NN_input_vec = create_features(all_game_outcomes, tourney_stats_dict, Tseeds_map)

    feature_names = ['A_avg_score', 'A_win_pct', 'B_avg_score', 'B_win_pct', 'winner']
    RF_data = pd.DataFrame(NN_input_vec, columns=feature_names)