'''

Author: Brian Sinclair

Editted: March 2014

Version: Python 2.7

Purpose: Get the known data from files/db


'''
from __future__ import division
import numpy as np
import pandas as pd
from pprint import pprint


def readfile(main_folder, filename, headings):  
    path = main_folder+filename
    content = pd.read_csv(path, names=headings)
    return content

def get_teams(): #get all teams in league 
    main_folder = "C:\\Users\\Brian\\Documents\\Dropbox\\University\\4B\\SYDE 522 Project\\"
    filename = "teams.csv"
    headings = ['id', 'team_name']
    data = readfile(main_folder,filename,headings)
    data = data[1:len(data)]
    return data

def get_seasons(): #get all seasons
    main_folder = "C:\\Users\\Brian\\Documents\\Dropbox\\University\\4B\\SYDE 522 Project\\"
    filename = "seasons.csv"
    headings = ['season', 'years', 'dayzero', 'regionW', 'regionX', 'regionY', 'regionZ']
    data = readfile(main_folder,filename,headings)
    data = data[1:len(data)]
    return data

def get_reg_season_results(): #get regular seasons results
    main_folder = "C:\\Users\\Brian\\Documents\\Dropbox\\University\\4B\\SYDE 522 Project\\"
    filename = "regular_season_results.csv"
    headings = ['season', 'daynum', 'wteam', 'wscore', 'lteam', 'lscore', 'wloc', 'numot']
    data = readfile(main_folder,filename,headings)
    data = data[1:len(data)]
    return data

def get_tourney_results(): #all tournaments in known history
    main_folder = "C:\\Users\\Brian\\Documents\\Dropbox\\University\\4B\\SYDE 522 Project\\"
    filename = "tourney_results.csv"
    headings = ['season', 'daynum', 'wteam', 'wscore', 'lteam', 'lscore', 'numot']
    data = readfile(main_folder,filename,headings)
    data = data[1:len(data)]
    return data

def get_tourney_seeds(): #all seeds in known history
    main_folder = "C:\\Users\\Brian\\Documents\\Dropbox\\University\\4B\\SYDE 522 Project\\"
    filename = "tourney_seeds.csv"
    headings = ['season', 'seed', 'team']
    data = readfile(main_folder,filename,headings)
    data = data[1:len(data)] 
    return data

def get_tourney_slots():
    main_folder = "C:\\Users\\Brian\\Documents\\Dropbox\\University\\4B\\SYDE 522 Project\\"
    filename = "tourney_slots.csv"
    headings = ['season', 'slot', 'strongseed', 'weakseed']
    data = readfile(main_folder,filename,headings)
    data = data[1:len(data)]
    return data

def single_tourney_teams(season, tourney_seeds): #teams from tournament of interest 
     teams = tourney_seeds.ix[tourney_seeds['season'] == season].ix[:,'team']
     return teams

def construct_round():
    #select round t
    t=1
    tourney_slots_sA = tourney_slots.ix[tourney_slots['season'] == 'A']
    tourney_slots_sA_R1 = tourney_slots_sA.ix[tourney_slots_sA['slot'].str.contains('R'+str(t))]
    tourney_R1_games = list(tourney_slots_sA_R1.slot)
    
    #get dictionary to describe round bracket
    R1_bracket=dict()
    for i in range(1,len(tourney_slots_sA_R1)+1):
        R1_bracket.update({tourney_slots_sA_R1.slot[i]: [int(tourney_slots_sA_R1.strongseed[i][1:3]), int(tourney_slots_sA_R1.weakseed[i][1:3])]})
        
    return R1_bracket, tourney_R1_games

def simple_prob_func(high_rank, low_rank):
    i=0
    tot_seeds=16
    prob_A = 1 - (high_rank / (high_rank+low_rank)) #probability of higher ranked team winning
    return prob_A

if __name__ == "__main__":
    teams = get_teams()
    seasons = get_seasons()
    reg_season_results = get_reg_season_results()
    tourney_results = get_tourney_results()
    tourney_seeds = get_tourney_seeds()
    tourney_slots = get_tourney_slots()
    
    [R1_bracket, tourney_R1_games] = construct_round()
    
    prob_A = []
    for game in range(0,len(tourney_R1_games)):
        high_rank = R1_bracket[tourney_R1_games[game]][0]
        low_rank = R1_bracket[tourney_R1_games[game]][1]
        prob_A.append(simple_prob_func(high_rank, low_rank))
    game_probs_dict = dict(zip(tourney_R1_games, prob_A))