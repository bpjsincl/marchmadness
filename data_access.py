import sqlite3
import os

from data_files.DatabaseManager import DatabaseManager


def get_games():
    q_str = 'SELECT * FROM games;'
    games = db.query('mm_main', qstr, as_dict=True)

    return games

def get_games_matchup(team1, team2):
    inputs = {'team1': team1, 'team2': team2}
    q_str = 'SELECT * FROM games where team1 = :team1 AND team2 = :team2;'
    games = db.query('mm_main', qstr, inputs, as_dict=True)

    return games

def get_games_year(year):
    inputs = {'year': year}
    q_str = 'SELECT * FROM games where year = :year;'
    games = db.query('mm_main', qstr, inputs, as_dict=True)

    return games

def get_teams():
    q_str = 'SELECT * FROM teams;'
    teams = db.query('mm_main', qstr, as_dict=True)

    return teams

def get_teams_year(year):
    inputs = {'year': year}
    q_str = 'SELECT * FROM teams where year = :year;'
    teams = db.query('mm_main', qstr, inputs, as_dict=True)

    return teams

def get_avgpoints():
    q_str = 'SELECT * FROM avgpoints;'
    apts = db.query('mm_main', qstr, as_dict=True)

    return apts

def get_avgpoints_team(team):
    inputs = {'team': team}
    q_str = 'SELECT * FROM avgpoints where team = :team;'
    apts = db.query('mm_main', qstr, inputs, as_dict=True)

    return apts

def get_avgpoints_year(year):
    inputs = {'year': year}
    q_str = 'SELECT * FROM avgpoints where year = :year;'
    apts = db.query('mm_main', qstr, inputs, as_dict=True)

    return apts

x = os.path.join(os.getcwd(), 'data_files/mm.db')
db = DatabaseManager(db_info={'mm_main': x})