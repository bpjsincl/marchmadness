import sqlite3
import os

from DatabaseManager import DatabaseManager


def get_games():
    qstr = 'SELECT * FROM games;'
    games = db.query('mm_main', qstr, as_dict=True)

    return games

def get_games_matchup(team1, team2):
    inputs = {'team1': team1, 'team2': team2}
    qstr = 'SELECT * FROM games where team1 = :team1 AND team2 = :team2;'
    games = db.query('mm_main', qstr, inputs, as_dict=True)

    return games

def get_games_year(year):
    inputs = {'year': year}
    qstr = 'SELECT * FROM games where year = :year;'
    games = db.query('mm_main', qstr, inputs, as_dict=True)

    return games

def get_teams():
    qstr = 'SELECT * FROM teams;'
    teams = db.query('mm_main', qstr, as_dict=True)

    return teams

def get_teams_year(year):
    inputs = {'year': year}
    qstr = 'SELECT * FROM teams where year = :year;'
    teams = db.query('mm_main', qstr, inputs, as_dict=True)

    return teams

def get_avgpoints():
    qstr = 'SELECT * FROM avgpoints;'
    apts = db.query('mm_main', qstr, as_dict=True)

    return apts

def get_avgpoints_team(team):
    inputs = {'team': team}
    qstr = 'SELECT * FROM avgpoints where team = :team;'
    apts = db.query('mm_main', qstr, inputs, as_dict=True)

    return apts

def get_avgpoints_year(year):
    inputs = {'year': year}
    qstr = 'SELECT * FROM avgpoints where year = :year;'
    apts = db.query('mm_main', qstr, inputs, as_dict=True)

    return apts

x = os.path.join(os.getcwd(), 'data_files/mm.db')
db = DatabaseManager(db_info={'mm_main': x})