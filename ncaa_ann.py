'''

Author: Brian Sinclair

Editted: March 2014

Version: Python 2.7

Purpose: Runs ANN


'''
from __future__ import division
import numpy as np
import pandas as pd
from pprint import pprint
import data_access as da
from itertools import combinations
from pybrain.datasets            import ClassificationDataSet
from pybrain.utilities           import percentError
from pybrain.tools.shortcuts     import buildNetwork
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.structure.modules   import SoftmaxLayer
from pybrain.structure 			 import TanhLayer
from pybrain.tools.neuralnets 	 import NNregression, Trainer
from pybrain.tools.validation 	 import ModuleValidator
from pybrain.tools.validation 	 import CrossValidator

# This merges the data together to create the vector to pass into the learning algos.
def make_features(games, teams, team_a, team_b, year, value):
    fts = pd.merge(
    pd.merge(games[games.year==year], 
             teams[teams.year==year], 
             left_on=(team_a, 'year'), 
             right_on=('name', 'year'), 
             how='left'), 
                    teams[teams.year==year], 
                    left_on=(team_b, 'year'), 
                    right_on=('name', 'year'), 
                    how='left',
                    suffixes=('_1','_2'))
    fts['value'] = value
    fts = fts[(fts.name_1.notnull()) & (fts.name_2.notnull())]
    return fts

def get_nn_input(years):
	reload(da)
	teams = da.get_teams()
	cols = ['year','name']
	cols += ['s'+str(i) for i in range(1,31)]
	teams_all = pd.DataFrame(teams, columns=cols)

	matchup_all = pd.DataFrame()
	for year in years:
		# CHANGE YEAR HERE:
		games_all_year = pd.DataFrame(da.get_games_year(year))

		features_1 = make_features(games_all_year, teams_all, 'team1', 'team2', int(year), 1)
		features_2 = make_features(games_all_year, teams_all, 'team2', 'team1', int(year), 0)

		# combine remove unwanted columns
		features_total = pd.concat([features_1, features_2])
		features_total.drop(['team1','team2','year','name_1','name_2'], inplace=True, axis=1)

		# move value to end
		cols = features_total.columns.tolist()
		cols = cols[1:] + cols[:1]
		features_total = features_total[cols]

		matchup_all = pd.concat([matchup_all, features_total], axis=0)
	return matchup_all

def build_nn(NN_input):
	DS = ClassificationDataSet(len(NN_input[1][:len(NN_input[1])-1]), nb_classes=2, class_labels=['Loss','Win'])

	for i,matchup in enumerate(NN_input):
		DS.appendLinked(matchup[:len(matchup)-1], matchup[len(matchup)-1])
	tstdata, trndata = DS.splitWithProportion(0.3)
	trndata._convertToOneOfMany( )
	tstdata._convertToOneOfMany( )

	print "Number of training patterns: ", len(trndata)
	print "Input and output dimensions: ", trndata.indim, trndata.outdim
	print "First sample (input, target, class):"
	print trndata['input'][0], trndata['target'][0], trndata['class'][0]

	# neural net with 2 hidden layers and 3 nodes -- 3 nodes chosen largely due to computation speed
	net = buildNetwork( trndata.indim, 3, 3, trndata.outdim, hiddenclass=TanhLayer, bias=True)
	net.sorted = False
	net.sortModules()

	trainer = BackpropTrainer( net, dataset=trndata)

	return net, trainer, trndata, tstdata, DS

def modify_teams(teams, mapper):
	for team in mapper:
		teams.ix[teams.name==team[0],'name'] = team[1]

	return teams

def create_tournament(teams, kag_teams, kag_seeds, year, season):
	cols = ['name']
	cols += ['s'+str(i) for i in range(1,31)]
	teams_combined = pd.merge(pd.merge(kag_seeds[kag_seeds.season==season], 
										kag_teams, 
										left_on='team', 
										right_on='id', 
										how='left'),
										teams[teams.year==year], 
										left_on='name', 
										right_on='name', 
										how='left',
										suffixes=('_1', '_2'))

	team_tuples = [tuple(x) for x in teams_combined[cols].values]
	tournament = [(i[1:] + j[1:]) for i, j in combinations(team_tuples, 2)]
	teams = [(i[0], j[0]) for i,j in combinations(team_tuples,2)]

	return tournament, teams

def run_nn(trainer, tstdata, trndata, numEpoch, numIter):
	# this will run the neural using the backprop trainer on test and training data with using 3 epochs and 20 iterations
	modval = ModuleValidator()
	for i in range(numIter):
		trainer.trainEpochs(numEpoch)
		trnresult = percentError( trainer.testOnClassData(), trndata['class'] )
		tstresult = percentError( trainer.testOnClassData( dataset=tstdata ), tstdata['class'] )

		#simultaneously cross validate
		cv = CrossValidator( trainer, trndata, n_folds=5, valfunc=modval.MSE )
		print "MSE %f @ %i" %( cv.validate(), i )
	return trnresult, tstresult

if __name__ == '__main__':
	years = ['2010', '2011', '2012', '2013']
	matchup_all = get_nn_input(years)
	NN_input = matchup_all.values.tolist()
	net, trainer = build_nn(NN_input)