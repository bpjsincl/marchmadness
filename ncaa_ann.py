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
import sys
sys.path.append("..//")
from get_data import *
import calc_stats as cs
from calc_stats import *

from pybrain.datasets            import ClassificationDataSet
from pybrain.utilities           import percentError
from pybrain.tools.shortcuts     import buildNetwork
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.structure.modules   import SoftmaxLayer

if __name__ == '__main__':
    main_folder = "..//data_files//"
    all_seasons = list(gd.get_seasons(main_folder)['season'])
    train_seasons = all_seasons[:15]
    test_seasons = all_seasons[15:]
    
    tourney_seeds = gd.get_tourney_seeds(main_folder)
    tourney_slots = gd.get_tourney_slots(main_folder)
    tourney_results = gd.get_tourney_results(main_folder)
    
    NN_input_vec = []
    for season in all_seasons[:-1]:
        tourney_results_s = tourney_results.ix[tourney_results['season'] == season]
        T_matchup_results = tourney_results_s[['wteam', 'lteam']]
        tourney_slots_s = tourney_slots.ix[tourney_slots['season'] == season]
        
        teams_in_tourney = gd.single_tourney_teams(season,tourney_seeds)
        tourney_stats_dict = cs.teams_season_stats(teams_in_tourney, season, main_folder) #(reg season avg_score, reg season win/loss %)
        Tseeds_map = cs.tourney_seeds_dict(tourney_seeds, season) #get mapping of teams to seed in tournament
    
        last_winners_dict, tourney_match_seeds, all_game_outcomes = cs.construct_tourney_winners(tourney_slots_s, T_matchup_results, Tseeds_map)
        tournament_champion = Tseeds_map[last_winners_dict['R6CH']]
        NN_input = create_features(all_game_outcomes, tourney_stats_dict, Tseeds_map)

        for x in NN_input:
            NN_input_vec.append(x)

   	DS = ClassificationDataSet(len(NN_input_vec[1][:len(NN_input_vec[1])-1]), nb_classes=2, class_labels=['Win','Loss'])

	for i,matchup in enumerate(NN_input_vec):
	    DS.appendLinked(matchup[:len(matchup)-1], matchup[len(matchup)-1])
    
    tstdata, trndata = DS.splitWithProportion(0.2)
	trndata._convertToOneOfMany( )
	tstdata._convertToOneOfMany( )

	print "Number of training patterns: ", len(trndata)
	print "Input and output dimensions: ", trndata.indim, trndata.outdim
	print "First sample (input, target, class):"
	print trndata['input'][0], trndata['target'][0], trndata['class'][0]

	fnn = buildNetwork( trndata.indim, 3, trndata.outdim, outclass=SoftmaxLayer )

	trainer = BackpropTrainer( fnn, dataset=trndata, momentum=0.1, verbose=True, weightdecay=0.01)

	for i in range(20):
	    trainer.trainEpochs(100)
	    trnresult = percentError( trainer.testOnClassData(), trndata['class'] )
	    tstresult = percentError( trainer.testOnClassData( dataset=tstdata ), tstdata['class'] )

	#print "epoch: %4d" % trainer.totalepochs, \
	 #     "  train error: %5.2f%%" % trnresult, \
	  #    "  test error: %5.2f%%" % tstresult

  	result = fnn.activateOnDataset(tstdata)
	print result.argmax(axis=1) #output classification from neural network

	dataset=tstdata 
	trainer.testOnClassData( dataset=tstdata ), tstdata['class']