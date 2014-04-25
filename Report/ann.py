ef build_nn(NN_input, numNeurons):
	DS= SupervisedDataSet(len(NN_input[1][:len(NN_input[1])-1]), 2)

	for i,matchup in enumerate(NN_input):
		DS.addSample(matchup[:len(matchup)-1], [matchup[len(matchup)-1], 1-abs(matchup[len(matchup)-1])])
	tstdata, trndata = DS.splitWithProportion(0.3)

	print "Number of training patterns: ", len(trndata)
	print "Input and output dimensions: ", trndata.indim, trndata.outdim
	print "First sample (input, target, class):"

	# neural net with 2 hidden layers and numNeurons for each
	net = buildNetwork( trndata.indim, numNeurons, numNeurons, trndata.outdim, hiddenclass=SoftmaxLayer, outclass=SigmoidLayer, bias=True)
	trainer = BackpropTrainer( net, dataset=trndata)
	return net, trainer, trndata, tstdata, DS

def run_nn(trainer, tstdata, trndata, numEpoch, numIter):
	# this will run the neural using the backprop trainer on test and training data with using numepochs and numiterations
	modval = ModuleValidator()
	for i in range(numIter):
		trainer.trainEpochs(numEpoch)
		trnresult = percentError( trainer.testOnClassData(), trndata['target'] )#trndata['class'] )
		tstresult = percentError( trainer.testOnClassData( dataset=tstdata ), tstdata['target'] )#tstdata['class'] )

		#simultaneously cross validate
		cv = CrossValidator( trainer, trndata, n_folds=5, valfunc=modval.MSE )
		print "MSE %f @ %i" %( cv.validate(), i )
	return trnresult, tstresult