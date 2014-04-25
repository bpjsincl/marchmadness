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