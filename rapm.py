from sklearn import linear_model
import numpy as np
from scipy.sparse import lil_matrix, csr_matrix
import api_helper

def run_bayes_model(X, y):
	import pymc as pm
	basic_model = pm.Model()
	with basic_model:
		alpha = pm.Normal("alpha", mu = 1.1, sigma = 0.1)
		beta = pm.Normal("beta", 0, 0.02, shape=(np.shape(X)[1],))#standard deviation needs to be supplied and isn't always immediately obvious
		mu = alpha + pm.math.dot(X, beta)
		Y_obs = pm.Normal("Y_obs", mu = mu, observed = y)
		idata = pm.find_MAP()
	return idata['beta']

def run_ridge_model(X, y, sample_weights):
	clf = linear_model.RidgeCV(alphas = [1500, 1750, 2000, 2250, 2500, 2750, 3000, 3250, 3500, 3750, 4000], cv = 4)#Other options are RidgeCV ElasticNetCV, Lasso etc, all with different penalization parameters
	clf.fit(X, y, sample_weight = sample_weights)
	print ('ALPHA:', clf.alpha_)
	return clf.coef_

def main():
	#cur = MySQLdb.connect()
	cur.execute("select home_poss, pts, a1, a2, a3, a4, a5, h1, h2, h3, h4, h5, season from matchups where season = 2024")
	data = cur.fetchall()

	all_players = {}#get all players in the dataset
	for item in data:
		for i in range(2, 12):
			all_players[item[i]] = 1

	player_to_col = {}; col_to_player = {}#for each player we create an 'offensive' and a 'defensive' variable. Each has to be translated to a specific column
	for p in all_players:
		for side in ['off', 'def']:
			p_side = str(p)+'_'+side
			if p_side not in player_to_col:
				number = len(player_to_col)
				player_to_col[p_side] = number
				col_to_player[number] = p_side

	X = lil_matrix((len(data), len(col_to_player)))#use sparse matrixes so memory doesn't blow up
	y = np.zeros(len(data))
	sample_weights = []
	season_weights = {2024: 1.0, 2023: 0.9, 2022: 0.8}#etc.
	counter = 0
	for item in data:
		home_poss = item[0]
		pts = item[1]
		season = item[12]
		home_list = []; away_list = []
		for i in range(2, 7):
			away_list.append(item[i])
		for i in range(7, 12):
			home_list.append(item[i])
		if home_poss:
			[off_list, def_list] = home_list, away_list
		else:
			[off_list, def_list] = away_list, home_list
		for p in off_list:
			off_p = str(p)+'_off'
			X[counter, player_to_col[off_p]] = 1#'switch on' dummy variables for all offensive players present
		for p in def_list:
			def_p = str(p)+'_def'
			X[counter, player_to_col[def_p]] = 1#'switch on' dummy variables for all defensive players present
		y[counter] = pts
		sample_weights.append(season_weights[season])
		counter += 1
	y_av = np.average(y)
	
	beta_ridge = run_ridge_model(X.tocsr(), y - y_av, sample_weights)
	beta_bm = run_bayes_model(X.todense(), y)#Bayesian model

	for i in range(0, len(beta_ridge)):
		print (col_to_player[i], ';', beta_ridge[i], ';', beta_bm[i])

if __name__=='__main__':
	main()
