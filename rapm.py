from sklearn import linear_model
import numpy as np
from scipy.sparse import lil_matrix, csr_matrix

def main():
	#cur = MySQLdb.connect()
	cur.execute("select home_poss, pts, a1, a2, a3, a4, a5, h1, h2, h3, h4, h5, season from matchups")
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
	season_weights = {2024: 1.0, 2023: 1.0, 2022: 1.0}#etc.
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

	y -= np.average(y)
	X = X.tocsr()#faster than lil_matrix for computation
	clf = linear_model.Ridge(alpha = 3000)#Other options are RidgeCV ElasticNetCV, Lasso etc, all with different penalization parameters
	clf.fit(X, y, sample_weight = sample_weights)
	for i in range(0, len(clf.coef_)):
		print (col_to_player[i], ';', clf.coef_[i])

if __name__=='__main__':
	main()
