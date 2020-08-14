
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn import preprocessing

#pd.set_option('display.max_columns', None)

print('Reading train data')
df = pd.read_csv('train.csv')
print('Reading Test Data')
df2 = pd.read_csv('test.csv')


print('Making modifications')
train_data = df
train_data['team_id'] = train_data['team_id'].str.extract('(\d+)').astype(int)
train_data['opp_team_id'] = train_data['opp_team_id'].str.extract('(\d+)').astype(int)
train_data['playoff'] = train_data['playoff'].replace(False,0)
df2['team_id'] = df2['team_id'].str.extract('(\d+)').astype(int)
df2['opp_team_id'] = df2['opp_team_id'].str.extract('(\d+)').astype(int)
df2['playoff'] = df2['playoff'].replace(False,0)
X_train = train_data.drop(labels=['game_result'],axis=1)
Y_train = train_data['game_result']
X_test = df2



X_train = train_data.drop(labels=['date','Id','win_equivalent','Elo','game_result','team_id','opp_team_id'],axis = 1)
Y_train = train_data['game_result']
X_test = df2.drop(labels=['date','Id','win_equivalent','Elo','team_id','opp_team_id'],axis = 1)

X_train['crowd_diff'] = X_train['home_crowd'] - X_train['opp_crowd']
X_test['crowd_diff'] = X_test['home_crowd'] - X_test['opp_crowd']

X_train = X_train.drop(labels=['home_crowd','opp_crowd','total_crowd','game_seq','season_end'],axis = 1)
X_test = X_test.drop(labels=['home_crowd','opp_crowd','total_crowd','game_seq','season_end'],axis = 1)

X_train = X_train.fillna(X_train.mean())
X_test = X_test.fillna(X_test.mean())

#X_train = preprocessing.scale(X_train)
#X_test = preprocessing.scale(X_test)

X_train,X_cross,y_train,y_cross = train_test_split(X_train,Y_train, test_size = 0.4)

#print(X_train.column)

clf = svm.SVC(kernel='linear', C=1, verbose=True).fit(X_train,y_train)

print(clf.score(X_cross,y_cross))


pickle.dump(clf, open('model.sav','wb'))
