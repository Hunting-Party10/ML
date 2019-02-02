import pandas as pd
from sklearn import linear_model

train_path = 'train.csv'
test_path = 'test.csv'

train_data = pd.read_csv(train_path, names = ['Hours','Score'])
test_data = pd.read_csv(test_path, names = ['Hours'])

#print(train_data.columns)




X = train_data['Hours'].values.reshape(-1,1)
y = train_data['Score']
model = linear_model.LinearRegression()

model.fit(X,y)

Xfind = test_data['Hours'].values.reshape(-1,1)
#print(Xfind.columns)

yfound = model.predict(Xfind)

Xfind = pd.DataFrame(Xfind)
yfound =pd.DataFrame(yfound)

result = pd.concat([ Xfind, yfound], axis =1 ,sort = False)

export_csv = result.to_csv(r'answer.csv' , index = None, header=train_data.columns)
