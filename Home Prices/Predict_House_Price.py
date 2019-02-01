import pandas as pd
from sklearn.tree import DecisionTreeRegressor

training_set_path = 'train.csv'
test_set_path = 'test.csv'

training_set = pd.read_csv(training_set_path)
test_set = pd.read_csv(test_set_path)

#print(training_set.columns)

features =['MSSubClass', 'LotArea']
to_predict = ['SalePrice']

X = training_set[features]
y = training_set[to_predict]

#X.dropna(how = 'any')

#print(X.head());
#print(y.describe());

model = DecisionTreeRegressor(random_state = 1)
model.fit(X, y)

test_set=test_set[features]
print(model.predict(test_set.head()))
