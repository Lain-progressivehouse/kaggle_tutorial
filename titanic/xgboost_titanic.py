import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from titanic import preprocessing


def train_data(df):
    train_x = df[["Pclass", "Age", "Sex", "Fare", "SibSp", "Parch", "Embarked"]].values
    train_y = df["Survived"].values

    # 学習データとテストデータに分割
    (train_x, test_x, train_y, test_y) = train_test_split(train_x, train_y, test_size=0.6, random_state=42)

    #
    dtrain = xgb.DMatrix(train_x, label=train_y)

    # パラメータ
    param = {'max_depth': 3, 'learning_rate': 0.6, 'objective': 'binary:logistic'}
    num_round = 2
    bst = xgb.train(param, dtrain, num_round)
    preds = bst.predict(xgb.DMatrix(test_x))
    print(accuracy_score(preds.round(), test_y))

    return bst


def predict(bst, df):
    return bst.predict(xgb.DMatrix(df))


def main():
    train, test = preprocessing.get_data()

    test_features = test[["Pclass", "Age", "Sex", "Fare", "SibSp", "Parch", "Embarked"]].values

    bst = train_data(train)
    answer = predict(bst, test_features).round().astype(int)
    submit_data = pd.Series(answer, name='Survived', index=test['PassengerId'])
    submit_data.to_csv('/Users/main/PycharmProjects/kaggle/titanic/my_xgboost_one.csv', header=True)
