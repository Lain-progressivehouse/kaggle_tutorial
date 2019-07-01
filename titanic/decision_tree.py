from sklearn import tree
from titanic import preprocessing
import numpy as np
import pandas as pd


def train_data():
    train, test = preprocessing.get_data()

    # 「train」の目的変数と説明変数の値を取得
    target = train["Survived"].values
    features_one = train[["Pclass", "Sex", "Age", "Fare"]].values

    # 決定木の作成
    my_tree_one = tree.DecisionTreeClassifier()
    my_tree_one = my_tree_one.fit(features_one, target)

    # 「test」の説明変数の値を取得
    test_features = test[["Pclass", "Sex", "Age", "Fare"]].values

    # 「test」の説明変数を使って「my_tree_one」のモデルで予測
    my_prediction = my_tree_one.predict(test_features)

    # PassengerIdを取得
    passenger_id = np.array(test["PassengerId"]).astype(int)

    # my_prediction(予測データ）とPassengerIdをデータフレームへ落とし込む
    my_solution = pd.DataFrame(my_prediction, passenger_id, columns=["Survived"])

    # my_tree_one.csvとして書き出し
    my_solution.to_csv("/Users/main/PycharmProjects/kaggle/titanic/my_tree_one.csv", index_label=["PassengerId"])

    return my_prediction


def train_data_2():
    train, test = preprocessing.get_data()

    # 「train」の目的変数と説明変数の値を取得
    target = train["Survived"].values
    features_two = train[["Pclass", "Age", "Sex", "Fare", "SibSp", "Parch", "Embarked"]].values

    # 決定木の作成とアーギュメントの設定
    max_depth = 10
    min_samples_split = 5
    my_tree_two = tree.DecisionTreeClassifier(max_depth=max_depth, min_samples_split=min_samples_split, random_state=1)
    my_tree_two = my_tree_two.fit(features_two, target)

    # tsetから「その2」で使う項目の値を取り出す
    test_features_2 = test[["Pclass", "Age", "Sex", "Fare", "SibSp", "Parch", "Embarked"]].values

    # 「test」の説明変数を使って「my_tree_one」のモデルで予測
    my_prediction = my_tree_two.predict(test_features_2)

    # PassengerIdを取得
    passenger_id = np.array(test["PassengerId"]).astype(int)

    # my_prediction(予測データ）とPassengerIdをデータフレームへ落とし込む
    my_solution = pd.DataFrame(my_prediction, passenger_id, columns=["Survived"])

    # my_tree_one.csvとして書き出し
    my_solution.to_csv("/Users/main/PycharmProjects/kaggle/titanic/my_tree_two.csv", index_label=["PassengerId"])

    return my_prediction
