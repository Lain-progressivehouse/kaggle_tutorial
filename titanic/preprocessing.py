import numpy as np
import pandas as pd

"""
PassengerId – 乗客識別ユニークID
Survived – 生存フラグ（0=死亡、1=生存）
Pclass – チケットクラス
Name – 乗客の名前
Sex – 性別（male=男性、female＝女性）
Age – 年齢
SibSp – タイタニックに同乗している兄弟/配偶者の数
parch – タイタニックに同乗している親/子供の数
ticket – チケット番号
fare – 料金
cabin – 客室番号
Embarked – 出港地（タイタニックへ乗った港）
"""


def get_data():
    train = pd.read_csv("/Users/main/PycharmProjects/kaggle/titanic/train.csv")
    test = pd.read_csv("/Users/main/PycharmProjects/kaggle/titanic/test.csv")

    # 欠損値のAgeは中央値
    train["Age"] = train["Age"].fillna(train["Age"].median())
    # 欠損値のEmbarkedは最頻値
    train["Embarked"] = train["Embarked"].fillna("S")

    # 文字を数字に変換
    train["Sex"] = train["Sex"].map({"male": 0, "female": 1})
    train["Embarked"] = train["Embarked"].map({"S": 0, "C": 1, "Q": 2})

    # testデータも同様に
    test["Age"] = test["Age"].fillna(test["Age"].median())
    test["Sex"] = test["Sex"].map({"male": 0, "female": 1})
    test["Embarked"] = test["Embarked"].map({"S": 0, "C": 1, "Q": 2})
    test["Fare"] = test["Fare"].fillna(test["Fare"].median())

    return train, test
