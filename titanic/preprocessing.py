import numpy as np
import pandas as pd
import re

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


def get_data_ex():
    """
    ['Survived', 'Pclass', 'Sex', 'Age', 'Parch', 'Fare', 'Embarked',
       'Name_length', 'Has_Cabin', 'FamilySize', 'IsAlone', 'Title']
    :return:
    """
    train = pd.read_csv("/Users/main/PycharmProjects/kaggle/titanic/train.csv")
    test = pd.read_csv("/Users/main/PycharmProjects/kaggle/titanic/test.csv")

    passengerId = test["PassengerId"]

    full_data = [train, test]

    # 名前の長さ
    train["Name_length"] = train["Name"].apply(len)
    test["Name_length"] = test["Name"].apply(len)

    # 客室番号データがあるなら1, 欠損値なら0
    train["Has_Cabin"] = train["Cabin"].apply(lambda x: 0 if type(x) == float else 1)
    test["Has_Cabin"] = test["Cabin"].apply(lambda x: 0 if type(x) == float else 1)

    # 家族の大きさを同乗している兄弟/配偶者の数と同乗している親/子供の数から定義
    for dataset in full_data:
        dataset["FamilySize"] = dataset["SibSp"] + dataset["Parch"] + 1

    # 独り身かどうか,
    for dataset in full_data:
        dataset["IsAlone"] = dataset["FamilySize"].map(lambda x: 1 if x == 1 else 0)

    # 出港地の欠損値を一番多い"S"としておく
    for dataset in full_data:
        dataset["Embarked"] = dataset["Embarked"].fillna("S")

    # 料金の欠損値を中央値としておく
    # 料金を大きく4つのグループに分ける
    for dataset in full_data:
        dataset["Fare"] = dataset["Fare"].fillna(train["Fare"].median())
    train["CategoricalFare"] = pd.qcut(train["Fare"], 4)

    # 年齢を5つのグループに分ける
    for dataset in full_data:
        age_avg = dataset["Age"].mean()
        age_std = dataset["Age"].std()  # 標準偏差
        age_null_count = dataset["Age"].isnull().sum()
        age_null_random_list = np.random.randint(age_avg - age_std, age_avg + age_std, size=age_null_count)
        dataset.loc[np.isnan(dataset["Age"]), "Age"] = age_null_random_list
        dataset["Age"] = dataset["Age"].astype(int)
    train["CategoricalAge"] = pd.cut(train["Age"], 5)

    # 名前を取り出す関数1
    def get_title(name):
        title_search = re.search(r"([A-Za-z]+)\.", name)
        # 名前があれば取り出して返す
        if title_search:
            return title_search.group(1)
        return ""

    # 関数1を使う
    for dataset in full_data:
        dataset["Title"] = dataset["Name"].apply(get_title)

        # 名前の変なところを変換
        dataset["Title"] = dataset["Title"].replace(
            ["Lady", "Countess", "Capt", "Col", "Don", "Dr", "Major", "Rev", "Sir", "Jonkheer", "Dona"], "Rare")

        dataset["Title"] = dataset["Title"].replace("Mlle", "Miss")
        dataset["Title"] = dataset["Title"].replace("Ms", "Miss")
        dataset["Title"] = dataset["Title"].replace("Mme", "Mrs")

    for dataset in full_data:
        # 女なら0, 男なら1
        dataset["Sex"] = dataset["Sex"].map({"female": 0, "male": 1}).astype(int)

        # 名前の5種類にラベル付
        title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}
        dataset["Title"] = dataset["Title"].map(title_mapping)
        dataset["Title"] = dataset["Title"].fillna(0)

        # 出港地の3種類にラベル付
        dataset["Embarked"] = dataset["Embarked"].map({"S": 0, "C": 1, "Q": 2}).astype(int)

        # 料金を4つのグループに分ける
        dataset.loc[dataset["Fare"] <= 7.91, "Fare"] = 0
        dataset.loc[(dataset["Fare"] > 7.91) & (dataset["Fare"] <= 14.454), "Fare"] = 1
        dataset.loc[(dataset["Fare"] > 14.545) & (dataset["Fare"] <= 31), "Fare"] = 2
        dataset.loc[dataset["Fare"] > 31, "Fare"] = 3
        dataset["Fare"] = dataset["Fare"].astype(int)

        # 年齢を5つのグループに分ける
        dataset.loc[dataset["Age"] <= 16, "Age"] = 0
        dataset.loc[(dataset["Age"] > 16) & (dataset["Age"] <= 32), "Age"] = 1
        dataset.loc[(dataset["Age"] > 32) & (dataset["Age"] <= 48), "Age"] = 2
        dataset.loc[(dataset["Age"] > 48) & (dataset["Age"] <= 64), "Age"] = 3
        dataset.loc[dataset["Age"] > 64, "Age"] = 4

    # 必要ない特徴を削除
    drop_elements = ["PassengerId", "Name", "Ticket", "Cabin", "SibSp"]
    train = train.drop(drop_elements, axis=1)
    train = train.drop(["CategoricalAge", "CategoricalFare"], axis=1)
    test = test.drop(drop_elements, axis=1)

    test["PassengerId"] = passengerId

    return train, test


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
