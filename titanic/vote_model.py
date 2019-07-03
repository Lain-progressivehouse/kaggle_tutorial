import pandas as pd
import numpy as np

def main():
    model1 = pd.read_csv("/Users/main/PycharmProjects/kaggle/titanic/my_xgboost_one.csv")
    model2 = pd.read_csv("/Users/main/PycharmProjects/kaggle/titanic/my_xgboost_two.csv")
    model3 = pd.read_csv("/Users/main/PycharmProjects/kaggle/titanic/my_tree_two.csv")

    answer = model1["Survived"] + model2["Survived"] + model3["Survived"]
    answer = answer.apply(lambda x: 0 if x < 2 else 1)
    answer = np.array(answer)
    passenger_id = np.array(model1["PassengerId"]).astype(int)
    my_solution = pd.DataFrame(answer, passenger_id, columns=["Survived"])

    # my_tree_one.csvとして書き出し
    my_solution.to_csv("/Users/main/PycharmProjects/kaggle/titanic/vote_model_one.csv", index_label=["PassengerId"])
