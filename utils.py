import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from tqdm.notebook import tqdm


def get_movielens_data():
    """
    Returns a dataframe with ratings + 
    indesx of user features and item features in this global dataframe.
    """
    rating = pd.read_csv("data/u.data", sep="\t", header=None)
    print(rating.shape)
    rating.columns = ["userid", "itemid", "rating", "timestep"]
    rating.head()

    rating["index"] = np.arange(len(rating))

    prod = pd.read_csv("data/u.item", sep="|", header=None)
    print(prod.shape)
    prod.head()

    prod["year"] = prod[2].apply(lambda x: float(str(x)[-4:]))

    prod_features = [
        5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23,
        "year"
    ]

    rating = pd.merge(rating,
                      prod[[0] + prod_features],
                      left_on="itemid",
                      right_on=0,
                      how="left").drop(0, axis=1)

    users = pd.read_csv("data/u.user", sep="|", header=None)
    print(users.shape)
    users.head()

    cat = pd.get_dummies(users[[2, 3]])
    user_features = np.concatenate([cat.columns, [1, 0]])
    users = pd.concat([users, cat], axis=1)
    rating = pd.merge(rating,
                      users[user_features],
                      left_on="userid",
                      right_on=0,
                      how="left").drop(0, axis=1)
    rating = rating.dropna()
    user_column_index = np.arange(
        list(rating.columns).index("2_F"), rating.shape[1])

    item_column_index = np.arange(5, list(rating.columns).index("2_F"))
    
    # scale user features
    scaled = StandardScaler().fit_transform(
    rating[rating.columns[user_column_index]])
    rating.loc[:, rating.columns[user_column_index]] = scaled

    # scale item features
    scaled = StandardScaler().fit_transform(
        rating[rating.columns[item_column_index]])
    rating.loc[:, rating.columns[item_column_index]] = scaled

    return rating, user_column_index, item_column_index


def movielens_train_test_split(rating):
    """Extracts from the train data a test split
    which cosists for each user of its last interaction.

    Args:
        rating (_type_): _description_

    Returns:
        _type_: _description_
    """
    test = rating.groupby("userid")[["timestep"]].max().reset_index()
    test = rating.groupby("userid").last().reset_index()
    test.shape

    test_ids = pd.merge(rating, test[["userid", "itemid"]], on = ["userid", "itemid"], how = "inner")["index"].values

    train_df = rating[~ rating["index"].isin(test_ids)]
    test_df = rating[rating["index"].isin(test_ids)]
    return train_df, test_df
