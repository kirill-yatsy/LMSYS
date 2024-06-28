import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

label_to_int = {
    "winner_model_a": 0,
    "winner_model_b": 1,
    "tie": 2
}
        
def get_dataset() -> tuple[pd.DataFrame, pd.DataFrame]:
    df = pd.read_csv("data/train_pairs.csv")
    df_train, df_test = train_test_split(df, test_size=0.2, random_state=0, stratify=df["label"])
 
    return df_train, df_test


if __name__ == "__main__":
    df, _ = get_dataset()
    print(df.head()["left"][3] ) 
