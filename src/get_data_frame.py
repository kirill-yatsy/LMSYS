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
    X = df.drop('label', axis=1)
    y = df['label']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    # df_train, df_test = train_test_split(df, test_size=0.2, random_state=0, stratify=df["label"])
    
    train_df = pd.concat([X_train, y_train], axis=1)
    test_df = pd.concat([X_test, y_test], axis=1)

    return train_df, test_df


if __name__ == "__main__":
    df, _ = get_dataset()
    print(df.head()["left"][3] ) 
