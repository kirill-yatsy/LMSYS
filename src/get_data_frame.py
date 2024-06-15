import pandas as pd

label_to_int = {
    "winner_model_a": 0,
    "winner_model_b": 1,
    "tie": 2
}

def get_dataset():
    df = pd.read_csv("data/train.csv")
    print(df.columns)
    df["left"] = df.apply(lambda row: f"Prompt: {row["prompt"]}\n\nResponse: {row["response_a"][0]}", axis=1)
    df["right"] = df.apply(lambda row: f"Prompt: {row["prompt"]}\n\nResponse: {row["response_b"][0]}", axis=1)

    # fill winner depending on value in 'winner_model_a', 'winner_model_b', 'winner_tie'
    df["winner"] = df.apply(
        lambda row: "winner_model_a" if row["winner_model_a"] == 1 else "winner_model_b" if row["winner_model_b"] == 1 else "tie",
        axis=1
    )

    df["label"] = df["winner"].map(label_to_int)
    df = df[["left", "right", "label"]]

    # separate into train and test
    df_train = df.sample(frac=0.8, random_state=0).reset_index(drop=True)
    df_test = df.drop(df_train.index).reset_index(drop=True)

    
    
    return df_train, df_test


if __name__ == "__main__":
    df = get_dataset()
    print(df.head()) 
