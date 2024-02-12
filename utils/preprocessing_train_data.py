import pandas as pd
import warnings
warnings.filterwarnings('ignore')


def fix_train_dataset(path: str) -> pd.DataFrame:
    """
    Removes skips and samples that the author didn't mark up.

    :param path: path to the training .csv file;
    :return: corrected pd.DataFrame.
    """
    df_train = pd.read_csv(path)

    df_train.dropna(subset=['tweet_text'], inplace=True)
    df_train.drop(columns=['emotion_in_tweet_is_directed_at'], inplace=True)
    df_train.drop_duplicates(subset=['tweet_text'], inplace=True)
    condition = df_train['is_there_an_emotion_directed_at_a_brand_or_product'] != "I can't tell"
    df_train = df_train[condition].reset_index(drop=True).copy(deep=True)

    return df_train


if __name__ == '__main__':
    cur_path = 'train.csv'
    print(fix_train_dataset(cur_path).head())

