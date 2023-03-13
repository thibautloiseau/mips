# Imports
import pandas as pd
import sklearn.metrics
import numpy as np


def samplewise_rand_index(y_true_df: pd.DataFrame, y_pred_df: pd.DataFrame) -> float:
    """Compute the rand index for each sample in the dataset and then average it"""
    y_pred_df = y_pred_df.T
    y_true_df = y_true_df.T
    individual_rand_index = []
    for row_index in range(y_true_df.values.shape[0]):
        individual_rand_index.append(sklearn.metrics.adjusted_rand_score(y_true_df.values[row_index].ravel(), y_pred_df.values[row_index].ravel()))

    return np.mean(individual_rand_index)



#The following lines show how the csv files are read
if __name__ == '__main__':
    CSV_FILE_Y_TRUE = '--------.csv'  # path of the y_true csv file
    CSV_FILE_Y_PRED = '--------.csv'  # path of the y_pred csv file
    df_y_true = pd.read_csv(CSV_FILE_Y_TRUE, index_col=0, sep=',')
    df_y_pred = pd.read_csv(CSV_FILE_Y_PRED, index_col=0, sep=',')
    df_y_pred = df_y_pred.loc[df_y_true.index]
    print(samplewise_rand_index(df_y_true, df_y_pred))
