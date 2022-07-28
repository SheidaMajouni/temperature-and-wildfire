import collections
from pathlib import Path
from typing import Union, Optional
from enum import Enum
from sklearn.metrics.pairwise import euclidean_distances

import pandas as pd

import numpy as np

from b_data_profile import *
from a_load_file import read_dataset

pd.options.mode.chained_assignment = None  # default='warn'


##############################################
# Example(s). Read the comments in the following method(s)
##############################################
class WrongValueNumericRule(Enum):
    """
    You'll use these enumeration possibilities in your implemented methods below
    """
    MUST_BE_POSITIVE = 0
    MUST_BE_NEGATIVE = 1
    MUST_BE_GREATER_THAN = 2
    MUST_BE_LESS_THAN = 3


class DistanceMetric(Enum):
    """
    You'll use these enumeration possibilities in your implemented methods below
    """
    EUCLIDEAN = 0
    MANHATTAN = 1


class OutlierAndNanFixMethod(Enum):
    """
    I'll use these enumeration possibilities in my implemented methods below
    """
    REMOVE_ROW = 0
    REPLACE_MEAN = 1
    # FIND_MOST_SIMILAR_DATA = 2


##############################################
# Implement all the below methods
# All methods should be dataset-independent, using only the methods done in the assignment
# so far and pandas/numpy/sklearn for the operations
##############################################
def fix_numeric_wrong_values(df: pd.DataFrame,
                             column: str,
                             must_be_rule: WrongValueNumericRule,
                             must_be_rule_optional_parameter: Optional[float] = None) -> pd.DataFrame:
    """
    This method should fix the wrong_values depending on the logic you think best to do and using the rule passed by parameter.
    Remember that wrong values are values that are in the dataset, but are wrongly inputted (for example, a negative age).
    Here you should only fix them (and not find them) by replacing them with np.nan ("not a number", or also called "missing value")
    :param df: Dataset
    :param column: the column to be investigated and fixed
    :param must_be_rule: one of WrongValueNumericRule identifying what rule should be followed to flag a value as a wrong value
    :param must_be_rule_optional_parameter: optional parameter for the "greater than" or "less than" cases
    :return: The dataset with fixed column
    """
    # have a copy of source df keep it safe from changing by initializing a new pointer for it.
    temp_df = df.copy()
    my_column = temp_df[column]
    mean = get_column_mean(df, column)
    for i in range(my_column.size):
        if must_be_rule == WrongValueNumericRule.MUST_BE_GREATER_THAN:
            if my_column[i] <= must_be_rule_optional_parameter:
                my_column[i] = mean  # I change np.nan with mean of column because of part e
        elif must_be_rule == WrongValueNumericRule.MUST_BE_LESS_THAN:
            if my_column[i] >= must_be_rule_optional_parameter:
                my_column[i] = mean  # I change np.nan with mean of column because of part e
        elif must_be_rule == WrongValueNumericRule.MUST_BE_NEGATIVE:
            if my_column[i] > 0:
                my_column[i] = np.nan
        elif must_be_rule == WrongValueNumericRule.MUST_BE_POSITIVE:
            if my_column[i] < 0:
                my_column[i] = np.nan

    return temp_df


def fix_outliers(df: pd.DataFrame, column: str, outlier_method: OutlierAndNanFixMethod) -> pd.DataFrame:
    """
    This method should fix the column in respective to outliers depending on the logic you think best to do.
    Feel free to choose which logic you prefer, but if you are in doubt, use the simplest one to remove the row
    of the dataframe which is an outlier (note that the issue with this approach is when the dataset is small,
    dropping rows will make it even smaller).
    Remember that some datasets are large, and some are small, so think wisely on how to calculate outliers
    and when to remove/replace them. Also remember that some columns are numeric, some are categorical, some are binary
    and some are datetime. Use the methods in b_dataset_profile to your advantage!
    :param df: Dataset
    :param column: the column to be investigated and fixed
    :return: The dataset with fixed column
    """

    # data with a bigger or smaller value than 3*standard deviation of the column will consider as an outlier
    if column in get_numeric_columns(df) or df[
        column].dtype == np.datetime64:  # if column is numeric we can define outlier
        # apply() function calls the lambda function and applies it to every row or column of the dataframe and returns
        # a modified copy of the dataframe:
        mean = get_column_mean(df, column)
        std = get_column_standard_deviation(df, column)
        if outlier_method == OutlierAndNanFixMethod.REMOVE_ROW:
            # first replace them with nan
            df[column] = df[column].apply(lambda x: x if x < must_be_rule_optional_parameter else np.nan)
            # then dro all nan row
            df = df[df[column].notna()]
        elif outlier_method == OutlierAndNanFixMethod.REPLACE_MEAN:
            df[column] = df[column].apply(lambda x: mean if (x - mean) / std > 3 else x)
        return df
    elif column in get_text_categorical_columns(df):
        # if doesnt contain string, remove...
        # my column is ["one", "two", Nan, None]
        # a = df[column].map(type) ==> [str, str, float, none type]
        df = df[df[column].map(type) == str]  # only keep str
        return df


def fix_nans(df: pd.DataFrame, column: str, nan_fix_method: OutlierAndNanFixMethod) -> pd.DataFrame:
    """
    This method should fix all nans (missing data) depending on the logic you think best to do
    Remember that some datasets are large, and some are small, so think wisely on when to use each possible
    removal/fix/replace of nans. Also remember that some columns are numeric, some are categorical, some are binary
    and some are datetime. Use the methods in b_dataset_profile to your advantage!
    :param df: Dataset
    :param column: the column to be investigated and fixed
    :return: The fixed dataset
    """

    # **better**
    if nan_fix_method == OutlierAndNanFixMethod.REMOVE_ROW:
        df = df[df[column].notna()]
    elif nan_fix_method == OutlierAndNanFixMethod.REPLACE_MEAN:
        mean = get_column_mean(df, column)
        df[column] = df[column].apply(lambda x: mean if pd.isnull(x) else x)
    # my_column = df[column]
    # for i in range(my_column.size):
    #     if pd.isnull(my_column[i]):
    #         if nan_fix_method == OutlierAndNanFixMethod.REMOVE_ROW:
    #             df = df.drop([i], axis=0)
    #         elif nan_fix_method == OutlierAndNanFixMethod.REPLACE_MEAN:
    #             my_column[i] = get_column_mean(df, column)
    df = df.reset_index(drop=True)
    return df


def normalize_column(df_column: pd.Series) -> pd.Series:
    """
    This method should recalculate all values of a numeric column and normalise it between 0 and 1.
    :param df_column: Dataset's column
    :return: The column normalized
    """
    # since values should be between 0 and 1 the max_min normalization is implemented in this function
    maximum = df_column.max()
    minimum = df_column.min()
    temp = maximum - minimum
    df_column = df_column.apply(lambda x: (x - minimum) / temp)
    # for j in range(df_column.size):
    #     df_column[j] = (df_column[j] - minimum) / temp
    return df_column


def standardize_column(df_column: pd.Series) -> pd.Series:
    """
    This method should recalculate all values of a numeric column and standardize it between -1 and 1 with its average at 0.
    :param df_column: Dataset's column
    :return: The column standardized
    """
    # z_score normalization (standardization) is implemented in this function
    mean = np.mean(df_column)
    std = np.std(df_column)
    df_column = df_column.apply(lambda x: (x - mean) / std)
    # for j in range(df_column.size):
    #     df_column[j] = (df_column[j] - mean) / std
    return df_column


def calculate_numeric_distance(df_column_1: pd.Series, df_column_2: pd.Series,
                               distance_metric: DistanceMetric) -> pd.Series:
    """
    This method should calculate the distance between two numeric columns
    :param df_column_1: Dataset's column
    :param df_column_2: Dataset's column
    :param distance_metric: One of DistanceMetric, and for each one you should implement its logic
    :return: A new 'column' with the distance between the two inputted columns
    """
    if distance_metric == DistanceMetric.EUCLIDEAN:
        sub = abs(df_column_1 - df_column_2)
        dis = np.sqrt(np.sum(sub.pow(2)))
        return sub
    elif distance_metric == DistanceMetric.MANHATTAN:
        dis = np.sum(abs(df_column_1 - df_column_2))
        return abs(df_column_1 - df_column_2)


def calculate_binary_distance(df_column_1: pd.Series, df_column_2: pd.Series) -> pd.Series:
    """
    This method should calculate the distance between two binary columns.
    Choose one of the possibilities shown in class for future experimentation.
    :param df_column_1: Dataset's column
    :param df_column_2: Dataset's column
    :return: A new 'column' with the distance between the two inputted columns
    """
    # Hamming distance
    distance = 0
    distance_array = []
    for i in range(df_column_1.size):
        if df_column_1[i] != df_column_2[i]:
            distance += 1
            distance_array.append(True)
        else:
            distance_array.append(False)
    return pd.Series(distance_array)

    # **better**
    # ham = lambda x: distance.hamming(x[0], x[1])
    # df = pd.concat([df_column_1, df_column_2], axis=1).apply(ham, axis=1)
    # return df


if __name__ == "__main__":
    df = pd.DataFrame({'a': [1, 2, 3, None], 'b': [True, True, False, None], 'c': ['one', 'two', np.nan, None]})
    df2 = pd.DataFrame({'a': [4, 3, 5, None], 'b': [False, True, False, None], 'c': ['one', 'two', np.nan, None]})
    assert fix_numeric_wrong_values(df, 'a', WrongValueNumericRule.MUST_BE_LESS_THAN, 2) is not None
    assert fix_numeric_wrong_values(df, 'a', WrongValueNumericRule.MUST_BE_GREATER_THAN, 2) is not None
    assert fix_numeric_wrong_values(df, 'a', WrongValueNumericRule.MUST_BE_POSITIVE, 2) is not None
    assert fix_numeric_wrong_values(df, 'a', WrongValueNumericRule.MUST_BE_NEGATIVE, 2) is not None
    assert fix_outliers(df, 'c', OutlierAndNanFixMethod.REMOVE_ROW) is not None
    assert fix_nans(df, 'c', OutlierAndNanFixMethod.REMOVE_ROW) is not None
    assert normalize_column(df.loc[:, 'a']) is not None
    assert standardize_column(df.loc[:, 'a']) is not None
    assert calculate_numeric_distance(df.loc[:, 'a'], df2.loc[:, 'a'], DistanceMetric.EUCLIDEAN) is not None
    assert calculate_numeric_distance(df.loc[:, 'a'], df2.loc[:, 'a'], DistanceMetric.MANHATTAN) is not None
    assert calculate_binary_distance(df.loc[:, 'b'], df2.loc[:, 'b']) is not None
    print("ok")
