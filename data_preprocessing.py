from pathlib import Path
import pandas as pd
import numpy as np
from c_data_cleaning import *
from b_data_profile import *
import matplotlib.pyplot as plt
import seaborn as sn


def pandas_profile(df: pd.DataFrame, result_html: str = 'report_fire_ddddddddddd.html'):
    from pandas_profiling import ProfileReport

    profile = ProfileReport(df, title="Pandas Profiling Report")
    if result_html is not None:
        profile.to_file(result_html)
    return profile.to_json()


def pandas_gui(df: pd.DataFrame, result_html: str = 'report.html'):
    from pandasgui import show
    gui = show(df)


if __name__ == "__main__":
    df_w = pd.read_excel("E:\\D\\PHD\\term1\\visual analystic\\prj\\avg weather\\temprature_avg.xlsx")
    df_h = pd.read_excel("E:\\D\\PHD\\term1\\visual analystic\\prj\\avg weather\\humidity_avg.xlsx")
    df_p = pd.read_excel("E:\\D\\PHD\\term1\\visual analystic\\prj\\avg weather\\pressure_avg.xlsx")
    df_wi = pd.read_excel("E:\\D\\PHD\\term1\\visual analystic\\prj\\avg weather\\wind_speed_avg.xlsx")

    df_w['datetime'] = pd.to_datetime(df_w['datetime'], unit='ms')
    df_w = df_w.melt(id_vars='datetime')
    df_w.columns = ['datetime', 'city', 'temperature']

    df_h['datetime'] = pd.to_datetime(df_h['datetime'], unit='ms')
    df_h = df_h.melt(id_vars='datetime')
    df_h.columns = ['datetime', 'city', 'humidity']

    df_p['datetime'] = pd.to_datetime(df_p['datetime'], unit='ms')
    df_p = df_p.melt(id_vars='datetime')
    df_p.columns = ['datetime', 'city', 'pressure']

    df_wi['datetime'] = pd.to_datetime(df_wi['datetime'], unit='ms')
    df_wi = df_wi.melt(id_vars='datetime')
    df_wi.columns = ['datetime', 'city', 'wind_speed']

    df_merge = pd.merge(df_w, df_h, how='inner', on=['datetime', 'city'])
    df_merge = pd.merge(df_merge, df_p, how='inner', on=['datetime', 'city'])
    df_merge = pd.merge(df_merge, df_wi, how='inner', on=['datetime', 'city'])

    numeric_columns = get_numeric_columns(df_merge)
    categorical_columns = get_text_categorical_columns(df_merge)

    for nc in numeric_columns:
        df_merge = fix_outliers(df_merge, nc, OutlierAndNanFixMethod.REPLACE_MEAN)
        df_merge = fix_nans(df_merge, nc, OutlierAndNanFixMethod.REMOVE_ROW)

    # df_w.to_csv("melted_weather.csv", index=False)
    # df_h.to_csv("melted_humidity.csv", index=False)
    # df_p.to_csv("melted_pressure.csv", index=False)
    # df_wi.to_csv("melted_wind.csv", index=False)
    # df_merge.loc[df_merge['city'] == "NewYork"].to_csv("weather_NewYork.csv", index=False)

    # print(df_merge)
    # fire = pd.read_csv("E:\\D\\PHD\\term1\\visual analystic\\prj\\visual data\\data_fire2.csv")
    fire = pd.read_csv("E:\\D\\PHD\\term2\\ML\\prj\\avg weather\\Fire_Weather\\second_fire_weather\\fire_weather_Denver.csv")
    # df = fire.query('city == "Denver"')
    # print(df)
    # pandas_profile(fire)
    pandas_gui(fire)