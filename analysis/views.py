import re
from django.conf import settings
from django.shortcuts import redirect, render
import numpy as np
import pandas as pd
from analysis.forms import InspectDataForm
from django.contrib import messages

# helper functions
def get_dtypes(df: pd.DataFrame) -> dict:
    dtypes_dict = df.dtypes.apply(lambda x: x.name).to_dict()
    return dict(sorted(dtypes_dict.items()))

def read_csv_to_df(path):
    return pd.read_csv("./data/original_data.csv", encoding="latin-1")

def column_name_norm(df: pd.DataFrame) -> pd.DataFrame: # camel case to spaced chars
    for col in df.columns:
        col_lower = col[0].lower() + col[1:]
        cap_chars = re.findall(r'[A-Z]', col_lower)
        if len(cap_chars) == 1:  # knowing that columns are in camelCase form (only one capital char)
            df.rename(columns={col: col[:col.index(
                cap_chars[0])] + " " + col[col.index(cap_chars[0]):]}, inplace=True)
    return df

def n_head_rows(df: pd.DataFrame, n: int) -> dict:
    df = column_name_norm(df=df)
    return df.head(n).to_dict(orient='split')


def n_tail_rows(df: pd.DataFrame, n: int) -> dict:
    df = column_name_norm(df=df)
    return df.tail(n).to_dict(orient='split')

def col_stats(df: pd.DataFrame, include=None):
    stats_df = df.describe(include=include)
    missing = {}

    for c in stats_df.columns:
        missing[c] = len(df.loc[df[c].isna()])
    
    df = column_name_norm(df)

    stats_dict = pd.concat([stats_df, pd.DataFrame(missing, index=["missing"])]).to_dict(orient="split")

    for col in stats_dict["index"]:
        stats_dict["data"][stats_dict["index"].index(col)] = [col] + stats_dict["data"][stats_dict["index"].index(col)]


    return stats_dict

# view functions

def details(request):

    context = {
        "subtitle": "≫ EDA & Preprocessing"
    }

    df = read_csv_to_df(settings.CSV_UPLOAD_PATH)

    context["nr_rows_total"] = df.shape[0]
    context["nr_cols"] = df.shape[-1]


    if request.method == "POST":

        if request.POST.get("order") == "first":
            context["data_dict"] = n_head_rows(
                df=df, n=int(request.POST.get("nr")))
        else:
            context["data_dict"] = n_tail_rows(
                df=df, n=int(request.POST.get("nr")))

        context["nr_rows"] = int(request.POST.get("nr")) + 1

        form = InspectDataForm({
            "order": request.POST.get("order"),
            "nr": request.POST.get("nr")
        })

    else:
        form = InspectDataForm()

    context["form"] = form

    return render(request, "analysis/inspect.html", context=context)


def feature_details(request):

    df = read_csv_to_df(settings.CSV_UPLOAD_PATH)

    context = {
        "subtitle": "≫ EDA & Preprocessing",
        "dtypes": get_dtypes(df),
        "nr_cols": 2,
    }

    context["nr_rows"] = len(context["dtypes"])

    return render(request, "analysis/features.html", context=context)


def stats(request):

    df = read_csv_to_df(settings.CSV_UPLOAD_PATH)

    context = {
        "subtitle": "≫ EDA & Preprocessing",
        "nr_rows":df.shape[0],
        "nr_cols": df.shape[-1],
    }

    context["numerical"] = col_stats(df=df)
    context["numerical_col_nr"] = len(context["numerical"]["columns"]) + 1
    context["numerical_col_nr_to_display"] = context["numerical_col_nr"] - 1
    context["numerical_row_nr"] = len(context["numerical"]["data"])

    context["categorical"] = col_stats(df=df, include='object')
    context["categorical_col_nr"] = len(context["categorical"]["columns"]) + 1
    context["categorical_col_nr_to_display"] = context["categorical_col_nr"] - 1
    context["categorical_row_nr"] = len(context["categorical"]["data"])

    return render(request, "analysis/statistics.html", context=context)

def drop_missing(request):
    df = read_csv_to_df(settings.CSV_UPLOAD_PATH)
    context = {
        "subtitle": "≫ EDA & Preprocessing",
        "missing_row_nr": df.isna().any(axis=1).sum()
    }
    context["missing_row_nr_pctg"] = "{:.2%}".format(context["missing_row_nr"] / df.shape[0])

    if request.method == "POST" and context["missing_row_nr"] > 0:
        if request.POST.get("delete_rows"):
            df_wo_missing = df.dropna()
            messages.info(request, "{} rows successfully deleted".format(len(df) - len(df_wo_missing)))
            df_wo_missing.to_csv(settings.CSV_UPLOAD_PATH, index=None)
            messages.info(request, "The CSV is updated successfully")
            return redirect("/missing")

        if request.POST.get("inspect_rows"):
            context["data_dict"] = df[df.isna().any(axis=1)].astype(object).replace(np.nan, None).to_dict("split")
            context["nr_cols"] = len(context["data_dict"]["columns"])
            context["nr_rows"] = len(context["data_dict"]["data"])

    return render(request, "analysis/missing.html", context=context)