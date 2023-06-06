import json
import os
import pickle
import shutil
from django.conf import settings
from django.shortcuts import redirect, render
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN, KMeans
from sklearn.metrics import f1_score, mean_absolute_error, mean_squared_error, precision_score, r2_score, accuracy_score, recall_score, davies_bouldin_score, silhouette_score
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeClassifier
from prediction.forms import FeatureFormSet, ModelForm, PredictionForm
from django.contrib import messages
from django.http import HttpResponse
from sklearn.preprocessing import OrdinalEncoder, StandardScaler, MinMaxScaler
from sklearn.linear_model import LinearRegression, LogisticRegression
from django import forms


# helper functions
def read_csv_to_df(path):
    return pd.read_csv(path, encoding="latin-1")

def get_dtypes(df: pd.DataFrame) -> dict:
    return df.dtypes.apply(lambda x: x.name).to_dict()
    

def initialize_form(df):
    cols = get_dtypes(df)
    
    formset_data = {
        'form-TOTAL_FORMS': str(len(cols)),
        'form-INITIAL_FORMS': str(len(cols)),
        'form-MAX_NUM_FORMS': str(len(cols)),
    }

    col_nr = 0

    for col in cols:
        formset_data["form-" + str(col_nr) + "-field_name"] = col
        formset_data["form-" + str(col_nr) + "-dtype"] = cols[col]
        formset_data["form-" + str(col_nr) + "-include"] = False

        if cols[col] == 'object':
            formset_data["form-" + str(col_nr) + "-normalize"] = "categorical - pandas.get_dummies()"
        else:
            formset_data["form-" + str(col_nr) + "-normalize"] = "numerical - standard_scaler"
        col_nr += 1
    
    return FeatureFormSet(formset_data)


def apply_standard_scaler(df, columns: list):

    sc = StandardScaler()
    df_new = pd.DataFrame(sc.fit_transform(df[columns]), columns=sc.feature_names_in_)
    pickle.dump(sc, open(settings.MODELS_PATH + "/standard_scaler.sav", 'wb+'))

    return df_new

def apply_min_max_scaler(df, columns:list):
    mms = MinMaxScaler()
    df_new = pd.DataFrame(mms.fit_transform(df[columns]), columns=mms.feature_names_in_)
    pickle.dump(mms, open(settings.MODELS_PATH + "/min_max_scaler.sav", 'wb+'))

    return df_new

def apply_get_dummies(df, columns:list):
    category_json = {}
    for column in columns:
        df[column] = df[column].astype("category")
        category_json[column] = list(df[column].dtype.categories)
    
    # save model
    with open(settings.MODELS_PATH + "/get_dummies.json", "w+") as f:
        f.write(json.dumps(category_json))
    
    # apply dummy encoding
    for column in category_json:
        col_dtype = pd.CategoricalDtype(categories=category_json[column], ordered=False)
        df[column] = df[column].astype(col_dtype)
        
    df_new = pd.get_dummies(df[columns])
    return df_new

def apply_ordinal_encoding(df, columns:list, categories:list):
    oe = OrdinalEncoder(categories=categories)
    df_new = pd.DataFrame(oe.fit_transform(df[columns]), columns=oe.feature_names_in_)
    pickle.dump(oe, open(settings.MODELS_PATH + "/ordinal_encoder.sav", 'wb+'))
    return df_new

def split(features, target):
    return train_test_split(features, target, test_size=0.20, random_state=0)

def regression_evals(y_train_true, y_train_pred, y_test_true, y_test_pred):

    evaluation_metrics = {
        "mean absolute error (MAE)": {},
        "mean squared error (MSE)": {},
        "r square": {}
        }

    evaluation_metrics["mean absolute error (MAE)"]["train"] = mean_absolute_error(y_true=y_train_true, y_pred=y_train_pred)

    evaluation_metrics["mean absolute error (MAE)"]["test"] = mean_absolute_error(y_true=y_test_true, y_pred=y_test_pred)

    evaluation_metrics["mean squared error (MSE)"]["train"] = mean_squared_error(y_true=y_train_true, y_pred=y_train_pred)

    evaluation_metrics["mean squared error (MSE)"]["test"] = mean_squared_error(y_true=y_test_true, y_pred=y_test_pred)

    evaluation_metrics["r square"]["train"] = r2_score(y_true=y_train_true, y_pred=y_train_pred)

    evaluation_metrics["r square"]["test"] = r2_score(y_true=y_test_true, y_pred=y_test_pred)

    return evaluation_metrics


def train_regression(model, features, target):

    x_train, x_test, y_train, y_test = split(features=features, target=target)    
    
    # fit the model
    model.fit(x_train, y_train)

    # save the model
    pickle.dump(model, open(settings.TRAINED_MODEL_PATH + "/trained_model.sav", "wb+"))

    return regression_evals(y_train_true=y_train,
                            y_train_pred=model.predict(x_train),
                            y_test_true=y_test,
                            y_test_pred=model.predict(x_test))

def classification_evals(y_train_true, y_train_pred, y_test_true, y_test_pred):
    evaluation_metrics = {
        "accuracy": {},
        "recall": {},
        "precision": {},
        "F1-score": {},
        }
    
    evaluation_metrics["accuracy"]["train"] = accuracy_score(y_true=y_train_true, y_pred=y_train_pred)

    evaluation_metrics["recall"]["train"] = recall_score(y_true=y_train_true, y_pred=y_train_pred)

    evaluation_metrics["precision"]["train"] = precision_score(y_true=y_train_true, y_pred=y_train_pred)
    
    evaluation_metrics["F1-score"]["train"] = f1_score(y_true=y_train_true, y_pred=y_train_pred)

    evaluation_metrics["accuracy"]["test"] = accuracy_score(y_true=y_test_true, y_pred=y_test_pred)

    evaluation_metrics["recall"]["test"] = recall_score(y_true=y_test_true, y_pred=y_test_pred)

    evaluation_metrics["precision"]["test"] = precision_score(y_true=y_test_true, y_pred=y_test_pred)
    
    evaluation_metrics["F1-score"]["test"] = f1_score(y_true=y_test_true, y_pred=y_test_pred)
    
    return evaluation_metrics
    
def train_classification(model, features, target):
    x_train, x_test, y_train, y_test = split(features=features, target=target)

    # fit the model
    model.fit(x_train, y_train)

    # save the model
    pickle.dump(model, open(settings.TRAINED_MODEL_PATH + "/trained_model.sav", "wb+"))

    return classification_evals(y_train_true=y_train,
                                y_train_pred=model.predict(x_train),
                                y_test_true=y_test,
                                y_test_pred=model.predict(x_test))

def clustering_evals(features, labels):
    evaluation_metrics = {
        "silhouette coefficient": {},
        "Davies-Bouldin Index": {}
    }

    evaluation_metrics["silhouette coefficient"]["train"] = silhouette_score(features, labels)
    evaluation_metrics["silhouette coefficient"]["test"] = "n/a" 

    evaluation_metrics["Davies-Bouldin Index"]["train"] = davies_bouldin_score(features, labels)

    evaluation_metrics["Davies-Bouldin Index"]["test"] = "n/a"

    for label in set(labels):
        evaluation_metrics["nr of observations with label " + str(label)] = {}
        evaluation_metrics["nr of observations with label " + str(label)]["train"] = list(labels).count(label)
        evaluation_metrics["nr of observations with label " + str(label)]["test"] = "n/a"

    return evaluation_metrics

def train_clustering(model, features):

    # fit the model
    model.fit(X=features)

    # save the model
    pickle.dump(model, open(settings.TRAINED_MODEL_PATH + "/trained_model.sav", "wb+"))

    return clustering_evals(
        features=features,
        labels = model.labels_
        )

# view functions
def config_features(request): 

    df = read_csv_to_df(settings.CSV_UPLOAD_PATH)

    context = {
        "subtitle": "≫ Feature configuration",
    }

    
    context["form_set"] = initialize_form(df)

    context["nr_cols"] = 4
    context["nr_rows"] = len(df.columns) + 1

    if request.method == "GET":
        if os.path.exists(settings.MODELS_PATH):
            shutil.rmtree(settings.MODELS_PATH)
        os.mkdir(settings.MODELS_PATH)

    elif request.method == "POST":

        context["post"] = request.POST
        cols = df.columns
        cols_included = {}
        ordinal_cols_order = {}

        for col in range(len(cols)):
            if request.POST.get("form-" + str(col) + "-include"):
                cols_included[cols[col]] = request.POST.get("form-" + str(col) + "-normalize")
                if "ordinal" in cols_included[cols[col]]:
                    unique_vals = list(df[cols[col]].dropna().astype(object).unique())
                    vals_order = {}
                    for v in unique_vals:
                        v_standard = v.replace(" ", "")
                        v_standard = v_standard.replace("-", "")
                        vals_order[v] = int(request.POST.get("form-"+str(col)+"-ordinal-" + v_standard))
                    
                    ordinal_cols_order[cols[col]] = vals_order
        
        context["cols_included"] = cols_included 

        df_encoded = pd.DataFrame()
        
        # apply standard scaler
        sc_cols = list(filter(lambda c: cols_included[c] == "numerical - standard_scaler", cols_included))
        if sc_cols:
            df_encoded_sc = apply_standard_scaler(df=df, columns=sc_cols)
            df_encoded[df_encoded_sc.columns] = df_encoded_sc
        # apply min_max scaler
        min_max_cols = list(filter(lambda c: cols_included[c] == "numerical - min_max", cols_included))
        if min_max_cols:
            df_encoded_min_max = apply_min_max_scaler(df=df, columns=min_max_cols)
            df_encoded[df_encoded_min_max.columns] = df_encoded_min_max

        # apply no encoding
        no_encoded_cols = list(filter(lambda c: cols_included[c] == "numerical - no encoding", cols_included))
        if no_encoded_cols:
            df_encoded[no_encoded_cols] = df[no_encoded_cols]
            with open(settings.MODELS_PATH + "/not_encoded.txt", "w+") as f:
                for col in no_encoded_cols:
                    f.write(col + "\n")

        # apply get dummies
        dummy_cols = list(filter(lambda c: cols_included[c] == "categorical - pandas.get_dummies()", cols_included))
        if dummy_cols:
            df_encoded_dummies = apply_get_dummies(df, dummy_cols)
            df_encoded[df_encoded_dummies.columns] = df_encoded_dummies

        # apply ordinal encoding
        ordinal_cols = list(filter(lambda c: cols_included[c] == "ordinal - sklearn.OrdinalEncoder", cols_included))
        if ordinal_cols:
            ordered_categories = []
            for col in ordinal_cols_order:
                ordered_categories.append(list(dict(sorted(ordinal_cols_order[col].items(), key=lambda t: t[1])).keys()))
            df_encoded_ordinal = apply_ordinal_encoding(df, ordinal_cols, categories=ordered_categories)
            df_encoded[df_encoded_ordinal.columns] = df_encoded_ordinal

        # export as encoded dataset
        df_encoded.to_csv(settings.DATA_PATH + "/encoded_data.csv", index=None, encoding="latin-1")

        if df_encoded.shape[-1] < 2:
            messages.error(request, "At least 2 features should be selected")
        else:
            return redirect("/model-config")

    return render(request, "prediction/features.html", context=context)


def model_config(request):
    context = {
        "subtitle": "≫ Model configuration",
    }

    if os.path.exists(settings.TRAINED_MODEL_PATH):
        shutil.rmtree(settings.TRAINED_MODEL_PATH)
    os.mkdir(settings.TRAINED_MODEL_PATH)

    if request.method == "POST":

        df = read_csv_to_df(settings.DATA_PATH + "/encoded_data.csv")
        model_eval_context = {
            "subtitle": "≫ Model evaluation",
        }

        if request.POST.get("model") == "regression":

            with open(settings.MODELS_PATH + "/target_column.txt", "w+") as f:
                f.write(request.POST.get("target_column") + "\n")

            target_column = df[request.POST.get("target_column")]
            features = df.drop(columns=request.POST.get("target_column"))
            model_eval_context["trained_model"] = request.POST.get("algorithm_to_use")

            if request.POST.get("algorithm_to_use") == "linear regression":
                metrics = train_regression(
                    model= LinearRegression(),
                    features=features, 
                    target=target_column)
            elif request.POST.get("algorithm_to_use") == "SVR - rbf kernel":
                metrics = train_regression(
                    model= SVR(),
                    features=features, 
                    target=target_column)

        if request.POST.get("model") == "classification":

            target_column_name = request.POST.get("target_column") + "_" + request.POST.get("prediction_class")

            with open(settings.MODELS_PATH + "/target_column.txt", "w+") as f:
                f.write(target_column_name + "\n")

            target_column = df[target_column_name]
            columns_to_drop = [column for column in list(df.columns) if request.POST.get("target_column") + "_" in column]

            features = df.drop(columns=columns_to_drop)

            model_eval_context["trained_model"] = request.POST.get("algorithm_to_use")

            if request.POST.get("algorithm_to_use") == "logistic regression":
                metrics = train_classification(
                    model=LogisticRegression(),
                    features=features,
                    target=target_column
                )

            elif request.POST.get("algorithm_to_use") == "decision tree":
                metrics = train_classification(
                    model=DecisionTreeClassifier(),
                    features=features,
                    target=target_column
                )

        if request.POST.get("model") == "clustering":
            messages.info(request, "clustering")

            features = df

            if os.path.exists(settings.MODELS_PATH + "/target_column.txt"):
                os.remove(settings.MODELS_PATH + "/target_column.txt")

            if request.POST.get("algorithm_to_use") == "kmeans":
                metrics = train_clustering(
                    model=KMeans(n_clusters=int(request.POST.get("cluster_nr"))),
                    features=features
                )
            
            if request.POST.get("algorithm_to_use") == "dbscan":
                metrics = train_clustering(
                    model = DBSCAN(),
                    features=features
                )

        model_eval_context["metrics"] = metrics
        
        return render(request, "prediction/model_eval.html", context=model_eval_context)

    else:

        df = read_csv_to_df(settings.CSV_UPLOAD_PATH)

        cols_in_model = {}

        # import scalers
        models = ["min_max_scaler", "standard_scaler", "ordinal_encoder"]

        for model in models:

            if os.path.exists(settings.MODELS_PATH + "/" + model + ".sav"):
                scaler = pickle.load(open(settings.MODELS_PATH + "/" + model + ".sav", 'rb'))
                for f in scaler.feature_names_in_:
                    cols_in_model[f] = model
        
        # import pandas get_dummy columns 
        if os.path.exists(settings.MODELS_PATH + "/get_dummies.json"):
            with open(settings.MODELS_PATH + "/get_dummies.json", "r") as f:
                category_json = json.loads(f.read())
            for column in list(category_json.keys()):
                cols_in_model[column] = "get_dummies encoder"

        # import non encoded columns
        if os.path.exists(settings.MODELS_PATH + "/not_encoded.txt"):
            with open(settings.MODELS_PATH + "/not_encoded.txt", "r") as f:
                for column in f.readlines():
                    cols_in_model[column.split("\n")[0]] = "not encoded"

        if len(cols_in_model) == 0:
            messages.error(request, "Feature selection should be submitted first")
            return redirect("/feature-config")

        context["encodings"] = cols_in_model
        
        default_target_choice = ''
        # by default classification is selected; so an object field should be selected as default parameter of the choicelist

        choices_tuple = []
        for col in list(cols_in_model.keys()):
            choices_tuple.append((col, col))
            if df[col].dtype == "object" and not default_target_choice:
                default_target_choice = col

        if default_target_choice:
            target_col_selected = default_target_choice
        else:
            target_col_selected = choices_tuple[0][0]

        output_options = []
        for value in list(df[target_col_selected].unique()):
            output_options.append((value, value))

        context["form"] = ModelForm(cols_included=tuple(choices_tuple),
                                    default_target_choice=default_target_choice,
                                    output_options = tuple(output_options))

    return render(request, "prediction/model_config.html", context=context)

def make_predictions(request):
    context = {
        "subtitle": "≫ Make predictions",
    }

    df = read_csv_to_df(settings.CSV_UPLOAD_PATH)

    target_column = None
    if os.path.exists(settings.MODELS_PATH + "/target_column.txt"):
        with open(settings.MODELS_PATH + "/target_column.txt", "r") as f:
                target_column = f.read()
                target_column = target_column.split("\n")[0]

    cols_in_model = {}
    used_models = []

    # import scalers
    models = ["min_max_scaler", "standard_scaler", "ordinal_encoder"]

    for model in models:
        if os.path.exists(settings.MODELS_PATH + "/" + model + ".sav"):
            scaler = pickle.load(open(settings.MODELS_PATH + "/" + model + ".sav", 'rb'))
            used_models.append(scaler)
            if model in ["min_max_scaler", "standard_scaler"]:
                for f in scaler.feature_names_in_:
                    if request.method == "POST" and f != target_column:
                        cols_in_model[f] = {}
                        cols_in_model[f]["value"] = float(request.POST.get(f))
                        cols_in_model[f]["model"] = model
                    else:
                        cols_in_model[f] = forms.DecimalField(min_value=0, step_size=0.01)
            elif model in ["ordinal_encoder"]:
                for f in scaler.feature_names_in_:
                    if request.method == "POST" and f != target_column:
                        cols_in_model[f] = {}
                        cols_in_model[f]["value"] = request.POST.get(f)
                        cols_in_model[f]["model"] = model
                    else:
                        categories = []
                        for category in scaler.categories_[list(scaler.feature_names_in_).index(f)]:
                            categories.append((category, category))
                        cols_in_model[f] = forms.ChoiceField(choices=tuple(categories))
    
    # import pandas get_dummy columns 
    if os.path.exists(settings.MODELS_PATH + "/get_dummies.json"):
        with open(settings.MODELS_PATH + "/get_dummies.json", "r") as f:
                category_json = json.loads(f.read())
        
        for column in list(category_json.keys()):

            if request.method == "POST" and column != target_column:
                cols_in_model[column] = {}
                cols_in_model[column]["value"] = request.POST.get(column)
                cols_in_model[column]["model"] = "pandas.get_dummies"

            else:            
                categories = []
                for category in category_json[column]:
                    categories.append((category, category))
                cols_in_model[column] = forms.ChoiceField(choices=categories)

    # import non encoded columns
    if os.path.exists(settings.MODELS_PATH + "/not_encoded.txt"):
        with open(settings.MODELS_PATH + "/not_encoded.txt", "r") as f:
            for column in f.readlines():
                col_name = column.split("\n")[0]
                if request.method == "POST" and col_name != target_column:
                    cols_in_model[col_name] = {}
                    cols_in_model[col_name]["value"] = request.POST.get(col_name)
                    cols_in_model[col_name]["model"] = "no encoding"
                else:
                    cols_in_model[col_name] = forms.DecimalField( min_value=0, step_size=0.01)

    if request.method == "POST":

        if target_column:
            cols_in_model[target_column] = {}
            cols_in_model[target_column]["value"] = np.nan
            cols_in_model[target_column]["model"] = "n/a"

        scaled_features = pd.DataFrame()

        # apply ordinal_encoding; min_max scaling; and standard scaler
        for scaler in used_models:
            data_dict = {}
            for column in scaler.feature_names_in_:
                data_dict[column] = cols_in_model[column]["value"]
            scaled_features[scaler.feature_names_in_] = scaler.transform(pd.DataFrame.from_dict({0: data_dict}, orient="index"))

        if os.path.exists(settings.MODELS_PATH + "/get_dummies.json"):
            with open(settings.MODELS_PATH + "/get_dummies.json", "r") as f:
                dummy_encoded = json.loads(f.read())

            data_df = pd.DataFrame()
            for column in dummy_encoded:
                col_dtype = pd.CategoricalDtype(categories=dummy_encoded[column])
                data_df[column] = pd.DataFrame.from_dict({0 : {column: cols_in_model[column]["value"]}}, orient="index", dtype=col_dtype)

            data_df = pd.get_dummies(data_df)

            scaled_features[list(data_df.columns)] = data_df     

        # apply no norm columns
        data_dict = {}
        for column in list(filter(lambda elements: cols_in_model[elements]["model"] == "no encoding", cols_in_model)):
            data_dict[column] = cols_in_model[column]["value"]
        scaled_features[list(data_dict.keys())] = pd.DataFrame.from_dict({0: data_dict}, orient="index")

        # read model
        model = pickle.load(open(settings.TRAINED_MODEL_PATH + "/trained_model.sav", "rb"))
        # predict
        context["pred"] = model.predict(scaled_features[model.feature_names_in_])[0]

        context["input_values"] = {}
        for feature in scaled_features.columns:
                if feature not in cols_in_model:
                    feature_real_name = feature.split("_", maxsplit=1)[0]
                    context["input_values"][feature_real_name] = {}
                    context["input_values"][feature_real_name]["real"] = cols_in_model[feature_real_name]["value"]
                    context["input_values"][feature_real_name]["scaled"] = 'dummy_scaled'
                else:
                    context["input_values"][feature] = {}
                    context["input_values"][feature]["scaled"] = scaled_features.loc[0, feature]
                    context["input_values"][feature]["real"] = cols_in_model[feature]["value"]

        if target_column and target_column in context["input_values"]:
            context["input_values"].pop(target_column)

        if target_column and target_column.split("_", maxsplit=1)[0] in context["input_values"]:
            context["input_values"].pop(target_column.split("_", maxsplit=1)[0])

    elif request.method == "GET":
        if target_column:
            if target_column in cols_in_model:        
                cols_in_model.pop(target_column) # remove target column
            else:
                cols_in_model.pop(target_column.split("_", maxsplit=1)[0])
        context["cols_in_model"] = cols_in_model
        context["form"] = PredictionForm(field_definition=cols_in_model)
    
    if target_column:
        context["target_column"] = target_column
    else:
        context["target_column"] = "cluster"

    return render(request, "prediction/pred_form_in.html", context=context)

# API for JavaScript
def unique_vals_in_column(df, column_name):
    df = read_csv_to_df(settings.CSV_UPLOAD_PATH)

    if column_name in df.columns:
        return HttpResponse(json.dumps(list(df[column_name].dropna().astype(object).unique()), allow_nan = True))
        
    return HttpResponse()
