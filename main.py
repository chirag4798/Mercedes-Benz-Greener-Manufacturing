import os, pickle, json, shutil
import pandas as pd
import numpy as np
import category_encoders as ce
from xgboost import XGBRegressor
from sklearn.metrics import r2_score, make_scorer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, StackingRegressor, GradientBoostingRegressor
from sklearn.linear_model import SGDRegressor, RidgeCV
from sklearn.random_projection import GaussianRandomProjection, SparseRandomProjection
from sklearn.decomposition import PCA, FastICA, TruncatedSVD
from flask import Flask, render_template, request

app = Flask(__name__)

with open("config.json", "r") as f:
    config = json.load(f)

CATEGORICAL_COLUMNS = config.get("CATEGORICAL_COLUMNS")
BINARY_COLUMNS = config.get("BINARY_COLUMNS")
FINAL_FEATURES = config.get("FINAL_FEATURES")
N_COMP = config.get("N_COMP")
PICKLE_OBJECTS_PATH = config.get("PICKLE_OBJECTS_PATH")
PICKLE_OBJECTS = dict()
for pickle_object in config.get("PICKLE_OBJECTS"):
    with open(os.path.join(PICKLE_OBJECTS_PATH, pickle_object + ".pkl"), "rb") as f:
        PICKLE_OBJECTS[pickle_object] = pickle.load(f)

def preprocess_data(file_path):
    dataframe = pd.read_csv(file_path)
    dataframe["X314 + X315"] = dataframe["X314"] + dataframe["X315"]
    X = dataframe[CATEGORICAL_COLUMNS + BINARY_COLUMNS]
    X = PICKLE_OBJECTS["target_encoder"].transform(X)
    X = pd.DataFrame(PICKLE_OBJECTS["scaler"].transform(X), columns = CATEGORICAL_COLUMNS + BINARY_COLUMNS)
    # Projections
    pca2_results = PICKLE_OBJECTS["pca"].transform(X)
    ica2_results = PICKLE_OBJECTS["ica"].transform(X)
    tsvd_results = PICKLE_OBJECTS["tsvd"].transform(X)
    grp_results = PICKLE_OBJECTS["grp"].transform(X)
    srp_results = PICKLE_OBJECTS["srp"].transform(X)
    # Append decomposition components to datasets
    for i in range(1, N_COMP + 1):
        X["pca_" + str(i)] = pca2_results[:, i - 1]
        X["ica_" + str(i)] = ica2_results[:, i - 1]
        X["tsvd_" + str(i)] = tsvd_results[:, i - 1]
        X["grp_" + str(i)] = grp_results[:, i - 1]
        X["srp_" + str(i)] = srp_results[:, i - 1]
    X = X.values
    if "y" in dataframe.columns:
        y = dataframe["y"].values
        return X, y
    else:
        return X


def metric_pipeline(file_path):
    if "y" not in pd.read_csv(file_path).columns:
        raise Exception("No target variable found to compute R2-score")
    else:
        X, y = preprocess_data(file_path)
        y_pred = np.exp(PICKLE_OBJECTS["model"].predict(X))
        score = r2_score(y, y_pred)
        return score


def inference_pipeline(file_path):
    if "y" not in pd.read_csv(file_path).columns:
        X = preprocess_data(file_path)
    else:
        X, _ = preprocess_data(file_path)
    ids = pd.read_csv(file_path)["ID"].values
    y_pred = np.exp(PICKLE_OBJECTS["model"].predict(X))
    submission = pd.DataFrame()
    submission["ID"] = ids
    submission["y"] = y_pred
    return submission


def predict(file_path):
    if "y" not in pd.read_csv(file_path).columns:
        X = preprocess_data(file_path)
    else:
        X, _ = preprocess_data(file_path)
    y_pred = np.exp(PICKLE_OBJECTS["model"].predict(X))
    return y_pred

@app.route("/")
def main():
    return render_template("index.html")

@app.route("/predict", methods=["GET", "POST"])
def predict_time():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            print('No file part')
            return render_template("index.html")
        csv_file = request.files['file']
        # if user does not select file, browser also
        # submit an empty part without filename
        if csv_file.filename == '':
            print('No selected file')
            return render_template("index.html")
        else:
            if not os.path.exists(config["TEMPORARY_STORAGE"]):
                os.mkdir(config["TEMPORARY_STORAGE"])
            complete_file_path = os.path.join(config["TEMPORARY_STORAGE"], csv_file.filename)
            csv_file.save(complete_file_path)
            y_pred = predict(complete_file_path)
            shutil.rmtree(config["TEMPORARY_STORAGE"])
            return f"Predicted Times: {y_pred}"

if __name__ == "__main__":
    app.run(debug=True)