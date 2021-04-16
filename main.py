import os, pickle, json, shutil, io, csv
import pandas as pd
import numpy as np
import category_encoders as ce
from werkzeug.utils import secure_filename
from xgboost import XGBRegressor
from sklearn.metrics import r2_score, make_scorer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, StackingRegressor, GradientBoostingRegressor
from sklearn.linear_model import SGDRegressor, RidgeCV
from sklearn.random_projection import GaussianRandomProjection, SparseRandomProjection
from sklearn.decomposition import PCA, FastICA, TruncatedSVD
from flask import Flask, render_template, send_file, request

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

def is_csv(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() == "csv"

def check_headers(filename):
    with open(filename, newline="") as f:
        rows = csv.reader(f)
        header = next(rows)
    return True if (header == config.get("TRAIN_HEADERS")) or (header == config.get("TEST_HEADERS")) else False

def preprocess_data(file_path):
    dataframe = pd.read_csv(file_path)
    dataframe["X314 + X315"] = dataframe["X314"] + dataframe["X315"]
    X = dataframe[CATEGORICAL_COLUMNS + BINARY_COLUMNS]
    X = PICKLE_OBJECTS.get("target_encoder").transform(X)
    X = pd.DataFrame(PICKLE_OBJECTS.get("scaler").transform(X), columns = CATEGORICAL_COLUMNS + BINARY_COLUMNS)
    # Projections
    pca2_results = PICKLE_OBJECTS.get("pca").transform(X)
    ica2_results = PICKLE_OBJECTS.get("ica").transform(X)
    tsvd_results = PICKLE_OBJECTS.get("tsvd").transform(X)
    grp_results = PICKLE_OBJECTS.get("grp").transform(X)
    srp_results = PICKLE_OBJECTS.get("srp").transform(X)
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

def predict(file_path):
    if "y" not in pd.read_csv(file_path).columns:
        X = preprocess_data(file_path)
    else:
        X, _ = preprocess_data(file_path)
    ids = pd.read_csv(file_path)["ID"].values
    y_pred = np.exp(PICKLE_OBJECTS.get("model").predict(X))
    submission = pd.DataFrame()
    submission["ID"] = ids
    submission["predicted_time (in seconds)"] = np.round(y_pred, 2)
    return submission

@app.route("/", methods=["GET", "POST"])
def predict_time():
    global prediction
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            return render_template("index.html", message="No file part!")
        csv_file = request.files['file']
        # if user does not select file, browser also
        # submit an empty part without filename
        if csv_file.filename == '':
            return render_template("index.html", message="No files selected!")
        if not is_csv(csv_file.filename):
            return render_template("index.html", message="Please select a CSV file!")
        if csv_file:
            if not os.path.exists(config.get("TEMPORARY_STORAGE")):
                os.mkdir(config.get("TEMPORARY_STORAGE"))
            filename = secure_filename(csv_file.filename)
            complete_file_path = os.path.join(config.get("TEMPORARY_STORAGE"), filename)
            csv_file.save(complete_file_path)
            if check_headers(complete_file_path):
                prediction = predict(complete_file_path)
                shutil.rmtree(config.get("TEMPORARY_STORAGE"))
                return render_template("prediction.html", tables=[prediction.to_html(classes="data", header="true")])
            else:
                return render_template("index.html", message="File header does not match with the format provided in kaggle!")
    return render_template("index.html")

@app.route("/download", methods=["POST"])
def download():
    csv = prediction.to_csv(index=False, header=True)
    buf_str = io.StringIO(csv)
    buf_byt = io.BytesIO(buf_str.read().encode("utf-8"))
    return send_file(buf_byt, mimetype="text/csv", as_attachment=True, attachment_filename="predictions.csv")

if __name__ == "__main__":
    app.run(debug=True, host="127.0.0.1", port=3000)