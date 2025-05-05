import warnings
import flwr as fl
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
import utils

# Chemin vers les données Europe
data_path = "/home/onyxia/work/Federated_Learning_Milliman/data/european_data.csv"
X, y = utils.load_custom_data(data_path)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LogisticRegression(
    penalty="l2",
    max_iter=1,
    warm_start=True,
    solver="liblinear"
)

utils.set_initial_params(model, X_train.shape[1])

class EuropeClient(fl.client.NumPyClient):
    def get_parameters(self):
        return utils.get_model_parameters(model)

    def fit(self, parameters, config):
        utils.set_model_params(model, parameters)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(X_train, y_train)
        print(f"Europe training done for round {config['rnd']}")
        return utils.get_model_parameters(model), len(X_train), {}

    def evaluate(self, parameters, config):
        utils.set_model_params(model, parameters)
        loss = log_loss(y_test, model.predict_proba(X_test))
        acc = model.score(X_test, y_test)
        return loss, len(X_test), {"accuracy": acc}

fl.client.start_numpy_client("0.0.0.0:8080", client=EuropeClient())
