import flwr as fl
import utils
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
import pandas as pd

from flwr.server import ServerConfig

# Load test data from EU for global evaluation
X_test, y_test = utils.load_custom_data("/home/onyxia/work/Federated_Learning_Milliman/data/european_data.csv")
from sklearn.model_selection import train_test_split
X_test, _, y_test, _ = train_test_split(X_test, y_test, test_size=0.8, random_state=42)

model = LogisticRegression()
utils.set_initial_params(model, X_test.shape[1])

def get_eval_fn(model):
    def evaluate(parameters):
        utils.set_model_params(model, parameters)
        loss = log_loss(y_test, model.predict_proba(X_test))
        acc = model.score(X_test, y_test)
        return loss, {"accuracy": acc}
    return evaluate

def fit_round(rnd):
    return {"rnd": rnd}

strategy = fl.server.strategy.FedAvg(
    evaluate_fn=get_eval_fn(model),
    on_fit_config_fn=fit_round,
    min_available_clients=2,
)

config = ServerConfig(num_rounds=5)
fl.server.start_server(
    server_address="0.0.0.0:8081",  # <- mot-clé requis maintenant
    config=config,
    strategy=strategy
)
