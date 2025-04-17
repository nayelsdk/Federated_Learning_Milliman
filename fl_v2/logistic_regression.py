import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegressionCV
from sklearn.linear_model import LogisticRegression
from sklearn.utils.class_weight import compute_class_weight
from sklearn.linear_model import SGDClassifier
from scipy.special import expit  
from sklearn.metrics import roc_curve
from federate_agregation import federated_averaging


np.array_split


def youden_index_threshold(y_true, y_proba):
    fpr, tpr, thresholds = roc_curve(y_true, y_proba)
    specificity = 1 - fpr
    youden_index = tpr + specificity - 1
    best_index = youden_index.argmax()
    best_threshold = thresholds[best_index]
    return best_threshold, youden_index[best_index]

def get_coefficients(model, X_train):
    """
    Get the coefficients of the model.
    """
    coefficients = dict(zip(X_train.columns, model.coef_[0]))
    coefficients['Intercept'] = model.intercept_[0]
    sorted_coefficients = dict(sorted(coefficients.items(), key=lambda item: abs(item[1]), reverse=True))
    
    return sorted_coefficients

def set_model_params(model, params):
    """
    Set the parameters of the model.
    """
    for key, value in params.items():
        if hasattr(model, key):
            setattr(model, key, value)
        else:
            raise ValueError(f"Parameter {key} not found in the model.")
    return model


class FederatedLogisticRegression:
    """Ici, nous nous concentrons sur la regression logistique logistique dans un modèle fédéré. Je récupère les coefficients du modèle de RL pour les agréger et les réinjecter avec warm_start dans le modèle de RL.
    Je fais cela pour chaque client, puis j'agrège selon différentes méthodes.
    """
    def __init__(self, df, local_epochs, federated_type="testSGD"):
        self.df = df.dropna()
        self.X_train, self.X_test, self.y_train, self.y_test = self.split_data()
        self.exposure_test, self.exposure_train = self.X_test['Exposure'], self.X_train['Exposure']
        self.X_train, self.X_test = self.X_train.drop('Exposure', axis=1), self.X_test.drop('Exposure', axis=1)
        self.model = None
        self.best_threshold = None
        self.scaler = None  
        self.local_epochs = local_epochs
        self.federated_type = federated_type


    
    def choose_model(self):
        if self.federated_type == "Averaging":
            self.model=LogisticRegression(random_state=42,
                            class_weight="balanced",# problème de classification non équilibrée
                            penalty='l2',
                            fit_intercept=True,
                            scoring='roc_auc', # on cherche à maximiser la fonction ROC car l'Accuracy n'est pas une bonne métrique ici
                            max_iter=self.local_epochs, #  warm start
                            warm_start=True,
                            C=4)
        elif self.federated_type == "testSGD":
            self.model=SGDClassifier(random_state=42,
                            penalty='l2',
                            fit_intercept=True,
                            eta0=0.01,
                            learning_rate='adaptive',
                            max_iter=self.local_epochs, #  warm start
                            warm_start=True,
                            loss="log_loss")
        
    def split_data(self, test_size=0.30):
        """
        Splits the data into training and testing sets.
        """
        X = self.df.drop('Sinistre', axis=1)
        y = self.df['Sinistre']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
        return X_train, X_test, y_train, y_test

    def normalize_dataframe(self, X_train: pd.DataFrame, X_test: pd.DataFrame):
        self.scaler = StandardScaler() 
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns, index=X_train.index)
        X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns, index=X_test.index)
        return X_train_scaled, X_test_scaled

    
    def logistic_regression(self):
        self.X_train, self.X_test = self.normalize_dataframe(self.X_train, self.X_test)
        self.choose_model()
        classes = np.unique(self.y_train)

        
        
        
        class_weights = compute_class_weight(class_weight='balanced', classes=classes, y=self.y_train)

        class_weight_dict = dict(zip(classes, class_weights))



        sample_weights = self.y_train.map(class_weight_dict)
        print(classes)
        
        self.model.partial_fit(self.X_train, self.y_train, classes=classes, sample_weight=sample_weights)
        

        y_proba = self.model.predict_proba(self.X_test)[:, 1]
        
        y_proba = y_proba*self.exposure_train
        y_proba_adjusted = y_proba.tolist()
        self.best_threshold, youden_index = youden_index_threshold(self.y_test, y_proba_adjusted)
        y_pred = (np.array(y_proba) >= self.best_threshold).astype(int)
        coefficients = get_coefficients(self.model, self.X_train)

        return y_proba, y_pred, coefficients



    def predict_with_coefficients(self, X: pd.DataFrame, coef_dict: dict, best_threshold: float):
        intercept = coef_dict.get("Intercept", 0)
        coef_dict_no_intercept = {k: v for k, v in coef_dict.items() if k != "Intercept"}
        coef_series = pd.Series(coef_dict_no_intercept)[X.columns]

        X_aligned = X.copy()
        linear_pred = np.dot(X_aligned.values, list(coef_series)) + intercept
        y_scores = expit(linear_pred)
        y_pred = (y_scores >=best_threshold).astype(int)
        return y_scores, y_pred




















def main():
    df_fr= pd.read_csv('data/french_data.csv')

    dict_fr= {'dataframe':df_fr[:50],
            'X_test':None,
            'y_test':None,
            'y_proba_local':None,
            'y_pred_local':None,
            'y_pred_global':None,
            'y_proba_global':None,
            'best_threshold':None,
            'coefs':None}
    
    coefs_test={'Power': np.float64(0.0014363473294638819),
                'DriverAge': np.float64(-0.07718851762903876),
                'Fuel_type': np.float64(0.0775480178743025),
                'Density': np.float64(0.06906436933479208),
                'Sex': np.float64(-0.013835696283748422),
                'Intercept': np.float64(-0.07318725450566628)}
    
    model=FederatedLogisticRegression(dict_fr['dataframe'], local_epochs=5)
    model.logistic_regression()
    df=(dict_fr, coefs_test)
    print(df)
if __name__ == "__main__":
    main()