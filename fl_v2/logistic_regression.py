import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegressionCV
from scipy.special import expit  
from sklearn.metrics import roc_curve


#df_be= pd.read_csv('data/belgium_data.csv')
#df_eu= pd.read_csv('data/european_data.csv')


###BROUILLON###################################
def split_data(df, test_size=0.25):
    """
    Splits the data into training and testing sets.
    """
    df=df.dropna()
    X = df.drop('Sinistre', axis=1)
    y = df['Sinistre']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    return X_train, X_test, y_train, y_test

def youden_index_threshold(y_true, y_proba):
    fpr, tpr, thresholds = roc_curve(y_true, y_proba)
    specificity = 1 - fpr
    youden_index = tpr + specificity - 1
    best_index = youden_index.argmax()
    best_threshold = thresholds[best_index]
    return best_threshold, youden_index[best_index]


def normalize_dataframe(X_train: pd.DataFrame, X_test: pd.DataFrame):
    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns, index=X_train.index)
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns, index=X_test.index)
    return X_train_scaled, X_test_scaled

    
def logistic_regression(X_train, y_train, X_test,y_test, Cs=np.logspace(-4,4,100)):
    exposure_test, exposure_train= X_test['Exposure'], X_train['Exposure']
    X_train,X_test= X_train.drop('Exposure', axis=1), X_test.drop('Exposure', axis=1)
    logreg=LogisticRegressionCV(random_state=16,
                              class_weight='balanced',# problème de classification non équilibrée
                              penalty='l2',
                              scoring='roc_auc', # on cherche à maximiser la fonction ROC car l'Accuracy n'est pas une bonne métrique ici
                              cv=5,
                              Cs=Cs,
                              max_iter=1000) 
    
    
    X_train, X_test= normalize_dataframe(X_train, X_test)
    
    logreg.fit(X_train, y_train,sample_weight=exposure_train) # car on veut qu 'Exposure soit pris en compte comme un coeff de durée de risque
    
    best_C=logreg.C_[0]
    best_alpha = 1 / best_C
    print(f"✅ Meilleur alpha : {best_alpha:.5f} (C = {best_C})")
    
    y_proba=logreg.predict_proba(X_test)[:,1]
    y_proba=y_proba.tolist()
    best_threshold, youden_index = youden_index_threshold(y_test, y_proba)
    y_pred= (y_proba >= best_threshold).astype(int)
    coefficients = dict(zip(X_train.columns, logreg.coef_[0]))
    coefficients['Intercept'] = logreg.intercept_[0]
    return y_proba, y_pred, coefficients



###BROUILLON###################################



class FederatedLogisticRegression:
    def __init__(self, df):
        self.df = df.dropna()
        self.X_train, self.X_test, self.y_train, self.y_test = self.split_data()
        self.exposure_test, self.exposure_train = self.X_test['Exposure'], self.X_train['Exposure']
        self.X_train, self.X_test = self.X_train.drop('Exposure', axis=1), self.X_test.drop('Exposure', axis=1)
        self.model = None
        self.best_threshold = None
        
        
    def split_data(self, test_size=0.25):
        """
        Splits the data into training and testing sets.
        """
        X = self.df.drop('Sinistre', axis=1)
        y = self.df['Sinistre']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
        return X_train, X_test, y_train, y_test

    def normalize_dataframe(self, X_train: pd.DataFrame, X_test: pd.DataFrame):
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns, index=X_train.index)
        X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns, index=X_test.index)
        return X_train_scaled, X_test_scaled
    
    def logistic_regression(self, Cs=np.logspace(-4,4,100)):
        logreg=LogisticRegressionCV(random_state=16,
                                  class_weight={0:1,1:10},# problème de classification non équilibrée
                                  penalty='l2',
                                  scoring='roc_auc', # on cherche à maximiser la fonction ROC car l'Accuracy n'est pas une bonne métrique ici
                                  cv=5,
                                  Cs=Cs,
                                  max_iter=1000)
        
        
        self.X_train, self.X_test= self.normalize_dataframe(self.X_train, self.X_test)
        logreg.fit(self.X_train, self.y_train,sample_weight=self.exposure_train)
        best_C=logreg.C_[0]
        best_alpha = 1 / best_C
        y_proba=logreg.predict_proba(self.X_test)[:,1]
        y_proba=y_proba.tolist()
        self.best_threshold, youden_index = youden_index_threshold(self.y_test, y_proba)
        y_pred= (y_proba >= self.best_threshold).astype(int)
        coefficients = dict(zip(self.X_train.columns, logreg.coef_[0]))
        coefficients['Intercept'] = logreg.intercept_[0]
        return y_proba, y_pred, coefficients



    def predict_with_coefficients(self, X: pd.DataFrame, coef_dict: dict, best_threshold: float):
        intercept = coef_dict.get("Intercept", 0)
        coef_dict_no_intercept = {k: v for k, v in coef_dict.items() if k != "Intercept"}
        coef_series = pd.Series(coef_dict_no_intercept)[X.columns]

        X_aligned = X.copy()
        linear_pred = np.dot(X_aligned.values, list(coef_series)) + intercept
        y_scores = expit(linear_pred)
        y_pred = (y_scores >=best_threshold).astype(int)
        print(y_pred)
        return y_scores, y_pred


def setup_local_models(df):
    lr=FederatedLogisticRegression(df['dataframe'])
    X_test= lr.X_test
    y_test= lr.y_test
    y_proba, y_pred, coefs = lr.logistic_regression()
    df['X_test']=X_test
    df['y_test']=y_test
    df['y_proba_local']=y_proba
    df['y_pred_local']=y_pred
    df['coefs']=coefs
    df['best_threshold']=lr.best_threshold
    return df

def setup_model(df, coeffs):
    df = setup_local_models(df)
    lr=FederatedLogisticRegression(df['dataframe'])
    df["y_pred_global"],df["y_proba_global"] = lr.predict_with_coefficients(df["X_test"], coeffs, best_threshold=df["best_threshold"])
    return df






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
    
    df=setup_model(dict_fr, coefs_test)
    
if __name__ == "__main__":
    main()