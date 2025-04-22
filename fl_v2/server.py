from federate_agregation import federated_averaging
from logistic_regression import FederatedLogisticRegression
import pandas as pd


def setup_local_models(df):
    lr=FederatedLogisticRegression(df['dataframe'], local_epochs=5)
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



def server_fed_learning(*dfs):
    dataframes=[]
    for df in enumerate(dfs):
        df = setup_local_models(df)
        dataframes.append(df)
    
    agregated_coefs = federated_averaging(dataframes)
    
    for i, df in enumerate(dataframes):
        df["coefs"]=agregated_coefs
        coeffs, intercept = df["coefs"].get("Intercept", 0), df["coefs"]
        




def main():
    df_fr= pd.read_csv('data/french_data.csv')
    df_be= pd.read_csv('data/belgium_data.csv')
    

    dict_fr= {'dataframe':df_fr[:50],
            'X_test':None,
            'y_test':None,
            'y_proba_local':None,
            'y_pred_local':None,
            'y_pred_global':None,
            'y_proba_global':None,
            'best_threshold':None,
            'coefs':None}
    dict_be= {'dataframe':df_be[:50],
        'X_test':None,
        'y_test':None,
        'y_proba_local':None,
        'y_pred_local':None,
        'y_pred_global':None,
        'y_proba_global':None,
        'best_threshold':None,
        'coefs':None}

    dict_fr=setup_local_models(dict_fr)

if __name__ == "__main__":
    main()