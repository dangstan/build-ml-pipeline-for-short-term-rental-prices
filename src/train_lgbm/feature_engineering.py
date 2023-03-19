import pandas as pd
import numpy as np
from boruta import BorutaPy
from lightgbm import LGBMRegressor

def delta_date_feature(dates):
    """
    Given a 2d array containing dates (in any format recognized by pd.to_datetime), it returns the delta in days
    between each date and the most recent date in its column
    """
    date_sanitized = pd.DataFrame(dates).apply(pd.to_datetime)
    return date_sanitized.apply(lambda d: (d.max() -d).dt.days, axis=0).to_numpy()


def reducing_features(df):

    run = wandb.init()

    artifact_local_path = run.use_artifact('dangstan/nyc_airbnb/featurized:latest', type='featurized').download()

    boruta_features = pd.read_json(artifact_local_path+'/dummies.json')[0].values.tolist()

    return df[['last_review','name','price']+boruta_features]