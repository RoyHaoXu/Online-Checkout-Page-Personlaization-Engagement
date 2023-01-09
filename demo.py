import pickle
import warnings

import pandas as pd
import numpy as np
import config
from tabulate import tabulate

from src.data_prepare import data_prepare_pipeline
from src.boosted_tree_training import boosted_tree_training_pipeline

if __name__ == '__main__':
    # read data and model
    data = pd.read_csv(config.DATA_PATH_DEMO) 
    with open(config.MODEL_PATH_DEMO, 'rb') as file:
        model = pickle.load(file)
    with open(config.MODEL_FEATURES_PATH_DEMO, 'rb') as file:
        model_features = pickle.load(file)

    # get the raw data for the chosen session
    session = input('Choose a session id: ')
    data_session = data[data['SESSION_ID'] == int(session)]

    # prepare data
    timezone_map_table = pd.read_csv(config.TIMEZONE_MAP_PATH)
    timezone_map_dict = timezone_map_table.set_index('REGION').to_dict()['DIFF']
    data_prepare_pipline1 = data_prepare_pipeline(data_session,
                                                    timezone_map_dict, 
                                                    config.STARTING_EVENT, 
                                                    config.ENDING_EVENT,
                                                    config.INFORMATIVE_LANDING_PAGES)

    # prepare data
    data_prepared = data_prepare_pipline1.prepare_data()
    X = data_prepared.drop(['reached','user_id','session_id'], axis = 1)
    y = data_prepared['reached']

    #add missing feature
    for new_col in np.setdiff1d(model_features,X.columns):
        X[new_col] = np.nan 
    #reorder 
    X = X.reindex(model_features, axis=1)

    #get final table with probability
    probability = pd.DataFrame(model.predict_proba(X))
    final_table = data_session.loc[:,["EVENT_NAME", "EVENT_CREATED_AT"]].reset_index(drop = True)
    final_table["Probability"] = np.nan
    final_table["Probability"] = probability[1].append(pd.Series("ended")).reset_index(drop = True)

    #print the result row by row.
    for i in range(len(final_table)):
        print(tabulate(final_table.iloc[[i],:], headers='keys', tablefmt='psql'))
        a = input("Please hit enter for next row.")