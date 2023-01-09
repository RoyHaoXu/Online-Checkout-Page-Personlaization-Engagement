import pickle
import warnings

import pandas as pd
import config

from src.data_prepare import data_prepare_pipeline
from src.boosted_tree_training import boosted_tree_training_pipeline


if __name__ == "__main__":
    #disable warnings 
    warnings.filterwarnings('ignore')

    # prepare data if haven't done so already
    if not config.DATA_PREPARED:
        print('Data Preparing Started.')

        # read the raw data
        data = pd.read_csv(config.DATA_PATH)

        # read timezone map table
        timezone_map_table = pd.read_csv(config.TIMEZONE_MAP_PATH)
        timezone_map_dict = timezone_map_table.set_index('REGION').to_dict()['DIFF']

        # create data prepare pipeline instance
        data_prepare_pipline1 = data_prepare_pipeline(data,
                                                      timezone_map_dict, 
                                                      config.STARTING_EVENT, 
                                                      config.ENDING_EVENT,
                                                      config.INFORMATIVE_LANDING_PAGES)

        # prepare data
        data_prepared = data_prepare_pipline1.prepare_data()
        # dump the prepared data
        data_prepared.to_csv(config.PREPARED_DATA_PATH,
                                chunksize = 50000, 
                                compression = 'gzip',
                                index=False)

        print('Prepared Data Stored.')

    # train the model if haven't done so already
    if not config.MODEL_TRAINED:
        # read in prepared data
        data_prepared = pd.read_csv(config.PREPARED_DATA_PATH, compression = 'gzip')

        # train model
        model_training_pipline1 = boosted_tree_training_pipeline(   data_prepared, 
                                                                    config.RESPONSE_NAME, config.IRRELEVANT_COLUMN,
                                                                    K = config.K, CV_seed=config.CV_SEED, split_seed=config.SPLIT_SEED,
                                                                    optimal_config=config.OPTIMAL_CONFIGS, parameters = config.PARAMETERS, CV_print_or_not= True)
        # train model
        model = model_training_pipline1.train_model()
        
        # save Final Model
        with open(config.MODEL_PATH, 'wb') as file:
            pickle.dump(model, file)
        # save model feature order
        with open(config.MODEL_FEATURES_PATH, 'wb') as file:
            pickle.dump(list(model_training_pipline1.X.columns),file)
        print('Final Model Stored.')
        # save optimal config
        with open(config.MODEL_OPTIMAL_CONFIG_PATH, 'wb') as file:
            pickle.dump(model_training_pipline1.optimal_config,file)
        print('Optimal Config Stored.')
        

        # save train/test X,y for evaluation:
        model_training_pipline1.X.to_csv(config.X_PATH,chunksize = 50000, compression = 'gzip', index=False)
        model_training_pipline1.X_hold_out.to_csv(config.X_HOLD_OUT_PATH,chunksize = 50000, compression = 'gzip', index=False)
        model_training_pipline1.y.to_csv(config.y_PATH,chunksize = 50000, compression = 'gzip', index=False)
        model_training_pipline1.y_hold_out.to_csv(config.y_HOLD_OUT_PATH,chunksize = 50000, compression = 'gzip', index=False)
        print('X and y Stored.')

        # save holdout user id list for demo:
        with open(config.HOLD_OUT_USER_ID_PATH, 'wb') as file:
            pickle.dump(model_training_pipline1.hold_out_user_ids,file)
        print('Hold UserIDs Stored.')

    # read model 
    with open(config.MODEL_PATH, 'rb') as file:
        model = pickle.load(file)
    print('Model Loaded.')
