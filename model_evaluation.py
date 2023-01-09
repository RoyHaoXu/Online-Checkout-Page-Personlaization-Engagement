import pickle
import pandas as pd
import config
from src.boosted_tree_evaluation import evaluation

if __name__ == '__main__':
    #load model and X_hold_out,y_hold_out
    with open(config.MODEL_PATH_EVAL, 'rb') as file:
        model = pickle.load(file)  
    X_hold_out = pd.read_csv(config.X_HOLD_OUT_PATH_EVAL, compression = 'gzip')
    y_hold_out = pd.read_csv(config.y_HOLD_OUT_PATH_EVAL, compression = 'gzip').iloc[:,0]
    X = pd.read_csv(config.X_PATH_EVAL, compression = 'gzip')
    y = pd.read_csv(config.y_PATH_EVAL, compression = 'gzip').iloc[:,0]

    #evaluation
    print('Testing dataset:')
    evaluation(model, X_hold_out, y_hold_out, config.TESTING_FEATURE_IMPORTANCE_SAVE_PATH)
    print('Training dataset:')
    evaluation(model, X, y, config.TRAINING_FEATURE_IMPORTANCE_SAVE_PATH)