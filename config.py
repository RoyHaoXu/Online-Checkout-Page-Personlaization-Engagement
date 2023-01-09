import itertools
###################################################################################################
#################################### AFTER GETTING THE MODEL ######################################
###################################################################################################
### demo.py
MODEL_DEMO = 'model.pkl'
MODEL_FEATURES_DEMO = 'model_features.pkl'
MODEL_PATH_DEMO = 'C:\\VDI 2020 ZealStrat Project Folder - Group Collaboration Space\\Heap_Boosted_Tree\\model\\' + MODEL_DEMO
MODEL_FEATURES_PATH_DEMO = 'C:\\VDI 2020 ZealStrat Project Folder - Group Collaboration Space\\Heap_Boosted_Tree\\model\\' + MODEL_FEATURES_DEMO

DATA_DEMO = 'heap_data.csv'
DATA_PATH_DEMO = 'C:\\VDI 2020 ZealStrat Project Folder - Group Collaboration Space\\Heap_Boosted_Tree\\data\\' + DATA_DEMO


### PD_plot.py
PLOT_OUT_PUT_PATH =  'C:\\VDI 2020 ZealStrat Project Folder - Group Collaboration Space\\Heap_Boosted_Tree\\plots\\'

### model_evaluation.py 
#which model to evaluate
MODEL_EVAL = 'model.pkl'
MODEL_FEATURES_EVAL = 'model_features.pkl'
MODEL_PATH_EVAL = 'C:\\VDI 2020 ZealStrat Project Folder - Group Collaboration Space\\Heap_Boosted_Tree\\model\\' + MODEL_EVAL
MODEL_FEATURES_PATH_EVAL = 'C:\\VDI 2020 ZealStrat Project Folder - Group Collaboration Space\\Heap_Boosted_Tree\\model\\' + MODEL_FEATURES_EVAL

X_EVAL = 'X.csv.gz'
y_EVAL = 'y.csv.gz'
X_HOLD_OUT_EVAL = 'X_hold_out.csv.gz'
y_HOLD_OUT_EVAL = 'y_hold_out.csv.gz'

X_PATH_EVAL = 'C:\\VDI 2020 ZealStrat Project Folder - Group Collaboration Space\\Heap_Boosted_Tree\\data\\' + X_EVAL
X_HOLD_OUT_PATH_EVAL = 'C:\\VDI 2020 ZealStrat Project Folder - Group Collaboration Space\\Heap_Boosted_Tree\\data\\' + X_HOLD_OUT_EVAL
y_PATH_EVAL = 'C:\\VDI 2020 ZealStrat Project Folder - Group Collaboration Space\\Heap_Boosted_Tree\\data\\' + y_EVAL
y_HOLD_OUT_PATH_EVAL = 'C:\\VDI 2020 ZealStrat Project Folder - Group Collaboration Space\\Heap_Boosted_Tree\\data\\' + y_HOLD_OUT_EVAL

### model_evaluation_session.py
MODEL_EVAL_SESSION = 'model.pkl'
MODEL_FEATURES_EVAL_SESSION = 'model_features.pkl'
MODEL_PATH_EVAL_SESSION = 'C:\\VDI 2020 ZealStrat Project Folder - Group Collaboration Space\\Heap_Boosted_Tree\\model\\' + MODEL_EVAL
MODEL_FEATURES_PATH_EVAL_SESSION = 'C:\\VDI 2020 ZealStrat Project Folder - Group Collaboration Space\\Heap_Boosted_Tree\\model\\' + MODEL_FEATURES_EVAL
TESTING_FEATURE_IMPORTANCE_SAVE_PATH =  'C:\\VDI 2020 ZealStrat Project Folder - Group Collaboration Space\\Heap_Boosted_Tree\\plots\\testing_feature_importance.png'
TRAINING_FEATURE_IMPORTANCE_SAVE_PATH = 'C:\\VDI 2020 ZealStrat Project Folder - Group Collaboration Space\\Heap_Boosted_Tree\\plots\\training_feature_importance.png'

HOLD_OUT_USER_ID_EVAL_SESSION = 'hold_out_user_id.pkl'
HOLD_OUT_USER_ID_PATH_EVAL_SESSION = 'C:\\VDI 2020 ZealStrat Project Folder - Group Collaboration Space\\Heap_Boosted_Tree\\data\\' + HOLD_OUT_USER_ID_EVAL_SESSION
PREPARED_DATA_EVAL_SESSION = 'data_prepared.csv.gz'
PREPARED_DATA_PATH_EVAL_SESSION = 'C:\\VDI 2020 ZealStrat Project Folder - Group Collaboration Space\\Heap_Boosted_Tree\\data\\' + PREPARED_DATA_EVAL_SESSION



###################################################################################################
####################################### GETTING THE MODEL #########################################
###################################################################################################
### pipeline control
DATA_PREPARED = True
MODEL_TRAINED = False

### data preparing related
# starting and ending event flags
STARTING_EVENT = 'ipq_application_prequalification__pageview_prequalification' 
ENDING_EVENT = 'ipq_application_prequalification___prequalification_submit_check_for_prequalified_offer'
INFORMATIVE_LANDING_PAGES = ['www.onemainfinancial.com/',
       'www.onemainfinancial.com/prequalification',
       'www.onemainfinancial.com/branches']

# data and model paths
DATA = 'heap_data.csv'
DATA_PATH = 'C:\\VDI 2020 ZealStrat Project Folder - Group Collaboration Space\\Heap_Boosted_Tree\\data\\' + DATA
TIMEZONE_MAP = 'us_timezone.csv'
TIMEZONE_MAP_PATH = 'C:\\VDI 2020 ZealStrat Project Folder - Group Collaboration Space\\Heap_Boosted_Tree\\data\\' + TIMEZONE_MAP
PREPARED_DATA = 'data_prepared.csv.gz'
PREPARED_DATA_PATH = 'C:\\VDI 2020 ZealStrat Project Folder - Group Collaboration Space\\Heap_Boosted_Tree\\data\\' + PREPARED_DATA


### training related
MODEL = 'model.pkl'
MODEL_PATH = 'C:\\VDI 2020 ZealStrat Project Folder - Group Collaboration Space\\Heap_Boosted_Tree\\model\\' + MODEL
MODEL_FEATURES = 'model_features.pkl'
MODEL_FEATURES_PATH =  'C:\\VDI 2020 ZealStrat Project Folder - Group Collaboration Space\\Heap_Boosted_Tree\\model\\' + MODEL_FEATURES
MODEL_OPTIMAL_CONFIG = 'optimal_config.pkl'
MODEL_OPTIMAL_CONFIG_PATH = 'C:\\VDI 2020 ZealStrat Project Folder - Group Collaboration Space\\Heap_Boosted_Tree\\model\\' + MODEL_OPTIMAL_CONFIG
X = 'X.csv.gz'
y = 'y.csv.gz'
X_HOLD_OUT = 'X_HOLD_OUT.csv.gz'
y_HOLD_OUT = 'y_HOLD_OUT.csv.gz'
X_PATH = 'C:\\VDI 2020 ZealStrat Project Folder - Group Collaboration Space\\Heap_Boosted_Tree\\data\\' + X
y_PATH = 'C:\\VDI 2020 ZealStrat Project Folder - Group Collaboration Space\\Heap_Boosted_Tree\\data\\' + y
X_HOLD_OUT_PATH = 'C:\\VDI 2020 ZealStrat Project Folder - Group Collaboration Space\\Heap_Boosted_Tree\\data\\' + X_HOLD_OUT
y_HOLD_OUT_PATH = 'C:\\VDI 2020 ZealStrat Project Folder - Group Collaboration Space\\Heap_Boosted_Tree\\data\\' + y_HOLD_OUT
HOLD_OUT_USER_ID = 'hold_out_user_id.pkl'
HOLD_OUT_USER_ID_PATH = 'C:\\VDI 2020 ZealStrat Project Folder - Group Collaboration Space\\Heap_Boosted_Tree\\data\\' + HOLD_OUT_USER_ID
RESPONSE_NAME = 'reached'
IRRELEVANT_COLUMN = ['reached', 'session_id', 'user_id']

# CV config
max_depth = [6, 7, 8, 9, 10]
n_estimators = [50, 100, 300, 500]
learning_rate = [0.04, 0.05, 0.06, 0.07, 0.08]
eval_metric = ['logloss']

a = [max_depth, n_estimators, learning_rate, eval_metric]
PARAMETERS = [{'max_depth': e[0],
            'n_estimators': e[1],
            'learning_rate': e[2],
            'eval_metric': e[3]}
           for e in itertools.product(*a)]

CV_SEED = 527
SPLIT_SEED = 1

# optimal config if already know
OPTIMAL_CONFIGS = {'max_depth': 8, 
                     'n_estimators': 100, 
                     'learning_rate': 0.05, 
                     'eval_metric': 'logloss'}


#K fold number
K = 3