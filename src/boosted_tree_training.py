import itertools
import random
import pickle

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.base import clone
from sklearn.model_selection import KFold
from xgboost import XGBClassifier


class boosted_tree_training_pipeline():
    """Boosted tree training pipeline."""

    def __init__(self, data, response_name, irrelevant_column,
                 split_percent=0.2, split_seed=1,
                 optimal_config=None,
                 K=5, CV_seed=1, parameters=None, CV_print_or_not=False):
        """

        Args:
            data (:obj: `pandas.DataFrame`): all data
            response_name (str): name of the y response variable
            split_percent (float): train test split percent
            split_seed (int): random seed
            optimal_config (dict): optimal configuration for hyperparameters
            K (int): fold number for CV
            CV_seed (int): random seed for CV
            parameters (dict): hyperparameters sets to perform CV on
            CV_print_or_not (bool): whether to do CV or not
        """
        self.data = data.fillna(0)  # fillna
        self.data.set_index('user_id', inplace=True, drop = True)  # set user_id as index
        self.irrelevant_column = irrelevant_column
        self.response_name = response_name
        self.variable_list = [e for e in list(self.data.columns) if e not in irrelevant_column]
        self.split_percent = split_percent
        self.split_seed = split_seed
        self.optimal_config = optimal_config
        self.K = K
        self.CV_seed = CV_seed
        if not parameters:
            max_depth = [3, 4, 5]
            n_estimators = [100, 300, 500, 1000]
            learning_rate = [0.01, 0.05, 0.1, 0.3, 0.5]
            eval_metric = ['logloss']

            a = [max_depth, n_estimators, learning_rate, eval_metric]
            self.parameters = [{'max_depth': e[0],
                                'n_estimators': e[1],
                                'learning_rate': e[2],
                                'eval_metric': e[3]}
                               for e in itertools.product(*a)]
        else:
            self.parameters = parameters
        self.CV_print_or_not = CV_print_or_not

    def _train_test_split_on_duplicated_index(self):
        """Train test split using unique index as identifier."""
        random.seed(self.split_seed)

        hold_out_user_ids = random.sample(list(self.data.index.unique()),
                                         round(len(self.data.index.unique()) * self.split_percent))
        used_user_ids = list(set(self.data.index.unique()) - set(hold_out_user_ids))

        self.data_hold_out = self.data.loc[hold_out_user_ids, :]
        self.data_used = self.data.loc[used_user_ids, :]

        self.X = self.data_used.loc[:, self.variable_list]
        self.y = self.data_used.loc[:, [self.response_name]]
        self.X_hold_out = self.data_hold_out.loc[:, self.variable_list]
        self.y_hold_out = self.data_hold_out.loc[:,[self.response_name]]
        # record used user_ids so we can use the hold out set for demo
        self.hold_out_user_ids = list(set(hold_out_user_ids))


    def _cv_index_generation_on_duplicated_index(self):
        """CV folds partition using unique index as identifier."""
        def _partition(list_in, n, seed):
            random.seed(seed)
            random.shuffle(list_in)
            return [list_in[i::n] for i in range(n)]

        indices = list(self.data_used.index.unique())
        return _partition(indices, self.K, self.CV_seed)

    def _cross_validation_for_boosted_tree(self):
        """Perform cross validation."""
        # get k fold index
        K_user_id_folds = self._cv_index_generation_on_duplicated_index()

        # CV
        result_cols = [tuple(p.values()) for p in self.parameters]
        result = pd.DataFrame(columns=result_cols, index=range(0, self.K))

        for i, test_user_ids in enumerate(K_user_id_folds):

            print('{} Fold Started.'.format(i + 1))

            train_user_ids = list(set(self.X.index) - set(test_user_ids))

            for parameter in self.parameters:
                print(parameter)
                # tree model
                model_init = XGBClassifier(
                    max_depth=parameter['max_depth'],
                    n_estimators=parameter['n_estimators'],
                    learning_rate=parameter['learning_rate'],
                    eval_metric=parameter['eval_metric']
                )

                fit = clone(model_init).fit(self.X.loc[train_user_ids, :], self.y.loc[train_user_ids,:])

                y_pred_test = fit.predict(self.X.loc[test_user_ids, :])
                result.loc[i, tuple(parameter.values())] = accuracy_score(self.y.loc[test_user_ids,:], y_pred_test)

        opt_tuple = result.mean().idxmax()
        self.optimal_config = {k: v for k, v in zip(parameter.keys(), opt_tuple)}

        if self.CV_print_or_not:
            printing_result = result.mean()
            for e in list(printing_result.index):
                print(e, printing_result[e])

    def train_model(self):
        """Calling function to train the model."""
        # train test split
        self._train_test_split_on_duplicated_index()

        # CV
        if not self.optimal_config:
            # CV
            print('Cross Validation Started.')
            self._cross_validation_for_boosted_tree()
        print('Cross Validation Finished.')

        # final model config
        model_init = XGBClassifier(
            max_depth=self.optimal_config['max_depth'],
            n_estimators=self.optimal_config['n_estimators'],
            learning_rate=self.optimal_config['learning_rate'],
            eval_metric=self.optimal_config['eval_metric']
        )

        # fit model
        print('Final Model Training Started.')
        model = clone(model_init).fit(self.X, self.y)
        print('Final Model Training Finished.')
        return model