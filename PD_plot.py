import pickle

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.inspection import partial_dependence

import config
from src.boosted_tree_evaluation import evaluation

def _pretty_print_list(l, number_per_row = 3):
    for i in range(0, len(l), number_per_row):
        index_set = list(range(i,i+number_per_row))
        element_set = l[i:i+number_per_row]
        combined_set = [str(index) + ': ' + element for index,element in zip(index_set, element_set)]
        s = str('{:<65s} '*len(element_set))
        print(s.format(*(e for e in combined_set)))

if __name__ == '__main__':
    #load model, feature names and X
    with open(config.MODEL_PATH_EVAL, 'rb') as file:
        model = pickle.load(file)
    with open(config.MODEL_FEATURES_PATH_EVAL, 'rb') as file:
        features = pickle.load(file)    
    X = pd.read_csv(config.X_PATH_EVAL, compression = 'gzip')

    #chose feature
    print_manu = input('Do you want to see the features list? (Y/N)')
    if print_manu == 'Y':
        _pretty_print_list(features)
    elif print_manu == 'N':
        pass
    else:
        print('Invalid option.')
        exit

    featue_input = input('Please select a feature index: (if want to input multiple index, seperate them by ",")')
    if ',' in featue_input:
        featue_numbers = featue_input.split(',')
    else:
        featue_numbers = [featue_input]


    #plot and save
    for featue_number in featue_numbers:
        y,x = partial_dependence(
            estimator=model,
            X=X,
            features = [int(featue_number)]
        )
        x = x[0]
        y = y[0]

        fig, ax = plt.subplots(1,1)
        ax.plot(x,y)
        ax.set_title('PD plot for {0}'.format(features[int(featue_number)]))
        plt.savefig(config.PLOT_OUT_PUT_PATH + features[int(featue_number)].replace('/','').replace('.com','com').replace('www.','www'))
        print('PD plots for {} is generated and saved.'.format(features[int(featue_number)]))

    print('All PD plots generated and saved.')
    
