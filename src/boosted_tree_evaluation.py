import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix
from xgboost import plot_importance


def evaluation(model, X, y, feature_importance_save_path):
    """
    Model evaluation: accuracy scores and feature importance.

    Args:
        model (:obj: sklearn model instance): model to be evaluated.
        X (np.array): X_hold_out matrix used to evaluate the model.
        y: (np.array): y_hold_out vector used to evaluate the model

    """
    # stats
    print('=' * 50)
    print('Accuracy:', accuracy_score(y, model.predict(X)))
    print('=' * 50)
    print('NIR:', max(y.value_counts()) / (y.value_counts()[0] + y.value_counts()[1]))
    print('=' * 50)
    print('Confusion Matrix:')
    print(confusion_matrix(y, model.predict(X)))
    print('-' * 50)
    tn, fp, fn, tp = confusion_matrix(y, model.predict(X)).ravel()
    print('Reach Precision: {}'.format(tp/(tp+fp))) 
    print('Reach Recall: {}'.format(tp/(tp+fn)))
    print('-' * 50)
    print('Drop Precision: {}'.format(tn/(tn+fn)))
    print('Drop Recall: {}'.format(tn/(tn+fp)))

    # importance
    print('=' * 50)
    print('Importance built-in:')

    features = [e + '_' + i for e, i in zip(X.columns.to_list(), [str(a) for a in range(0, len(X.columns.to_list()))])]
    feat_imp = model.feature_importances_
    # naming the entries using Pandas series
    feat_imp_named = pd.Series(feat_imp, index=features)
    # sort in decreasing order of feature importance
    print(feat_imp_named.sort_values(ascending=False).round(4)[:20])

    # print('=' * 50)
    # print('Importance F-score:')
    ax = plot_importance(model, max_num_features=15)
    ax.figure.savefig(feature_importance_save_path)