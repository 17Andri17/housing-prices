from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_classif, RFE, mutual_info_regression, f_regression
import numpy as np
import pandas as pd


def as_dataframe(X, feature_names):
    if isinstance(X, pd.DataFrame):
        df = X.copy()
        if feature_names is not None:
            df = df.loc[:, list(feature_names)]
        return df
    return pd.DataFrame(X, columns=list(feature_names))

class StaticFeatureSelection:
    '''
    Basic feature selection methods not including the
    response variable: `feature_correlation` and `variance_threshold`
    
    '''
    def __init__(self, X, feature_names):
        self.X = X
        self.feature_names = feature_names
        self.df = as_dataframe(X, feature_names)

    def feature_correlation(self, threshold=0.9):
        corr_matrix = self.df.corr().abs().to_numpy()
        rows, cols = np.triu_indices(len(self.feature_names))
        self.correlation_pairs = [(self.feature_names[i], self.feature_names[j], corr_matrix[i, j]) for i, j in zip(rows, cols) if corr_matrix[i, j] > threshold and i!=j]
        print(f'Found {len(self.correlation_pairs)} correlated pairs!')

    def variance_threshold(self, threshold=0.01):
        selector = VarianceThreshold(threshold) 
        selector.fit(self.X)
        self.low_variance = self.feature_names[np.where(~selector.get_support())]


class DynamicFeatureSelection:
    '''
    Feature selection methods involving the response variable (y).
    Includes F-test and Mutual Information for both regression and classification.
    '''
    def __init__(self, X, y, feature_names):
        self.X = X
        self.y = y
        self.feature_names = np.array(feature_names)
        self.scores_ = None
        self.results_df = None

    def calculate_f_test(self):
        selector = SelectKBest(score_func=f_regression, k='all')
        selector.fit(self.X, self.y)
        
        self.scores_ = selector.scores_
        self._create_results_df('F-Score')
        return self.results_df

    def calculate_mutual_info(self, discrete_features='auto', random_state=42):
        mi_scores = mutual_info_regression(self.X, self.y, discrete_features=discrete_features, random_state=random_state)
        
        self.scores_ = mi_scores
        self._create_results_df('MI-Score')
        return self.results_df

    def _create_results_df(self, score_name):
        '''Helper to store and sort results'''
        self.results_df = pd.DataFrame({
            'Feature': self.feature_names,
            score_name: self.scores_
        }).sort_values(by=score_name, ascending=False).reset_index(drop=True)