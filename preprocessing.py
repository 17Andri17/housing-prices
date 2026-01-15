import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler


# ----------------------------
# BASEMENT FEATURES
# ----------------------------
BASEMENT_CATEGORICAL = ['BsmtQual', 'BsmtCond',
                        'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2']
BASEMENT_CATEGORICAL_FILL = 'NoBasement'
BASEMENT_NUMERICAL = ['BsmtFinSF1', 'BsmtFinSF2',
                      'BsmtUnfSF', 'TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath']

# ----------------------------
# GARAGE FEATURES
# ----------------------------
GARAGE_CATEGORICAL = ['GarageType', 'GarageFinish', 'GarageQual', 'GarageCond']
GARAGE_CATEGORICAL_FILL = 'NoGarage'
GARAGE_NUMERICAL = ['GarageCars', 'GarageArea'] # Will be filled with 0

# --------------------------------------------------------
# OTHER CATEGORICAL FEATURES AND WHAT THEY ARE FILLED WITH
# --------------------------------------------------------
CATEGORICAL_MISSING_FILL = {
    'MasVnrType': 'None',
    'FireplaceQu': 'NoFireplace',
    'PoolQC': 'NoPool',
    'Fence': 'NoFence',
    'MiscFeature': 'None',
    'Alley': 'NoAlley',
}

# --------------------------------------------------------
# OTHER NUMERICAL FEATURES AND WHAT THEY ARE FILLED WITH
# --------------------------------------------------------
NUMERICAL_MISSING_FILL_ZERO = ['MasVnrArea', 'GarageYrBlt']

class AmesImputer(BaseEstimator, TransformerMixin):
    def __init__(self):

        self.bsmt_cat = BASEMENT_CATEGORICAL
        self.bsmt_cat_fill = BASEMENT_CATEGORICAL_FILL

        self.bsmt_num = BASEMENT_NUMERICAL

        self.garage_cat = GARAGE_CATEGORICAL
        self.garage_cat_fill = GARAGE_CATEGORICAL_FILL

        self.garage_num = GARAGE_NUMERICAL

        self.const_cat_fill = CATEGORICAL_MISSING_FILL
        self.zero_num_fill = NUMERICAL_MISSING_FILL_ZERO

    def fit(self, X, y=None):
        '''
        Fitting the Imputer for `LotFrontage` per `Neighborhood` and `Electrical` modes from training data
        '''
        X = X.copy()

        # Neighborhood -> LotFrontage median and global median from train data only
        self.lf_by_nbhd_ = X.groupby('Neighborhood')['LotFrontage'].median()
        self.lf_global_ = X['LotFrontage'].median()

        mode = X['Electrical'].mode(dropna=True)
        self.electrical_mode_ = mode.iloc[0] if len(mode) else 'SBrkr'

        return self

    def transform(self, X):
        X = X.copy()

        # basement
        for c in self.bsmt_cat:
            X[c] = X[c].fillna(self.bsmt_cat_fill)
        for c in self.bsmt_num:
            X[c] = X[c].fillna(0)

        # garage
        for c in self.garage_cat:
            X[c] = X[c].fillna(self.garage_cat_fill)
        for c in self.garage_num:
            X[c] = X[c].fillna(0)

        # Electrical fill
        X['Electrical'] = X['Electrical'].fillna(self.electrical_mode_)

        # LotFrontage: fill by Neighborhood median, then global median
        if 'LotFrontage' in X.columns:
            if self.lf_by_nbhd_ is not None:
                mapped = X['Neighborhood'].map(self.lf_by_nbhd_)
                X['LotFrontage'] = X['LotFrontage'].fillna(mapped)
            if self.lf_global_ is not None:
                X['LotFrontage'] = X['LotFrontage'].fillna(self.lf_global_)
            else:
                X['LotFrontage'] = X['LotFrontage'].fillna(
                    X['LotFrontage'].median())

        # other constant categorical fills
        for c, fillv in self.const_cat_fill.items():
            X[c] = X[c].fillna(fillv)
        # other numerical fills with 0
        for c in self.zero_num_fill:
            X[c] = X[c].fillna(0)

        

        return X



def make_preprocessor(df: pd.DataFrame) -> Pipeline:
    # Treat these as categorical even if stored as integers
    treat_as_nominal = [c for c in ['MSSubClass', 'MoSold'] if c in df.columns]

    # Ordinal definitions (lowest -> highest)
    ordinal_map = {
        # qual/cond style
        'ExterQual':   ['Po', 'Fa', 'TA', 'Gd', 'Ex'],
        'ExterCond':   ['Po', 'Fa', 'TA', 'Gd', 'Ex'],
        'HeatingQC':   ['Po', 'Fa', 'TA', 'Gd', 'Ex'],
        'KitchenQual': ['Po', 'Fa', 'TA', 'Gd', 'Ex'],

        # basement
        'BsmtQual':     ['NoBasement', 'Po', 'Fa', 'TA', 'Gd', 'Ex'],
        'BsmtCond':     ['NoBasement', 'Po', 'Fa', 'TA', 'Gd', 'Ex'],
        'BsmtExposure': ['NoBasement', 'No', 'Mn', 'Av', 'Gd'],
        'BsmtFinType1': ['NoBasement', 'Unf', 'LwQ', 'Rec', 'BLQ', 'ALQ', 'GLQ'],
        'BsmtFinType2': ['NoBasement', 'Unf', 'LwQ', 'Rec', 'BLQ', 'ALQ', 'GLQ'],

        # functionality
        'Functional': ['Sal', 'Sev', 'Maj2', 'Maj1', 'Mod', 'Min2', 'Min1', 'Typ'],

        # garage
        'GarageFinish': ['NoGarage', 'Unf', 'RFn', 'Fin'],
        'GarageQual':   ['NoGarage', 'Po', 'Fa', 'TA', 'Gd', 'Ex'],
        'GarageCond':   ['NoGarage', 'Po', 'Fa', 'TA', 'Gd', 'Ex'],

        # other ordered
        'LotShape':   ['IR3', 'IR2', 'IR1', 'Reg'],
        'LandSlope':  ['Sev', 'Mod', 'Gtl'],
        'PavedDrive': ['N', 'P', 'Y'],

        # “absence” quality fields
        'FireplaceQu': ['NoFireplace', 'Po', 'Fa', 'TA', 'Gd', 'Ex'],
        'PoolQC':      ['NoPool', 'Fa', 'TA', 'Gd', 'Ex'],

        # ????? Maybe maybe not
        'Fence':       ['NoFence', 'MnWw', 'GdWo', 'MnPrv', 'GdPrv'],
    }

    ordinal_cols = [c for c in ordinal_map.keys() if c in df.columns]

    # Nominal categoricals = object/category cols + treat_as_nominal, excluding ordinal
    cat_cols = df.select_dtypes(
        include=['object', 'category']).columns.tolist()
    nominal_cols = sorted(set(cat_cols + treat_as_nominal) - set(ordinal_cols))

    # Numeric cols = numeric dtypes excluding treat_as_nominal (kept as categorical)
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    num_cols = [c for c in num_cols if c not in treat_as_nominal]

    # Pipelines
    ordinal_encoder = OrdinalEncoder(
        categories=[ordinal_map[c] for c in ordinal_cols],
        handle_unknown='use_encoded_value',
        unknown_value=-1
    )

    nominal_pipe = Pipeline(steps=[
        # after AmesImputer most NaNs are handled; keep this just to be sure pls
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False)),
    ])

    ordinal_pipe = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('ordinal', ordinal_encoder),
    ])

    numeric_pipe = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
    ])

    encoder = ColumnTransformer(
        transformers=[
            ('num', numeric_pipe, num_cols),
            ('ord', ordinal_pipe, ordinal_cols),
            ('nom', nominal_pipe, nominal_cols),
        ],
        remainder='drop',
        verbose_feature_names_out=False,
    )

    return Pipeline(steps=[
        ('impute_domain', AmesImputer()),
        ('encode', encoder),
    ])



