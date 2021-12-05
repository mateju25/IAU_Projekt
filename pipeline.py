from datetime import datetime
import pandas as pd
from sklearn.base import TransformerMixin
from sklearn.feature_selection import f_classif, SelectKBest, chi2
from sklearn.impute import KNNImputer
from sklearn.preprocessing import OrdinalEncoder, MinMaxScaler, PowerTransformer, QuantileTransformer


def load_datasets(filename1, filename2):
    df_profiles = pd.read_csv(filename1, sep='\t', parse_dates=['birthdate'])
    df_profiles.drop('Unnamed: 0', axis=1, inplace=True)
    df_labor = pd.read_csv(filename2, sep='\t')
    df_labor.drop('Unnamed: 0', axis=1, inplace=True)
    return pd.merge(df_profiles, df_labor, on='name', how='outer')


def fix_dataset(dataset):
    dataset.replace('no', 'N', inplace=True)
    dataset.replace('yes', 'Y', inplace=True)
    dataset.replace('divoced', 'divorced', inplace=True)
    dataset.replace('black', 'Black', inplace=True)
    dataset.replace('blsck', 'Black', inplace=True)
    dataset.replace('white', 'White', inplace=True)
    dataset["age"] = datetime.now().year - pd.DatetimeIndex(dataset["birthdate"]).year
    dataset.drop(dataset[dataset.weight < 0].index, inplace=True)
    dataset = dataset.drop(columns=['residence', 'birthdate', 'ssn_x', 'current_location', 'name', 'ssn_y', 'job'])
    dataset.drop_duplicates()
    return dataset


feature_cols = ['sex', 'blood_group', 'race', 'relationship', 'smoker', 'erytrocyty', 'trombocyty', 'weight', 'hbver',
                'er-cv', 'hematokrit', 'leukocyty', 'alp', 'hemoglobin', 'ast', 'alt', 'etytr', 'age']
output_cols = ['indicator']
categorical_cols = ['sex', 'blood_group', 'race', 'relationship', 'smoker']
numerical_cols = ['erytrocyty', 'trombocyty', 'weight', 'hbver', 'er-cv', 'hematokrit', 'leukocyty', 'alp',
                  'hemoglobin', 'ast', 'alt', 'etytr', 'age']


class CustomEncoderTransformer(TransformerMixin):
    def __init__(self, column_names=[]):
        self.column_names = column_names

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        global categorical_cols
        X = pd.DataFrame(X, columns=self.column_names)
        enc = OrdinalEncoder()
        X[categorical_cols] = enc.fit_transform(X[categorical_cols])
        return X


class CustomOutlierTransformer(TransformerMixin):
    def __init__(self, column_names=[], strategy="drop"):
        self.column_names = column_names
        self.strategy = strategy

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = pd.DataFrame(X, columns=self.column_names)
        for column in X.columns:
            df_merged_out = self.identify_outliers(X[column])
            if (self.strategy == "drop"):
                X = X.drop(df_merged_out.index)
            elif (self.strategy == "mean"):
                X[column].fillna(X[column].mean(), inplace=True)

        return X

    def identify_outliers(self, x):
        iqr = x.quantile(0.75) - x.quantile(0.25)
        lower_ = x.quantile(0.25) - 1.5 * iqr
        upper_ = x.quantile(0.75) + 1.5 * iqr

        return x[(x > upper_) | (x < lower_)]


class CustomMinMaxTransformer(TransformerMixin):
    def __init__(self, column_names=[], strategy="scaler"):
        self.column_names = column_names
        self.strategy = strategy

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        global numerical_cols
        X = pd.DataFrame(X, columns=self.column_names)
        if (self.strategy == "scaler"):
            scaler = MinMaxScaler()
            X[numerical_cols] = scaler.fit_transform(X[numerical_cols])
        elif (self.strategy == "transformer"):
            transformer =  PowerTransformer(method='yeo-johnson')
            X[numerical_cols] = transformer.fit_transform(X[numerical_cols])
        return X


class CustomNullValuesTransformer(TransformerMixin):
    def __init__(self, column_names=[], strategy="knn"):
        self.column_names = column_names
        self.strategy = strategy

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = pd.DataFrame(X, columns=self.column_names)
        if self.strategy == "knn":
            imputer = KNNImputer(n_neighbors=5)
            X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)
        elif self.strategy == "drop":
            X.dropna(inplace=True)

        return X


class CustomAtributeSelectiomTransformer(TransformerMixin):
    def __init__(self, column_names=[], k=1):
        self.column_names = column_names
        self.k = k

    def fit(self, X, y=None):
        return self

    def transform(self, x, y):
        fs = SelectKBest(score_func=f_classif, k=self.k)
        fs.fit(x, y)
        cols = fs.get_support(indices=True)
        x = x.iloc[:, cols]

        return x, y

class InputAndOutputTransformer(TransformerMixin):
    def __init__(self, column_names=[]):
        self.column_names = column_names

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = pd.DataFrame(X, columns=self.column_names)
        X, y = self.split_dataset(X)
        return X, y

    def split_dataset(self, dataset):
        global feature_cols
        global output_cols

        X = dataset[feature_cols]
        y = dataset[output_cols]
        return X, y
