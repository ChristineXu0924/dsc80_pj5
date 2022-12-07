import pandas as pd
import numpy as np
import os
import seaborn as sns

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import FunctionTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.base import BaseEstimator, TransformerMixin


class StdScalerByGroup(BaseEstimator, TransformerMixin):

    def __init__(self):
        pass

    def fit(self, X, y=None):
        """
        :Example:
        >>> cols = {'g': ['A', 'A', 'B', 'B'], 'c1': [1, 2, 2, 2], 'c2': [3, 1, 2, 0]}
        >>> X = pd.DataFrame(cols)
        >>> std = StdScalerByGroup().fit(X)
        >>> std.grps_ is not None
        True
        """
        # X might not be a pandas DataFrame (e.g. a np.array)
        df = pd.DataFrame(X)

        # Compute and store the means/standard-deviations for each column (e.g. 'c1' and 'c2'), 
        # for each group (e.g. 'A', 'B', 'C').  
        # (Our solution uses a dictionary)
        self.grps_ = df.groupby(df.iloc[:,0]).agg(['mean', 'std']).to_dict()

        return self

    def transform(self, X, y=None):
        """
        :Example:
        >>> cols = {'g': ['A', 'A', 'B', 'B'], 'c1': [1, 2, 3, 4], 'c2': [1, 2, 3, 4]}
        >>> X = pd.DataFrame(cols)
        >>> std = StdScalerByGroup().fit(X)
        >>> out = std.transform(X)
        >>> out.shape == (4, 2)
        True
        >>> np.isclose(out.abs(), 0.707107, atol=0.001).all().all()
        True
        """

        try:
            getattr(self, "grps_")
        except AttributeError:
            raise RuntimeError("You must fit the transformer before tranforming the data!")
        
        # Hint: Define a helper function here!

        df = pd.DataFrame(X)
        name = df.columns[0]
        out_lst = []

        def find_attr(col, attr, group):
            return self.grps_[(col, attr)][group]
        
        groups = df.groupby(name).mean().index
        for gr in groups:
            sub_df = df[df[name] == gr]
            for col in sub_df.columns[1:]:
                sub_mean = find_attr(col, 'mean', gr)
                sub_std = find_attr(col, 'std', gr)
                sub_df[col] = (sub_df[col]- sub_mean)/sub_std
            out_lst.append(sub_df)

        out = pd.concat(out_lst)

        return out.drop(columns = name)
