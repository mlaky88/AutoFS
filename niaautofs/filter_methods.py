import numpy as np
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from sklearn import preprocessing
from pandas.api.types import is_float_dtype, is_integer_dtype, is_bool_dtype
from sklearn.metrics import pairwise_distances
from skrebate import ReliefF
import os


class FilterMethods:
    
    def __init__(self,dataset,dataset_name):
        self.dataset = dataset
        self.hash_mi = dict()
        self.hash_mrmr = dict()
        self._relevance = None
        self._redundancy = None
        self._norm_order = None

        self.dataset_name = dataset_name


    def _mi(self):
        r"""Calculate normalized mutual information between features and target, and between features."""
        features_df = self.dataset.transactions.drop(columns=["class"])
        class_values = self.dataset.transactions["class"]
        self._relevance = mutual_info_classif(features_df, class_values, random_state=42) 
        self._relevance /= np.sqrt(np.sum(self._relevance**2))

        feature_names = self.dataset.features[1:]
        column_names = features_df.columns
        self._redundancy = np.zeros((len(feature_names), len(feature_names)))

        for i, col_i_name in enumerate(column_names):
            for j, col_j_name in enumerate(column_names):
                if i != j and j > i:
                    if feature_names[j].dtype == "cat" or feature_names[j].dtype == "int":
                        self._redundancy[i, j] = mutual_info_classif(features_df[col_i_name].values.reshape(-1,1), features_df[col_j_name].values,discrete_features="auto",random_state=42)[0]
                    elif feature_names[j].dtype == "float":                        
                        self._redundancy[i, j] = mutual_info_regression(features_df[col_i_name].values.reshape(-1,1), features_df[col_j_name].values,discrete_features="auto",random_state=42)[0]                    

        self._redundancy /= np.sqrt(np.sum(self._redundancy**2))

    def mi(self,x,beta=0.5):
        r"""
        Calculate mutual information between features and target
        x is a vector of presence/absence of features. This vector is always raw
        """

        if self._relevance is None and self._redundancy is None:
            self._mi()

        # Check if already in hash
        # hash_mi["[1,0,0,1,0]"] = {'relevance': 0.5, 'redundancy': 0.2}
        if str(x) in self.hash_mi:
            print("[MI]Found {} in hash".format(x))
            mi_rel_red = self.hash_mi[str(x)]
            return mi_rel_red['relevance'] - beta * mi_rel_red['redundancy']
        
        # If not, the calculate it
        else:
            print("[MI]Not found {} in hash. Calculating ...".format(x))
            sum_relevance = sum([self._relevance[i] for i in range(len(x)) if x[i] >= 0.5])
            sum_redundancy = 0
            for i in range(len(x)):
                for j in range(len(x)):                    
                    if i != j and j > i and x[i] >= 0.5 and x[j] >= 0.5:
                        sum_redundancy += self._redundancy[i, j]
            
            # Store in hash
            self.hash_mi[str(x)] = {'relevance': sum_relevance, 'redundancy': sum_redundancy}
            return sum_relevance - beta * sum_redundancy

    def _mrmr(self):
        r"""Calculate the max-relevance min-redundancy."""
        
        if os.path.exists("intermediate_files/{}_mrmr.npy".format(self.dataset_name)):
            print("Found ReliefF in file. Loading file ...")
            self._norm_order = np.load("intermediate_files/{}_mrmr.npy".format(self.dataset_name))
            print(self._norm_order)
        else:
            N = len(self.dataset.features) - 1
            relieff = ReliefF(n_features_to_select=N,n_jobs=-1,n_neighbors=10,verbose=True)
            relieff.fit(self.dataset.transactions.drop(columns=["class"]).values, self.dataset.transactions["class"].values)
            x = relieff.feature_importances_
            sorted_indexes = sorted(range(len(x)), key=lambda i: x[i])
            self._norm_order = [(x + 1)/N for x in sorted_indexes]
            print(self._norm_order)
            print("Saving ReliefF scores to file ...")
            np.save("intermediate_files/{}_mrmr.npy".format(self.dataset_name), self._norm_order)         

        
    def mrmr(self,x,alpha=0.5, beta=0.5):
        r"""Calculate the max-relevance min-redundancy."""

        if self._relevance is None:
            self._mi()
        if self._norm_order is None:
            self._mrmr()

        ranks = 0
        mi_rel = 0
        if str(x) in self.hash_mrmr:
            print("Found {} in hash".format(x))
            ranks = self.hash_mrmr[str(x)]
        else:
            ranks =  np.sum([self._norm_order[i] for i in range(len(x)) if x[i] >= 0.5])
            self.hash_mrmr[str(x)] = ranks

        if str(x) in self.hash_mi:
            mi_rel_red = self.hash_mi[str(x)]
            mi_rel = mi_rel_red['relevance']
        else:
            self.mi(x) #doda v hash
            mi_rel = self.hash_mi[str(x)]['relevance']

        return mi_rel - beta * alpha * ranks    

    def _imrmr(self,x):
        r"""Calculate the improved max-relevance min-redundancy."""
        pass
    def imrmr(self,x):
        r"""Calculate the improved max-relevance min-redundancy."""
        pass

    def _jmi(self,x):
        r"""Calculate the improved max-relevance min-redundancy."""
        pass
