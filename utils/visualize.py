"""Class to do the visualization of the data."""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib_venn import venn2
import os


class Visualization(object):
    def __init__(self, data_path):
        self.data_path = data_path

    def _plot_cancer(self, train, test, ds_name):
        ds = pd.concat((train, test), axis='index', ignore_index=True)
        ds = ds.iloc[:, -35:-5]

        sums = pd.DataFrame(ds.sum(axis=0))
        sums.sort_index(inplace=True)

        sns.set_style('darkgrid')
        plt.figure(figsize=(12, 10))

        sns.barplot(x=sums[0], y=sums.index)

        plt.xlabel('number of ncRNAs')
        plt.title(f'Number of ncRNAs associated to each cancer type in {ds_name} dataset')

        plt.savefig(os.path.join('images',f'{ds_name}_cancer.png'))
        plt.show()

    def plot_cancer(self, ds_name):
        """Plot the cancer data distributions"""
        train = pd.read_csv(os.path.join(self.data_path, f"{ds_name}_train.csv"))
        test = pd.read_csv(os.path.join(self.data_path, f"{ds_name}_test.csv"))
        self._plot_cancer(train, test, ds_name)

    def plot_cancer_sequence(self):
        """Plot the cancer data distributions for sequence data."""
        self.plot_cancer('sequences')

    def plot_cancer_expression(self):
        """Plot the cancer data distributions for sequence data."""
        self.plot_cancer('expressions')

    def plot_intersection(self):
        plt.figure(figsize=(10, 10))

        exp_train = pd.read_csv(os.path.join(self.data_path, 'expressions_train.csv'))
        exp_test = pd.read_csv(os.path.join(self.data_path, 'expressions_test.csv'))
        exp = exp_train.shape[0] + exp_test.shape[0]

        seq_train = pd.read_csv(os.path.join(self.data_path, 'sequences_train.csv'))
        seq_test = pd.read_csv(os.path.join(self.data_path, 'sequences_train.csv'))
        seq = seq_train.shape[0] + seq_test.shape[0]

        ees_train = pd.read_csv(os.path.join(self.data_path, 'sequences_expressions_train.csv'))
        ees_test = pd.read_csv(os.path.join(self.data_path, 'sequences_expressions_test.csv'))
        ees = ees_train.shape[0] + ees_test.shape[0]

        venn2(subsets=(exp, seq, ees), set_labels=('Expressions', 'Sequences'))
        plt.title('Intersection between expression and sequence datasets')
        plt.savefig(os.path.join('images', 'intersection.png'))
        plt.show()