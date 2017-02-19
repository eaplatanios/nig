import os

from experiment.nig import experiments

__author__ = 'eaplatanios'

metrics = ['accuracy', 'f1 score', 'auc']
results = experiments.load_results(
    filename=os.path.join(os.getcwd(), 'working', 'results.pk'))
experiments.plot_results(results, metrics=metrics)
