import os

from experiment.nig import experiments

__author__ = 'eaplatanios'

metrics = ['accuracy', 'f1 score', 'auc']
results = experiments.load_results(
    filename=os.path.join(os.getcwd(), 'working', 'results.pk'),
    yaml_format=False)
experiments.log_results_summary(results, metrics=metrics)
# experiments.plot_results(results, metrics=metrics)
