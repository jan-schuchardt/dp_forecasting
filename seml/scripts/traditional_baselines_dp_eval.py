import os

import torch
from dp_timeseries.experiments.traditional_baselines_dp_eval import \
    traditional_baselines_dp_eval
from seml import Experiment

ex = Experiment()

traditional_baselines_dp_eval = ex.capture(traditional_baselines_dp_eval)


@ex.automain
def run(_config: dict) -> dict:

    db_collection = _config['db_collection']
    run_id = _config['overwrite']

    experiment_name = f'{db_collection}_{str(run_id)}'

    log_dir, metrics_val, metrics_test = traditional_baselines_dp_eval(
        experiment_name=experiment_name)

    results_dict = {
        'log_dir': log_dir,
        'metrics_val': metrics_val,
        'metrics_test': metrics_test
    }

    save_dict = {
        'config': _config,
        'results': results_dict
    }

    torch.save(save_dict, os.path.join(log_dir, 'config_and_results.pyt'))

    return results_dict
