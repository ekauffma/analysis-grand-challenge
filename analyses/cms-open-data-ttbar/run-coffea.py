import runipynb

from hyperopt import hp, STATUS_OK, fmin, tpe, Trials
from hyperopt.pyll.base import scope
import mlflow
import mlflow.xgboost
from mlflow.models.signature import infer_signature

import numpy as np
import argparse
import os

# TODO: add mlflow logging

def runProcessing(params):
    
    print("_____________________________________________________________________________________________")
    print("_____________________________________________________________________________________________")
    print("params = ", params)
    changed_variables = {7: params}
    
    with mlflow.start_run(experiment_id=EXP_ID, nested=True):
    
        notebook = runipynb.Notebook("coffea-nanoaod.ipynb", 
                                     changed_variables = changed_variables)
        
        mlflow.log_params(params)
        mlflow.log_params({"af": notebook.AF_NAME, 
                           "use_dask": notebook.USE_DASK, 
                           "chunksize": notebook.CHUNKSIZE,
                           "disable_processing": notebook.DISABLE_PROCESSING,
                           "io_file_percent": notebook.IO_FILE_PERCENT,
                           "pipeline": notebook.PIPELINE})
        
        loss = np.abs(notebook.ttbar_norm_bestfit-1.0)
        print("loss = ", loss)
        
        mlflow.log_metric(f'ttbar_norm_bestfit', notebook.ttbar_norm_bestfit)
        mlflow.log_metric(f'exec_time', notebook.exec_time)
        mlflow.log_metric(f'ttbar_norm_bestfit', notebook.ttbar_norm_bestfit)
        mlflow.log_metric(f'ttbar_norm_bestfit', notebook.ttbar_norm_bestfit)
        mlflow.log_metric('event_rate_per_worker_kHz', 
                          notebook.metrics['entries'] / notebook.metrics['processtime'] / 1000)
        mlflow.log_metric('data_read_MB', notebook.metrics['bytesread']*1e-6)
        mlflow.log_metric('process_time', notebook.metrics['processtime'])
        mlflow.log_metric('n_events', notebook.metrics['entries'])
    
    return {'status': STATUS_OK, 'loss': loss}

    
if __name__ == "__main__":
    
    
    parser = argparse.ArgumentParser(description="This script runs the cells of the AGC notebook to optimize different notebook parameters.")
    parser.add_argument("-n", "--n_trials", help="number of trials", type=int, default=5)
    args = parser.parse_args()
    
    # hyperopt trial values
    trial_params = {
        'N_FILES_MAX_PER_SAMPLE': scope.int(hp.uniform('N_FILES_MAX_PER_SAMPLE', 1, 10)),
        'NUM_BINS': scope.int(hp.uniform('NUM_BINS', 10, 50)),
        'BIN_LOW': hp.uniform('BIN_LOW', 0, 100),
        'BIN_HIGH': hp.uniform('BIN_HIGH', 450, 650),
        'PT_THRESHOLD': hp.uniform('PT_THRESHOLD', 20, 40),
        'PLOTTING': False,
        'PRINT_PROGRESS': False,
                   }
    
    mlflow.set_tracking_uri("https://mlflow.software-dev.ncsa.cloud")
    EXPERIMENT_ID = mlflow.set_experiment('cms-opendata-ttbar-optimization')
    
    os.environ['MLFLOW_TRACKING_URI'] = "https://mlflow.software-dev.ncsa.cloud"
    # fill credentials below
    os.environ['MLFLOW_S3_ENDPOINT_URL'] = ""
    os.environ['AWS_ACCESS_KEY_ID'] = ""
    os.environ['AWS_SECRET_ACCESS_KEY'] = ""
    
    current_experiment=dict(mlflow.get_experiment_by_name('cms-opendata-ttbar-optimization'))
    EXP_ID=current_experiment['experiment_id']
    
    
    trials = Trials()
    
    best_parameters = fmin(
        fn=runProcessing, 
        space=trial_params,
        algo=tpe.suggest,
        trials=trials,
        max_evals=args.n_trials # how many trials to run
    )
    
    print("Optimized Parameters = ", best_parameters)
    
    
    
    
    