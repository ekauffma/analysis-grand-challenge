import runipynb
from hyperopt import hp, STATUS_OK, fmin, tpe, Trials
from hyperopt.pyll.base import scope
import numpy as np
import argparse

# TODO: add mlflow logging

def runProcessing(params):
    
    print("_____________________________________________________________________________________________")
    print("_____________________________________________________________________________________________")
    print("params = ", params)
    changed_variables = {7: params}
    
    
    notebook = runipynb.Notebook("coffea-nanoaod.ipynb", 
                                 changed_variables = changed_variables)
    loss = np.abs(notebook.ttbar_norm_bestfit-1.0)
    print("loss = ", loss)
    
    return {'status': STATUS_OK, 'loss': loss}

    
if __name__ == "__main__":
    
    
    parser = argparse.ArgumentParser(description="This script runs the cells of the AGC notebook to optimize different notebook parameters.")
    parser.add_argument("-n", "--n_trials", help="number of trials", type=int, default=5)
    args = parser.parse_args()
    
    # hyperopt trial values
    trial_params = {
        'N_FILES_MAX_PER_SAMPLE': scope.int(hp.uniform('N_FILES_MAX_PER_SAMPLE', 1, 5)),
        'NUM_BINS': scope.int(hp.uniform('NUM_BINS', 10, 50)),
        'BIN_LOW': hp.uniform('BIN_LOW', 0, 100),
        'BIN_HIGH': hp.uniform('BIN_HIGH', 450, 650),
        'PT_THRESHOLD': hp.uniform('PT_THRESHOLD', 20, 40),
        'PLOTTING': False,
        'PRINT_PROGRESS': False,
                   }
    
    trials = Trials()
    
    best_parameters = fmin(
        fn=runProcessing, 
        space=trial_params,
        algo=tpe.suggest,
        trials=trials,
        max_evals=args.n_trials # how many trials to run
    )
    
    print("Optimized Parameters = ", best_parameters)
    
    
    
    
    