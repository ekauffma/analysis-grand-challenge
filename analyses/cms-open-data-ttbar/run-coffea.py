import runipynb

N_FILES_MAX_PER_SAMPLE = [1,5,10]

for n in N_FILES_MAX_PER_SAMPLE:
    
    print("running notebook for N_FILES_MAX_PER_SAMPLE = ", n)
    notebook = runipynb.Notebook("coffea.ipynb", 
                                 changed_variables = {"N_FILES_MAX_PER_SAMPLE": n})
    print("______________________________________________________________________________")