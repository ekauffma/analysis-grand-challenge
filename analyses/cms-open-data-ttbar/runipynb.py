import json
import hist
from collections import ChainMap

class Notebook:
    def __init__(self, file, changed_variables={}):
        self._globals = {}
        self._locals = {}
        self.changed_variables = changed_variables
        if isinstance(file, str):
            with open(file, "r") as fileobj:
                self._exec(fileobj)
        else:
            self._exec(file)

    def __getattr__(self, name):
        if name in self._locals:
            return self._locals[name]
        elif name in self._globals:
            return self._globals[name]
        else:
            raise AttributeError(f"name {name!r} not found in notebook environment")

    def _exec(self, file):
        self.json = json.load(file)

        x = self.json.get("metadata")
        if x is not None:
            x = x.get("kernelspec")
            if x is not None:
                x = x.get("language")
        if x != "python":
            raise TypeError("notebook kernel is not specified or is not 'python'")
            

        for cell_number, cell in enumerate(self.json["cells"]):
                        
            if cell_number in self.changed_variables.keys():
                
                print("Cell Number = ", cell_number)
                variables_to_change = list(set(self._locals) & set(self.changed_variables[cell_number]))
                print("Variables to Change = ", variables_to_change)
                
                for var_name in variables_to_change:
                    self._locals[var_name] = self.changed_variables[cell_number][var_name]
            
            if cell["cell_type"] == "code":
                source = "".join(cell["source"])
                
                try:
                    # run cell
                    exec(source, self._globals, self._locals)
                    
                except Exception as err:
                    
                    print("Exception")
                
                    x = cell.get("metadata")
                    if x is not None:
                        x = x.get("tags")
                        if x is not None:
                            if "raises-exception" in x:
                                continue
                                
                    note = f"in cell number {cell_number} (where the first cell is 0)"
                    if hasattr(err, "add_note"):
                        err.add_note(note)
                        raise err from None
                    else:
                        raise err from None
                    #     raise type(err)(str(err) + "\n\n" + note) from err
