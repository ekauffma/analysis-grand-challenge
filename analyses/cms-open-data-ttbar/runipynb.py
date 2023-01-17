import json



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
            if cell["cell_type"] == "code":
                source = "".join(cell["source"])
                # print("Cell Number = ", cell_number)
                try:
                    # change values if needed
                    for var_name in list(set(self._locals) & set(self.changed_variables)):
                        self._locals[var_name] = self.changed_variables[var_name]
                        
                    # run cell
                    exec(source, self._globals, self._locals)
                except Exception as err:
                    
                    print("Exception")
                    print("self._locals = ", self._locals)
                    print("self._globals = ", self._globals)
                    
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
