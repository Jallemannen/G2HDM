# Methods for handling data, such as saving and loading data

import sys
import os
import importlib
import inspect

#################### Importing and reloading packages ####################

def reload_module(module_name):
    """
    Reload the specified module and update the global namespace with all public
    attributes from that module.
    """
    if module_name in sys.modules:
        module = sys.modules[module_name]
        module = importlib.reload(module)
    else:
        module = importlib.import_module(module_name)

def reload_and_import(module_full_name, symbol_name):
    """
    Reloads the module given by module_full_name and imports the symbol 
    into the caller's global namespace.

    Parameters:
      module_full_name (str): e.g. "Code.Model2HDM.methods_Model2HDM"
      symbol_name (str): e.g. "function_name"
      
    Returns:
      The symbol (function, class, etc.) from the reloaded module.
    """
    # Get the caller's globals
    caller_globals = inspect.stack()[1].frame.f_globals

    # Reload or import the module
    if module_full_name in sys.modules:
        module = importlib.reload(sys.modules[module_full_name])
    else:
        module = importlib.import_module(module_full_name)

    # Retrieve the symbol from the module
    symbol = getattr(module, symbol_name)

    # Update the caller's globals
    caller_globals[symbol_name] = symbol

    return symbol

#################### General Methods ####################

# Delete file or folder
def delete(path:str):
    """Deletes a folder or file.

    Args:
        path (str): Path to folder/file.
    """
    import os
    import shutil
    
    # Check if the path is in the 'Saved' directory
    if path.startswith("Saved") == False:
        #raise Exception("Only files in the 'Saved' directory can be deleted.")
        ans = input("You are currently trying to delete a file/folder outside the 'Saved' directory. Are you sure you want to continue? (y/n)")
        if ans != "y":
            print("Deletion aborted.")
            return None
        
    # Directory
    if os.path.isdir(path):
        # If the path is a directory, use rmtree to remove it
        try:
            shutil.rmtree(path)
            print(f"Directory '{path}' has been removed.")
        except Exception as e:
            print(f"Error deleting directory '{path}': {e}")
    # File
    elif os.path.isfile(path):
        # If the path is a file, use remove to delete it
        try:
            os.remove(path)
            print(f"File '{path}' has been removed.")
        except Exception as e:
            print(f"Error deleting file '{path}': {e}")
    else:
        print(f"The path '{path}' does not exist or is not a file or directory.")

#################### Pickle ####################

# Pickle Save func
def save_pkl(data, path:str):

    # Check if path is a string
    assert isinstance(path, str), "Path must be a string" 
        
    #can use match - case for later python versions
    if path.endswith(".pkl"):
        import pickle as pkl
        #import dill as pkl
        with open(path, "wb") as file:
            pkl.dump(data, file)
    
    else:
        raise ValueError("Invalid file format. Use .pkl")

# Pickle Load func
def load_pkl(path:str):
    
    # Check if path is a string
    assert isinstance(path, str), "Path must be a string"  
    
    # load from .pkl
    if not path.endswith(".pkl"):
        raise ValueError("Invalid file format. Use .pkl or .txt")
        
    # Load data
    try:
        #load_func()
        import pickle as pkl 
        with open(path, "rb") as file:
            data = pkl.load(file)
            return data    
        
    except FileNotFoundError as e:
        print("File not found\n", e)
        return None
    except ModuleNotFoundError:
        raise "A Module was not found, try and remake the savefiles"
    except Exception as e:  # Catch all other exceptions
        raise RuntimeError(f"An unexpected error occurred: {e}")

#################### Csv / Txt ####################

# General data saving function
def save_data(data, filename, path, merge = False, show_size = False):
    """
    Save data to a CSV file. The Data should be a dictionary with the keys as the column names and the values as the data.
    """
    
    import pandas as pd
    import os
    
    filename = filename if filename.endswith(".csv") else filename + ".csv"
    file_path = os.path.join(path, filename)
    
    if not os.path.exists(path):
        raise Exception("Data folder does not exist. Create it first.")
    
    # Convert data to dataframe
    dataframe = pd.DataFrame(data)
    
    # If merge is True, read the existing file and merge it with the new data
    if merge and os.path.exists(file_path):
        existing_data = pd.read_csv(file_path)
        
        # Check if any new columns/headers are present in the new data
        new_columns = set(dataframe.columns) - set(existing_data.columns)
        if new_columns:
            raise Exception(f"New columns present in data: {new_columns}. Aborting merge.")
        
        # Merge data based on common columns (you can adjust merge logic here if necessary)
        dataframe = pd.concat([existing_data, dataframe], ignore_index=True)
    
    # Save data to file
    dataframe.to_csv(file_path, index=False)
    
    # Print 
    if show_size:
        file_size = os.path.getsize(file_path)
        file_size_string = file_size
        
        if file_size / 1024**2 >= 1:
            file_size_string = f"{(file_size / 1024**2):.0f} MB"
        elif file_size / 1024 >= 1:
            file_size_string = f"{(file_size / 1024):.0f} KB"
        else:
            file_size_string = f"{file_size:.0f} Byte"
        
        print(f"Data saved to: {file_path} \nFile size on disk: {file_size_string}")

# General data loading function
def load_data(filename, path, as_dict=True) -> dict:
    """
    Load data from a CSV file and return it as a dictionary.
    """
    import pandas as pd
    import os
    
    filename = filename if filename.endswith(".csv") else filename + ".csv"
    file_path = os.path.join(path, filename)  
        
    if not os.path.exists(file_path):
        raise Exception(f"File {file_path} does not exist.")
    
    dataframe = pd.read_csv(file_path)
    try:
        dataframe = dataframe.apply(pd.to_numeric)
    except:
        raise Exception("Data could not be converted to numeric.")
    
    if as_dict:
        return dataframe.to_dict(orient="list")
    else:
        return dataframe