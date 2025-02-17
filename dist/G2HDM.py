# This script is used to perform various model tasks

# Make the code safe to run, ie add trys and excepts

############### Opening program ################
print("G2HDM: Loading packages...")

# declare the root directory
import sys, os
project_root = os.path.abspath(os.path.join(os.getcwd()))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
    
path_saved_data = os.path.join(project_root, "Models", "Saved models" )


print("Project root directory:", project_root)
print("Saved data directory:", path_saved_data)

# Import custom packages
print("Importing packages...")
from src.Model2HDM.class_Model2HDM import Model2HDM
from src.Model2HDM.methods_Model2HDM import *
from src.MultiProcessing.methods_MultiProcessing import *


print("=====================================")
print("Welcome to the G2HDM program!\n")

print("list of models to choose from:")

if not os.path.exists(path_saved_data):
    print(f"Error: Folder '{path_saved_data}' does not exist!")

folders = next(os.walk(path_saved_data))[1]
for n, folder in enumerate(folders):
    print(f" {n}. {folder}")
print("\n")   

# Load the model
modelname_provided = False
while modelname_provided == False:
    modelname = input("Enter the name of the model: ")
    
    try:
        modelname = int(modelname)
        modelname = folders[modelname]
    except:
        pass
        
    if modelname == "exit":
        print("Exiting program...")
        sys.exit()
        
    # Check if the model exists
    if modelname not in folders:
        print("Model not found. Please enter a valid model name.")
        continue
    
    print(f"Loading model: {modelname} ...")
    model = load_model(modelname, project_root = os.path.abspath(os.path.join(os.getcwd(), '.')))
    if isinstance(model, Model2HDM):
        modelname_provided = True
    else:
        print("Please enter a valid model name.")

print("Model loaded successfully.\n")
print("please type 'help' to see the available commands.\n")

##############################################################


def ps():
    filename = input("Enter the filename: ")
    if filename == "":
        filename = "unnamed"
    # check if file exist and if merge
    
    N_processes = int(input("Enter the number of processes: "))
    if N_processes == "":
        N_processes = None
    try:
        N_processes = int(N_processes)
    except:
        N_processes = None
    
    runtime = input("Enter the runtime: ")
    if runtime == "":
        runtime = None
    try:
        runtime = int(runtime)
    except:
        runtime = None
    
    iterations = input("Enter the number of iterations: ")
    if iterations == "":
        iterations = None
    try:
        iterations = int(iterations)
    except:
        iterations = None
        
    model.param_search(N_processes=N_processes, runtime=runtime, iterations=iterations, filename=filename, merge=True)
    








##############################################################

# Main loop
def main():
    command = input("Enter a command: ")
    
    if command == "exit":
        print("Exiting program...")
        running = False
        sys.exit()
        
    elif command == "help":
        print("\nCommands:")
        print("  - param search / ps: Perform a parameter search.")
        print("  - exit: Exit the program.")
        
    elif command in ["param search", "ps"]:
        ps()
        
    elif command in ["level0", "l0"]:
        print("Level 0")
        
    elif command in ["plot", "p"]:
        command2 = input("Select what plots to create: ")
        if command2 in ["level0", "l0"]:
            print("Level 0 plots")
        elif command2 in ["level1", "l1"]:
            print("Level 1 plots")
        elif command2 in ["parameter search", "ps"]:
            print("Parameter search plots")
        
    else:
        print("Invalid command. Try again.")
    


# Run the main loop
running = True
while running:
    main()


