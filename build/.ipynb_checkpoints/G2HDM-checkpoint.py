# This script is used to perform various model tasks

# declare the root directory
import sys, os
project_root = os.path.abspath(os.path.join(os.getcwd(), '.'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import packages
from src.Model2HDM.class_Model2HDM import Model2HDM
from src.Model2HDM.methods_Model2HDM import *

print("Welcome to the G2HDM program!")

# Load the model
modelname_provided = False
while modelname_provided == False:
    modelname = input("Enter the name of the model: ")
    print(f"Loading model: {modelname} ...")
    model = load_model(modelname)
    if isinstance(model, Model2HDM):
        modelname_provided = True
    else:
        print("Please enter a valid model name.")

print("Model loaded successfully.")
print("please type 'help' to see the available commands.")

# main loop
running = True
while running:
    command = input("Enter a command: ")
    
    if command == "exit":
        running = False
        
    if command == "help":
        print("Commands:")
        print("  - param_search: Perform a parameter search.")
        print("  - exit: Exit the program.")
        
    elif command == "param_search":
        print("Not implemented yet.")
        
    else:
        print("Invalid command. Try again.")


