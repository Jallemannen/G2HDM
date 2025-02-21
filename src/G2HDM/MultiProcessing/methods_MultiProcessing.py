#from numba import jit, vectorize # type: ignore
import numpy as np
import sympy as sp

from multiprocessing import Process, Value, Manager
import time
import threading
import os

# Custom Packages, Utils
#from .methods_MultiProcessing import * 
from ..utils.methods_math import *
from ..utils.methods_data import *
from ..utils.methods_general import *

################# Multiprocessing #################

# progress printer (multiline)
def print_progress(max_runtime=None, max_iterations=None):
    t0 = time.time()

    while not DONE.value:
        total_time = time.time()-t0
        if progress_int.value == 0:
            sec_per_iteration = 0
        else:
            sec_per_iteration = total_time/progress_int.value
        progress_str = f"Iterations: {progress_int.value} | Runtime: {total_time//60**2:.0f}h:{(total_time%60**2)//60:.0f}m:{(total_time%60**2)%60:.0f}s | Sec/iteration: {sec_per_iteration:.2f} | Iterations/hour: {progress_int.value/(time.time()-t0)*60**2:.0f} |         "
        if not DONE.value:
            print(progress_str, end="\r")
        
        # Stop conditions
        if max_runtime != None:
            if total_time > max_runtime:
                print("\nMax runtime reached. Stopping...")
                with DONE.get_lock():
                    DONE.value = True
        if max_iterations != None:
            if progress_int.value > max_iterations:
                print("\nMax iterations reached. Stopping...")
                with DONE.get_lock():
                    DONE.value = True
        time.sleep(1)

# Stop processes from multiprocessing (manual stop)
def stop_processes():
    input()
    with DONE.get_lock():
        DONE.value = True
    print("\nStopping...")

# Multiprocessing (might make this a class)
def multiprocessing(func, kwargs, processes=1, max_runtime=None, max_iterations=None):
    """Starts work function among multiple processes."""
    from multiprocessing import Process, Pool, Value, Manager
    import threading
    import time
    import queue

    #print("Number of logical threads on this computer:", os.cpu_count())
    
    if os.cpu_count() < processes:
        print("Warning: Number of processes exceeds number of logical cores on this computer.")
        
    
    # Global flag to stop all processes (shared across processes)
    global DONE
    DONE = Value('b', False)
    
    # Global queue to store results
    global result_queue 
    result_queue = queue.Queue() 
    shared_list = Manager().list()
    
    #global progress_int
    global progress_int
    progress_int = Value('i', 0) 
    # add an optional work distributer?
    
    
    #with Pool(processes) as pool:
    #    results = pool.imap_unordered(func, args)
    
    # Start worker threads
    processes_list = []
    for i in range(processes):
        try:
            process = Process(target=worker, args=(i+1,shared_list, func, kwargs))  #, args=args
            process.start()
            processes_list.append(process)
        except Exception as e:
            #print(e)
            if i == 0:
                print(e)
            else:
                print(f"processes {i+1} failed to start. Continuing with {i} processes instead.")
            break
    
    print("Processes:", processes)
    if max_runtime != None:
        print(f"Max runtime: {max_runtime//60**2:.0f}h:{(max_runtime%60**2)//60:.0f}m:{(max_runtime%60**2)%60:.0f}s")
    if max_iterations != None:
        print("Max iterations:", max_iterations)
    
    print("Press enter to stop the worker processes. \n")       
    
    # Extra thread for displaying progress
    thread_progress = threading.Thread(target=print_progress, args=(max_runtime, max_iterations))
    thread_progress.daemon = True 
    thread_progress.start()
    
    # Extra thread for stopping processes (manual stop)  
    thread_stopping = threading.Thread(target=stop_processes, daemon = True)
    #thread_stopping = Process(target=stop_processes, daemon = True)
    thread_stopping.daemon = True 
    thread_stopping.start()
    
    thread_progress.join()
    #thread_stopping.terminate()
    #input()
    #with DONE.get_lock():
    ##    DONE.value = True
        

    # Stop and wait for all threads to finish 
    for process in processes_list:
        process.join()
        
    
    
        #thread_stopping.terminate()
        #thread_progress.terminate()
    
    # Collect results
    print("All processes finished. Collecting results...")
    
    
    # Sort results for orderly output
    results = []
    # Sort by thread ID
    #shared_list.sort(key=lambda x: (x[0]))  
    for result_process in shared_list:
        for data_point in result_process[1]:
            results.append(data_point)
    
    #results.sort(key=lambda x: (x[0], x[1]))  # Sort by thread ID, then task number
    #print("Results:", results)
    return results

# Worker wrapper function
def worker(thread_ID, shared_list, func, *args):
    """thread worker function"""

    results = []
    while not DONE.value:
        result = func(*args)
        if result != None:
            results.append(result)
            with progress_int.get_lock():
                progress_int.value += 1
    
    # Sum the total finished threads later instead
    print(f"Thread {thread_ID} finished.") #, end="\r"
    ## Return results
    return_results = [thread_ID, results]
    shared_list.append(return_results)
    #result_queue.put(return_results)

################# Div methods #################
