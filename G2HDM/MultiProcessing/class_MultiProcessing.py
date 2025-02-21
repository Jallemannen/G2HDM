from multiprocessing import Process, Value, Manager
import time
import threading
import os
import time 

# Custom Packages, Utils
from .methods_MultiProcessing import * 
from ..utils.methods_math import *
from ..utils.methods_data import *
from ..utils.methods_general import *


class DataProcessing:
    def __init__(self, model, paramsearch_func, paramsearch_func_args):
        self.model = model
        self.paramsearch_func = paramsearch_func
        self.paramsearch_func_args = paramsearch_func_args
        self.save_path = model.path_data

    def run(self, processes=1, max_runtime=None, max_iterations=None):
        """Run parameter search using multiprocessing."""
        manager = MultiprocessingManager(
            worker_func=self.paramsearch_func,
            max_runtime=max_runtime,
            max_iterations=max_iterations,
            processes=processes,
        )
        
        return manager.run(**self.paramsearch_func_args)


class MultiprocessingManager:
    def __init__(self, worker_func, max_runtime=None, max_iterations=None, processes=1):
        self.worker_func = worker_func
        self.max_runtime = max_runtime
        self.max_iterations = max_iterations
        self.processes = processes

        self.results = []
        self.done = Value('b', False)
        self.progress = Value('i', 0)

    def _worker(self, thread_id, shared_list, *args):
        """Worker function for parallel tasks."""
        results = []
        while not self.done.value:
            result = self.worker_func(*args)
            if result is not None:
                results.append(result)
                with self.progress.get_lock():
                    self.progress.value += 1

        # Append thread results to shared list
        shared_list.append((thread_id, results))

    def _print_progress(self, start_time):
        while not self.done.value:
            elapsed = time.time() - start_time
            progress = self.progress.value
            rate = progress / elapsed if elapsed > 0 else 0
            print(
                f"Progress: {progress} | Elapsed: {elapsed:.2f}s | Rate: {rate:.2f} iters/s",
                end="\r"
            )

            # Check stop conditions
            if self.max_runtime and elapsed > self.max_runtime:
                print("\nMax runtime reached.")
                self.done.value = True
            if self.max_iterations and progress >= self.max_iterations:
                print("\nMax iterations reached.")
                self.done.value = True

            time.sleep(1)

    def run(self, *args):
        """Start multiprocessing tasks."""
        shared_list = Manager().list()
        processes = []

        # Start worker processes
        for i in range(self.processes):
            p = Process(target=self._worker, args=(i + 1, shared_list, *args))
            p.start()
            processes.append(p)

        # Start progress monitoring
        start_time = time.time()
        progress_thread = threading.Thread(target=self._print_progress, args=(start_time,))
        progress_thread.start()

        # Wait for processes to complete
        for p in processes:
            p.join()

        self.done.value = True
        progress_thread.join()

        # Collect and merge results
        for thread_id, result_list in shared_list:
            self.results.extend(result_list)

        return self.results


