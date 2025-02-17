import os
import platform
import sys
import subprocess

def build_executable(script_name="main.py"):
    system = platform.system()

    # Ensure PyInstaller is installed
    try:
        import PyInstaller
    except ImportError:
        print("PyInstaller not found. Installing now...")
        subprocess.run([sys.executable, "-m", "pip", "install", "pyinstaller"])

    print(f"Detected OS: {system}")
    
    # Check if script exists
    script_path = os.path.abspath(script_name)
    if not os.path.exists(script_path):
        print(f"Error: Script file '{script_path}' does not exist!")
        return

    # Define build command based on OS
    if system == "Windows":
        build_command = f'pyinstaller --onefile {script_name}'
        output_file = "dist\main.exe"

    elif system == "Linux": 
        build_command = f'pyinstaller --onefile {script_name}'
        output_file = "dist/main"

    elif system == "Darwin":  # macOS
        build_command = f'pyinstaller --onefile {script_path}'
        output_file = "dist/main"

    else:
        print("Unsupported operating system!")
        return 

    # Run PyInstaller
    print(f"Running: {build_command}")
    os.system(build_command)

    # Ensure the Linux/macOS output file is executable
    if system in ["Linux", "Darwin"] and os.path.exists(output_file):
        os.chmod(output_file, 0o755)

    # Automatically create a .sh launch script for Linux/macOS
    if system in ["Linux", "Darwin"]:
        sh_script_path = "run_G2HDM.sh"
        with open(sh_script_path, "w") as sh_script:
            sh_script.write(f"#!/bin/bash\n./{output_file}\n")
        os.chmod(sh_script_path, 0o755)
        print(f"Shell script created: {sh_script_path}")

    print(f"Executable created: {output_file}")

# Run the script
if __name__ == "__main__":
    #import os
    
    build_executable(os.path.join("build", "G2HDM.py"))
    #build_executable(os.path.join("G2HDM.py"))
