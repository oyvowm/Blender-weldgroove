'''
# works on windows
import subprocess
import sys

py_exec = str(sys.executable)
# ensure pip is installed
subprocess.call([py_exec, "-m", "ensurepip", "--user" ])
# update pip
subprocess.call([py_exec, "-m", "pip", "install", "--upgrade", "pip" ])
# install packages
subprocess.call([py_exec,"-m", "pip", "install", f"--target={py_exec[:-14]}" + "lib", "scipy"])
'''


# for ubuntu 20.04
import subprocess
import sys
import os
 
# path to python.exe
python_exe = os.path.join(sys.prefix, 'bin', 'python3.7m')
 
# upgrade pip
subprocess.call([python_exe, "-m", "ensurepip"])
subprocess.call([python_exe, "-m", "pip", "install", "--upgrade", "pip"])
 
# install required packages
subprocess.call([python_exe, "-m", "pip", "install", "scipy"])