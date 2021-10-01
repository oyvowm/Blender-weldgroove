import subprocess
import sys

py_exec = str(sys.executable)
# ensure pip is installed
subprocess.call([py_exec, "-m", "ensurepip", "--user" ])
# update pip
subprocess.call([py_exec, "-m", "pip", "install", "--upgrade", "pip" ])
# install packages
subprocess.call([py_exec,"-m", "pip", "install", f"--target={py_exec[:-14]}" + "lib", "scipy"])