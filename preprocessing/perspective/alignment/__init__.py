# set the global path that can import other modules
import os
import sys

# set the path to the root directory of the project
root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
# add the root path to the system path
sys.path.append(root_path)