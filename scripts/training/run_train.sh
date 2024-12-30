#!/bin/bash

if [ $# -eq 0 ]; then
  echo "No extra config provided, using default config."
fi
 
# Enable opening multiple files
ulimit -n 4096

# Enable to save bigger files
ulimit -f unlimited

# Enable more threads per process by increasing virtual memory (https://stackoverflow.com/questions/344203/maximum-number-of-threads-per-process-in-linux)
ulimit -v unlimited

source $(dirname "${BASH_SOURCE[0]}")/../../configs/local_paths.sh

# For some reason, if I don't use eval, argparse treats array of args separated by space as a string separated by space (i.e. the args don't get parsed correctly)
eval "${SRC_ROOT}/sge_tools/python" "${SRC_ROOT}/src/main.py" "$@"
