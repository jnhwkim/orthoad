#!/bin/bash

# Find the path recursively following parent directories.
# E.g.,
#   find_up.sh . MVTec_AD

path=$1
shift 1
while [[ $path != / ]];
do
    LIST=( $(find -L $path -maxdepth 3 -mindepth 1 -type d -name $1) )
    if [ ${#LIST[@]} -gt 0 ]
    then
        echo ${LIST[0]}
        break
    fi
    # Note: if you want to ignore symlinks, use "$(realpath -s "$path"/..)"
    if [[ "$OSTYPE" == "darwin"* ]]; then
        # greadlink is in GNU coreutils; brew install coreutils
        path="$(greadlink -f "$path"/..)"
    elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
        path="$(readlink -f "$path"/..)"
    fi
done