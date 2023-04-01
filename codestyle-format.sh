#!/usr/bin/env bash


format() {
    local path=$1 
    local check=$2
    py_files=$(find $path -name '*.py')
    target="__init__.py"
    black $py_files
    isort $py_files
    autoflake -r --in-place --remove-unused-variables $py_files
    if [ $check = true ]; then
        pylint $py_files --rcfile=.pylintrc || pylint_ret=$?
        for file in $py_files 
        do
            if ! [[ $file =~ $target ]]; then
                flake8 --max-line-length 100 --max-doc-length 120 $file
            fi
        done
        if [ "$pylint_ret" ]; then
            exit $pylint_ret
        fi
    fi
}

format "tests/" false
format "megbox/" true
