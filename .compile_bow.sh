#!/bin/bash
echo "compiling $1"
g++ -ggdb `pkg-config --cflags opencv` -o `basename $1 .cpp` $1 `pkg-config --libs opencv` -lboost_filesystem -lboost_system;
echo "Output file => ${1%.*}"
