#!/bin/bash

reset
mkdir -p build
cd build
rm -rf *
cmake ..
make -j12