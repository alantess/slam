#!/bin/sh

cd build
make -j$(nproc) && ./main
cd ..

