#!/bin/bash

cd "$(dirname "$0")"

mkdir -p ../data
cd ../data

if [ ! -f gsn-2021-1.zip ]; then
    wget https://www.mimuw.edu.pl/~ciebie/gsn-2021-1.zip
fi

unzip ../data/gsn-2021-1.zip
mv data extracted
