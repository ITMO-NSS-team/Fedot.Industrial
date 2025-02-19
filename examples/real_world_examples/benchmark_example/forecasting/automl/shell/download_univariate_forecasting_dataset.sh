#!/bin/bash

output_dir=./data/univariate_libra
mkdir -p $output_dir

url=https://zenodo.org/api/files/bac72c17-2f56-44f5-afaa-694484187f40

echo "Downloading files to $output_dir..."

_start=1
_end=100
for i in $(seq ${_start} ${_end})
do
    echo $i%
    if ! [ -f $output_dir/economics_$i.csv ]; then
        curl --silent $url/economics_$i.csv --output $output_dir/economics_$i.csv
    fi
    if ! [ -f $output_dir/finance_$i.csv ]; then
        curl --silent $url/finance_$i.csv --output $output_dir/finance_$i.csv
    fi
    if ! [ -f $output_dir/human_$i.csv ]; then
        curl --silent $url/human_$i.csv --output $output_dir/human_$i.csv
    fi
    if ! [ -f $output_dir/nature_$i.csv ]; then
        curl --silent $url/nature_$i.csv --output $output_dir/nature_$i.csv
    fi
        # sleep 0.5 # May be needed to prevent timeouts from website
done

echo "Finshed downloading files to $output_dir."
