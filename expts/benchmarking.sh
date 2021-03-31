#!/usr/bin/env bash

mode=""
while getopts ":s:a:d:" opt; do
  case ${opt} in
    s )
    server="$OPTARG"
    ;;
    a )
    algo="$OPTARG"
    ;;
    d )
    mode="--debug"
    ;;
    \? ) echo "Invalid option -$OPTARG. Options are -s [local|graham] -t [unsupervised| supervised]" >&2
    ;;
  esac
done
shift $((OPTIND -1))


#exit
set -e
collections="uspto_50k"

echo "Processing algo"
fname=expts/configs/benchmarking.json
mkdir -p "$(dirname $fname)"
python expts/benchmarking.py -d "$collections" -a "$algo" -o $fname $mode --debug
run_rxn_expts dispatch --server $server --config-file $fname --exp-name rxn_debuging --cpus 8 --partition v100