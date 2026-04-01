#!/bin/bash

# Array of commands to execute
commands=(

    'python bqskit/shuttling/qccd/run.py "TFXY_n16_s100_compiled" "H" "5" "4" "FM" {i}'
    'python bqskit/shuttling/qccd/run.py "TFXY_n16_s100_compiled" "H" "5" "6" "FM" {i}'
    'python bqskit/shuttling/qccd/run.py "TFXY_n16_s100_compiled" "H" "5" "8" "FM" {i}'
    'python bqskit/shuttling/qccd/run.py "TFXY_n16_s100_compiled" "H" "5" "10" "FM" {i}'

    'python bqskit/shuttling/qccd/run.py "TFXY_n16_s100_compiled" "G2x3" "4" "4" "FM" {i}'
    'python bqskit/shuttling/qccd/run.py "TFXY_n16_s100_compiled" "G2x3" "4" "6" "FM" {i}'
    'python bqskit/shuttling/qccd/run.py "TFXY_n16_s100_compiled" "G2x3" "4" "8" "FM" {i}'
    'python bqskit/shuttling/qccd/run.py "TFXY_n16_s100_compiled" "G2x3" "4" "10" "FM" {i}'

    'python bqskit/shuttling/qccd/run.py "QFT_20_compiled" "H" "6" "4" "FM" {i}'
    'python bqskit/shuttling/qccd/run.py "QFT_20_compiled" "H" "6" "6" "FM" {i}'
    'python bqskit/shuttling/qccd/run.py "QFT_20_compiled" "H" "6" "8" "FM" {i}'
    'python bqskit/shuttling/qccd/run.py "QFT_20_compiled" "H" "6" "10" "FM" {i}'

    'python bqskit/shuttling/qccd/run.py "QFT_20_compiled" "G2x3" "5" "4" "FM" {i}'
    'python bqskit/shuttling/qccd/run.py "QFT_20_compiled" "G2x3" "5" "6" "FM" {i}'
    'python bqskit/shuttling/qccd/run.py "QFT_20_compiled" "G2x3" "5" "8" "FM" {i}'
    'python bqskit/shuttling/qccd/run.py "QFT_20_compiled" "G2x3" "5" "10" "FM" {i}'
)

# Number of repetitions
repetitions=3

# Run each command multiple times
for cmd in "${commands[@]}"; do
    for ((i=1; i<=repetitions; i++)); do
        # Replace {i} in the command with the current iteration index
        updated_cmd=$(echo $cmd | sed "s/{i}/$i/g")

        # Generate a unique log file name for each run
        base_log_name=$(echo $updated_cmd | awk -F' ' '{print $5 "_" $6 "_" $3 "_" $4}')
        log_file="bqskit/shuttling/qccd/logs/${base_log_name}_layout${i}.log"

        # Run the command and redirect output to the log file
        eval "${updated_cmd} &> ${log_file}"
        echo "Executed: $cmd -> Log: $log_file"
    done
done

