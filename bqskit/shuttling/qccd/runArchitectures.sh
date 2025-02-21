#!/bin/bash

# Array of commands to execute
commands=(
    'python bqskit/shuttling/qccd/run.py "QAOA_45_compiled" "SHAPER" "H2" "4" "2" "FM" {i}'
    'python bqskit/shuttling/qccd/run.py "QFT_45_compiled" "SHAPER" "H2" "4" "2" "FM" {i}'
    'python bqskit/shuttling/qccd/run.py "bqskit_QuantumVolume_45" "SHAPER" "H2" "4" "2" "FM" {i}'

    'python bqskit/shuttling/qccd/run.py "QFT_50_compiled" "SHAPER" "H2" "4" "2" "FM" {i}'
    'python bqskit/shuttling/qccd/run.py "QAOA_50_compiled" "SHAPER" "H2" "4" "2" "FM" {i}'
    'python bqskit/shuttling/qccd/run.py "bqskit_QuantumVolume_50" "SHAPER" "H2" "4" "2" "FM" {i}'
)

# Number of repetitions
repetitions=1

# Run each command multiple times
for cmd in "${commands[@]}"; do
    for ((i=1; i<=repetitions; i++)); do
        # Replace {i} in the command with the current iteration index
        updated_cmd=$(echo $cmd | sed "s/{i}/$i/g")

        # Generate a unique log file name for each run
        base_log_name=$(echo $updated_cmd | awk -F' ' '{print $5 "_" $6 "_" $3 "_" $4}')
        log_file="bqskit/shuttling/qccd/logs/${base_log_name}_run${i}.log"

        # Run the command and redirect output to the log file
        eval "${updated_cmd} &> ${log_file}"
        echo "Executed: $cmd -> Log: $log_file"
    done
done

