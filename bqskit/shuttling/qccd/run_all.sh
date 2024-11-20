#!/bin/bash

# Array of commands to execute
commands=(
    'python bqskit/shuttling/qccd/run.py "QAOA_16_compiled" "H" "4" "2" "FM" {i}'
    'python bqskit/shuttling/qccd/run.py "QAOA_16_compiled" "H" "5" "2" "FM" {i}'
    'python bqskit/shuttling/qccd/run.py "QuantumVolume_16" "H" "4" "2" "FM" {i}'
    'python bqskit/shuttling/qccd/run.py "QuantumVolume_16" "H" "5" "2" "FM" {i}'
    'python bqskit/shuttling/qccd/run.py "QFT_16_compiled" "H" "4" "2" "FM" {i}'
    'python bqskit/shuttling/qccd/run.py "QFT_16_compiled" "H" "5" "2" "FM" {i}'
    'python bqskit/shuttling/qccd/run.py "TFIM_n16_s100_compiled" "H" "4" "2" "FM" {i}'
    'python bqskit/shuttling/qccd/run.py "TFIM_n16_s100_compiled" "H" "5" "2" "FM" {i}'
    'python bqskit/shuttling/qccd/run.py "TFXY_n16_s100_compiled" "H" "4" "2" "FM" {i}'
    'python bqskit/shuttling/qccd/run.py "TFXY_n16_s100_compiled" "H" "5" "2" "FM" {i}'
    'python bqskit/shuttling/qccd/run.py "QAOA_20_compiled" "H" "5" "2" "FM" {i}'
    'python bqskit/shuttling/qccd/run.py "QAOA_20_compiled" "H" "6" "2" "FM" {i}'
    'python bqskit/shuttling/qccd/run.py "QuantumVolume_20" "H" "5" "2" "FM" {i}'
    'python bqskit/shuttling/qccd/run.py "QuantumVolume_20" "H" "6" "2" "FM" {i}'
    'python bqskit/shuttling/qccd/run.py "QFT_20_compiled" "H" "5" "2" "FM" {i}'
    'python bqskit/shuttling/qccd/run.py "QFT_20_compiled" "H" "6" "2" "FM" {i}'
)

# Number of repetitions
repetitions=5

# Run each command multiple times
for cmd in "${commands[@]}"; do
    for ((i=1; i<=repetitions; i++)); do
        # Replace {i} in the command with the current iteration index
        updated_cmd=$(echo $cmd | sed "s/{i}/$i/g")
        # Extract the base log filename from the command
        log_base=$(echo $updated_cmd | awk -F'>' '{print $2}' | xargs | sed 's/.log//')
        # Add the run number to the log file name
        log_file="${log_base}_run${i}.log"
        # Run the command and redirect output to the log file
        eval "${cmd/&>.*/} &> ${log_file}"
        echo "Executed: $cmd -> Log: $log_file"
    done
done

