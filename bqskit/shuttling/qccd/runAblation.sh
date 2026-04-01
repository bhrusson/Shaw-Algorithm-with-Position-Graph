#!/bin/bash

# Array of commands to execute
commands=(
    'python bqskit/shuttling/qccd/run.py "QAOA_16_compiled" "SHAW" "H" "4" "2" "FM" {i}'
    'python bqskit/shuttling/qccd/run.py "QAOA_16_compiled" "SHAW" "H" "5" "2" "FM" {i}'
    'python bqskit/shuttling/qccd/run.py "QuantumVolume_16" "SHAW" "H" "4" "2" "FM" {i}'
    'python bqskit/shuttling/qccd/run.py "QuantumVolume_16" "SHAW" "H" "5" "2" "FM" {i}'
    'python bqskit/shuttling/qccd/run.py "QFT_16_compiled" "SHAW" "H" "4" "2" "FM" {i}'
    'python bqskit/shuttling/qccd/run.py "QFT_16_compiled" "SHAW" "H" "5" "2" "FM" {i}'
    'python bqskit/shuttling/qccd/run.py "TFIM_n16_s100_compiled" "SHAW" "H" "4" "2" "FM" {i}'
    'python bqskit/shuttling/qccd/run.py "TFIM_n16_s100_compiled" "SHAW" "H" "5" "2" "FM" {i}'
    'python bqskit/shuttling/qccd/run.py "TFXY_n16_s100_compiled" "SHAW" "H" "4" "2" "FM" {i}'
    'python bqskit/shuttling/qccd/run.py "TFXY_n16_s100_compiled" "SHAW" "H" "5" "2" "FM" {i}'
    'python bqskit/shuttling/qccd/run.py "QAOA_20_compiled" "SHAW" "H" "5" "2" "FM" {i}'
    'python bqskit/shuttling/qccd/run.py "QAOA_20_compiled" "SHAW" "H" "6" "2" "FM" {i}'
    'python bqskit/shuttling/qccd/run.py "QuantumVolume_20" "SHAW" "H" "5" "2" "FM" {i}'
    'python bqskit/shuttling/qccd/run.py "QuantumVolume_20" "SHAW" "H" "6" "2" "FM" {i}'
    'python bqskit/shuttling/qccd/run.py "QFT_20_compiled" "SHAW" "H" "5" "2" "FM" {i}'
    'python bqskit/shuttling/qccd/run.py "QFT_20_compiled" "SHAW" "H" "6" "2" "FM" {i}'

    'python bqskit/shuttling/qccd/run.py "QAOA_16_compiled" "SHAW" "G2x3" "3" "2" "FM" {i}'
    'python bqskit/shuttling/qccd/run.py "QAOA_16_compiled" "SHAW" "G2x3" "4" "2" "FM" {i}'
    'python bqskit/shuttling/qccd/run.py "QuantumVolume_16" "SHAW" "G2x3" "3" "2" "FM" {i}'
    'python bqskit/shuttling/qccd/run.py "QuantumVolume_16" "SHAW" "G2x3" "4" "2" "FM" {i}'
    'python bqskit/shuttling/qccd/run.py "QFT_16_compiled" "SHAW" "G2x3" "3" "2" "FM" {i}'
    'python bqskit/shuttling/qccd/run.py "QFT_16_compiled" "SHAW" "G2x3" "4" "2" "FM" {i}'
    'python bqskit/shuttling/qccd/run.py "TFIM_n16_s100_compiled" "SHAW" "G2x3" "3" "2" "FM" {i}'
    'python bqskit/shuttling/qccd/run.py "TFIM_n16_s100_compiled" "SHAW" "G2x3" "4" "2" "FM" {i}'
    'python bqskit/shuttling/qccd/run.py "TFXY_n16_s100_compiled" "SHAW" "G2x3" "3" "2" "FM" {i}'
    'python bqskit/shuttling/qccd/run.py "TFXY_n16_s100_compiled" "SHAW" "G2x3" "4" "2" "FM" {i}'
    'python bqskit/shuttling/qccd/run.py "QAOA_20_compiled" "SHAW" "G2x3" "4" "2" "FM" {i}'
    'python bqskit/shuttling/qccd/run.py "QAOA_20_compiled" "SHAW" "G2x3" "5" "2" "FM" {i}'
    'python bqskit/shuttling/qccd/run.py "QuantumVolume_20" "SHAW" "G2x3" "4" "2" "FM" {i}'
    'python bqskit/shuttling/qccd/run.py "QuantumVolume_20" "SHAW" "G2x3" "5" "2" "FM" {i}'
    'python bqskit/shuttling/qccd/run.py "QFT_20_compiled" "SHAW" "G2x3" "4" "2" "FM" {i}'
    'python bqskit/shuttling/qccd/run.py "QFT_20_compiled" "SHAW" "G2x3" "5" "2" "FM" {i}'
)

# Number of repetitions
repetitions=5

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

