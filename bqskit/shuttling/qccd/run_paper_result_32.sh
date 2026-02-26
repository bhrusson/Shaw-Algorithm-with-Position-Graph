#!/bin/bash

# Array of commands to execute
commands=(
#    'python bqskit/shuttling/qccd/run.py "QAOA_wsq_32_compiled" "SHAPER" "grid" "4" "2" "FM" {i}'
#    'python bqskit/shuttling/qccd/run.py "QAOA_wsq_32_compiled" "SHAPER" "grid" "5" "2" "FM" {i}'
#    'python bqskit/shuttling/qccd/run.py "QFT_wsq_32_compiled" "SHAPER" "grid" "4" "2" "FM" {i}'
    #'python bqskit/shuttling/qccd/run.py "QFT_wsq_32_compiled" "SHAPER" "grid" "5" "2" "FM" {i}'
    #'python bqskit/shuttling/qccd/run.py "TFIM_wsq_n32_s100_compiled" "SHAPER" "grid" "4" "2" "FM" {i}'
    'python bqskit/shuttling/qccd/run.py "TFIM_wsq_n32_s100_compiled" "SHAPER" "grid" "5" "2" "FM" {i}'
    #'python bqskit/shuttling/qccd/run.py "TFXY_wsq_n32_s100_compiled" "SHAPER" "grid" "4" "2" "FM" {i}'
    'python bqskit/shuttling/qccd/run.py "TFXY_wsq_n32_s100_compiled" "SHAPER" "grid" "5" "2" "FM" {i}'

#    'python bqskit/shuttling/qccd/run.py "QAOA_wsq_32_compiled" "SHAW" "grid" "4" "2" "FM" {i}'
#    'python bqskit/shuttling/qccd/run.py "QAOA_wsq_32_compiled" "SHAW" "grid" "5" "2" "FM" {i}'
#    'python bqskit/shuttling/qccd/run.py "QFT_wsq_32_compiled" "SHAW" "grid" "4" "2" "FM" {i}'
#    'python bqskit/shuttling/qccd/run.py "QFT_wsq_32_compiled" "SHAW" "grid" "5" "2" "FM" {i}'
    #'python bqskit/shuttling/qccd/run.py "TFIM_wsq_n32_s100_compiled" "SHAW" "grid" "4" "2" "FM" {i}'
    'python bqskit/shuttling/qccd/run.py "TFIM_wsq_n32_s100_compiled" "SHAW" "grid" "5" "2" "FM" {i}'
    #'python bqskit/shuttling/qccd/run.py "TFXY_wsq_n32_s100_compiled" "SHAW" "grid" "4" "2" "FM" {i}'
    'python bqskit/shuttling/qccd/run.py "TFXY_wsq_n32_s100_compiled" "SHAW" "grid" "5" "2" "FM" {i}'

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

