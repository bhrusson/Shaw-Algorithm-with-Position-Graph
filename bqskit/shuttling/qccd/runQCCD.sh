#!/bin/bash

# Array of commands to execute
commands=(
    'python bqskit/shuttling/qccd/QCCDSim/run.py "QAOA_16_compiled" "H" "5" "Greedy" "Naive" "1" "0" "0" "FM" "GateSwap"'
    'python bqskit/shuttling/qccd/QCCDSim/run.py "QuantumVolume_16" "H" "5" "Greedy" "Naive" "1" "0" "0" "FM" "GateSwap"'
    'python bqskit/shuttling/qccd/QCCDSim/run.py "QFT_16_compiled" "H" "5" "Greedy" "Naive" "1" "0" "0" "FM" "GateSwap"'
    'python bqskit/shuttling/qccd/QCCDSim/run.py "TFIM_n16_s100_compiled" "H" "5" "Greedy" "Naive" "1" "0" "0" "FM" "GateSwap"'
    'python bqskit/shuttling/qccd/QCCDSim/run.py "TFXY_n16_s100_compiled" "H" "5" "Greedy" "Naive" "1" "0" "0" "FM" "GateSwap"'
    'python bqskit/shuttling/qccd/QCCDSim/run.py "QAOA_20_compiled" "H" "6" "Greedy" "Naive" "1" "0" "0" "FM" "GateSwap"'
    'python bqskit/shuttling/qccd/QCCDSim/run.py "QuantumVolume_20" "H" "6" "Greedy" "Naive" "1" "0" "0" "FM" "GateSwap"'
    'python bqskit/shuttling/qccd/QCCDSim/run.py "QFT_20_compiled" "H" "6" "Greedy" "Naive" "1" "0" "0" "FM" "GateSwap"'

    'python bqskit/shuttling/qccd/QCCDSim/run.py "QAOA_16_compiled" "G2x3" "4" "Greedy" "Naive" "1" "0" "0" "FM" "GateSwap"'
    'python bqskit/shuttling/qccd/QCCDSim/run.py "QuantumVolume_16" "G2x3" "4" "Greedy" "Naive" "1" "0" "0" "FM" "GateSwap"'
    'python bqskit/shuttling/qccd/QCCDSim/run.py "QFT_16_compiled" "G2x3" "4" "Greedy" "Naive" "1" "0" "0" "FM" "GateSwap"'
    'python bqskit/shuttling/qccd/QCCDSim/run.py "TFIM_n16_s100_compiled" "G2x3" "4" "Greedy" "Naive" "1" "0" "0" "FM" "GateSwap"'
    'python bqskit/shuttling/qccd/QCCDSim/run.py "TFXY_n16_s100_compiled" "G2x3" "4" "Greedy" "Naive" "1" "0" "0" "FM" "GateSwap"'
    'python bqskit/shuttling/qccd/QCCDSim/run.py "QAOA_20_compiled" "G2x3" "5" "Greedy" "Naive" "1" "0" "0" "FM" "GateSwap"'
    'python bqskit/shuttling/qccd/QCCDSim/run.py "QuantumVolume_20" "G2x3" "5" "Greedy" "Naive" "1" "0" "0" "FM" "GateSwap"'
    'python bqskit/shuttling/qccd/QCCDSim/run.py "QFT_20_compiled" "G2x3" "5" "Greedy" "Naive" "1" "0" "0" "FM" "GateSwap"'
)

# Run each command multiple times
for cmd in "${commands[@]}"; do
    # Generate a unique log file name for each run
    base_log_name=$(echo "$cmd"| awk -F' ' '{print $5 "_" $6 "_" $3 "_" $4}')
    log_file="bqskit/shuttling/qccd/logs/rebuttal/${base_log_name}_run$.log"

    # Run the command and redirect output to the log file
    eval "${cmd} &> ${log_file}"
    echo "Executed: $cmd -> Log: $log_file"
done

