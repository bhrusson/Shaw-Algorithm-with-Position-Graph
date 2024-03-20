# FILES INFORMATION
The experiments are conducted with 3-qubit circuits with different types: __Toffoli__, __Fredkin__, __Quantum Fourier Transform__ and __Quantum Volume__. Those types accordingly saved in the following files: `toffoli.qasm`, `fredkin.qasm`, `qft.qasm` and `qv.qasm`.
* The file contains two directories: `baseline` and `output`
* Directory `baseline` contains the input circuits and were generated as follows: 
  * `toffoli.qasm`: The common implementation of toffoli with 6 cnots using t, tdg, and h gates.
  * `qft.qasm`: The Quantum Fourier Transform circuit was generated using qiskit without swaps at the end.
  * `qv.qasm`: The Quantum Volume circuit was generated using qiskit with 3 layers.
  * `fredkin.qasm`: The fredkin was generated using the common implementation with 8 cnots, but was preprocessed with a default qsearch [1] synthesis call to reduce it to 7 cnots. This ensures the most accurate comparison between pytket-phir and our synthesis based compilation by controlling for circuit-level optimizations.
  * __Note__: Only the fredkin was compiled with qsearch since the other inputs did not show improvement under a default qsearch
* Directory `output` contains the input circuits compiled to the 20 qubits H1 machine using our synthesis-based approach. The circuits are already decomposed into native gates for Quantinuum's computers and have the same behavior as the baseline circuit.
* __Notice__: Our method utilizes permutation-aware synthesis [2], so the input and output qubit orderings can be different, listed below:  
  * __Toffoli__: Initial permutation: `[0, 1, 2]` and  final permutation: `[1, 0, 2]`
  * __Fredkin__: Initial permutation: `[0, 1, 2]` and  final permutation: `[0, 1, 2]`
  * __QFT__: Initial permutation: `[0, 1, 2]` and final permutation: `[0, 2, 1]`
  * __QV__: Initial permutation: `[0, 1, 2]` and final permutation: `[0, 1, 2]`

[1] Davis, Marc G., et al. "Towards Optimal Topology Aware Quantum Circuit Synthesis." Proceedings of the 2020 IEEE International Conference on Quantum Computing and Engineering (QCE), 2020, pp. 223-234. DOI: 10.1109/QCE49297.2020.00036.

[2] Liu, Ji, et al. "Tackling the Qubit Mapping Problem with Permutation-Aware Synthesis." Proceedings of the 2023 IEEE International Conference on Quantum Computing and Engineering (QCE), vol. 01, 2023, pp. 745-756. DOI: 10.1109/QCE57702.2023.00090.
