# Qiskit SABRE / LightSABRE

This folder contains Qiskit-native comparison scripts for the same Eagle and
16x16-grid workloads used in the local PGS tests.

Modes:
- `sabre`: Qiskit's SABRE-compatible LightSABRE setting
- `lightsabre`: Qiskit's default/current LightSABRE implementation

Examples:

```powershell
python -m bqskit_local.testCasesPGS.qiskit_lightsabre.ibmEagle_qiskit --workload stress --mode both
python -m bqskit_local.testCasesPGS.qiskit_lightsabre.grid16_qiskit --rounds 2 --mode both
```
