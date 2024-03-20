OPENQASM 2.0;
include "qelib1.inc";
gate gate_QFT q0,q1,q2 { h q2; cp(pi/2) q2,q1; cp(pi/4) q2,q0; h q1; cp(pi/2) q1,q0; h q0; }
qreg q[3];
gate_QFT q[0],q[1],q[2];
