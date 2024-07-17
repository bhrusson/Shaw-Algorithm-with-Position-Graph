from bqskit.ir import Circuit, Gate
from bqskit.ir.gates import SwapGate, RZGate
from pytket.phir.qtm_machine import QTM_MACHINES_MAP, QtmMachine
from .ShuttlingShift import ShuttlingShiftGate


def get_gate_time(gate: Gate, qtm_machine: QtmMachine) -> float:
    if not isinstance(gate, Gate):
        raise TypeError(
            'Expected Gate type , got %s.'
            % type(gate),
        )
    if not isinstance(qtm_machine, QtmMachine):
        raise TypeError(
            'Expected QtmMachine type , got %s.'
            % type(qtm_machine),
        )
    qtm_machine = QTM_MACHINES_MAP.get(qtm_machine)
    if gate == SwapGate():
        return qtm_machine.qb_swap_time
    if gate.num_qudits == 2:
        return qtm_machine.tq_time
    if gate.num_qudits == 1:
        return qtm_machine.sq_time
    raise ValueError(f"Invalid gate. {gate} is not supported")


def get_duration_from_circ(circuit: Circuit, qtm_machine: QtmMachine) -> float:
    if not isinstance(circuit, Circuit):
        raise TypeError(
            'Expected Circuit type , got %s.'
            % type(circuit),
        )
    if not isinstance(qtm_machine, QtmMachine):
        raise TypeError(
            'Expected QtmMachine type , got %s.'
            % type(qtm_machine),
        )
    circ_depth = circuit.num_cycles
    total_duration = 0.0
    for i in range(circ_depth):
        layer = circuit[i]
        layer_duration = 0.0
        for op in layer:
            gate_duration = get_gate_time(op.gate, qtm_machine)
            if gate_duration > layer_duration:
                layer_duration = gate_duration
        total_duration += layer_duration
    return total_duration


def get_duration_from_circ_after_scheduling(circuit: Circuit, qtm_machine: QtmMachine) -> [float, int]:
    circ_depth = circuit.num_cycles
    total_duration = 0.0
    count_shift_gate = 0
    for ix in range(circ_depth):
        layer = circuit[ix]
        op = layer[0]
        if op.gate == ShuttlingShiftGate(circuit.num_qudits):
            count_shift_gate += 1
            gate_duration = 0
        elif op.gate == RZGate():
            gate_duration = 0
        else:
            gate_duration = get_gate_time(op.gate, qtm_machine)
        total_duration += gate_duration
    return total_duration, count_shift_gate


def check_executable_circuit(circuit: Circuit, tq_zone: list[int]) -> bool:
    if not isinstance(circuit, Circuit):
        raise TypeError(
            'Expected Gate type , got %s.'
            % type(circuit),
        )
    sq_zone = []
    for q in tq_zone:
        sq_zone.append(q)
        sq_zone.append(q + 1)
    circ_depth = circuit.num_cycles
    for i in range(circ_depth):
        layer = circuit[i]
        for op in layer:
            if op.gate == SwapGate():
                continue
            if op.num_qudits == 1 and op.location[0] not in sq_zone:
                print(f"Operation {op.gate} at location {op.location} is not executable")
                return False
            elif op.num_qudits == 2 and op.location[0] not in tq_zone and op.location[1] not in tq_zone:
                print(f"Operation {op.gate} at location {op.location} is not executable")
                return False
    return True
