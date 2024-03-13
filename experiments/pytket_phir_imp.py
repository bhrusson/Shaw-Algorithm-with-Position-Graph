import json
import numpy as np
from pytket.circuit import Circuit, Unitary1qBox, Unitary2qBox, Unitary3qBox
from pytket.phir.qtm_machine import QtmMachine
from pytket.phir.api import pytket_to_phir


def pytket_circ_gen(target_mat: np.array) -> Circuit:
    if len(target_mat.shape) != 2:
        raise ValueError(f"The acceptable dimension of target matrix is 2, while f{len(target_mat.shape)} dimensions "
                         f"found")

    if target_mat.shape[0] != target_mat.shape[1]:
        raise ValueError(f"The matrix is not square, containing {target_mat.shape[0]} rows "
                         f"and {target_mat.shape[1]} columns")

    n_dim = np.log2(target_mat.shape[0])

    if np.ceil(n_dim) != np.floor(n_dim) or n_dim == 0:
        raise ValueError(f"The matrix dimension is not the power of 2")

    if n_dim > 3.0:
        raise ValueError(f"Current pytket does not support bigger than 3 dimensions matrix block")

    if n_dim == 3.0:
        u_box = Unitary3qBox(target_mat)
        circ = Circuit(3).add_unitary3qbox(u_box, 0, 1, 2)
    elif n_dim == 2.0:
        u_box = Unitary2qBox(target_mat)
        circ = Circuit(2).add_unitary3qbox(u_box, 0, 1)
    else:
        u_box = Unitary1qBox(target_mat)
        circ = Circuit(1).add_unitary3qbox(u_box, 0)
    return circ


def pytket_phir_gen(target_mat: np.array, qtm_machine: QtmMachine) -> json:
    target_circ = pytket_circ_gen(target_mat)
    phir_json = pytket_to_phir(circuit=target_circ, qtm_machine=qtm_machine)
    return phir_json
