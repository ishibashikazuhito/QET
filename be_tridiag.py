import numpy as np
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit, QuantumRegister
from qiskit.quantum_info import Operator
from qiskit.circuit.library import RYGate, XGate

alpha, beta, gamma = 0.1, 0.3, 0.3

A = np.array([[alpha, gamma,     0,     0,     0,     0,     0,  beta],
              [ beta, alpha, gamma,     0,     0,     0,     0,     0],
              [    0,  beta, alpha, gamma,     0,     0,     0,     0],
              [    0,     0,  beta, alpha, gamma,     0,     0,     0],
              [    0,     0,     0,  beta, alpha, gamma,     0,     0],
              [    0,     0,     0,     0,  beta, alpha, gamma,     0],
              [    0,     0,     0,     0,     0,  beta, alpha, gamma],
              [gamma,     0,     0,     0,     0,     0,  beta, alpha]])

n = int(np.log2(A.shape[0])) 
thetas = 2 * np.arccos(np.array([alpha - 1, beta, gamma])) 

wire_j = QuantumRegister(n, 'j')
wire_i = QuantumRegister(2, 'i')
anc = QuantumRegister(1, 'ancilla')
qc = QuantumCircuit(wire_j, wire_i, anc)

qc.h(wire_i)

qc.barrier()

controls = [(0, 0), (1, 0), (0, 1)]
for idx, (c0, c1) in enumerate(controls):
    ry = RYGate(thetas[idx]).control(2, ctrl_state=f"{c0}{c1}")
    qc.append(ry, [wire_i[1], wire_i[0], anc[0]])

qc.barrier()

x = XGate().control(3, ctrl_state='111')
qc.append(x, [wire_j[1], wire_j[0], wire_i[0], wire_j[2]])

x = XGate().control(2, ctrl_state='11')
qc.append(x, [wire_j[0], wire_i[0], wire_j[1]])

qc.cx(wire_i[0], wire_j[0])

x = XGate().control(3, ctrl_state='100')
qc.append(x, [wire_j[1], wire_j[0], wire_i[1], wire_j[2]])

x = XGate().control(2, ctrl_state='10')
qc.append(x, [wire_j[0], wire_i[1], wire_j[1]])

qc.cx(wire_i[1], wire_j[0])

qc.barrier()

qc.h(wire_i)

qc.draw("mpl")
#plt.savefig("be_circuit.png")

U = Operator(qc).data

#Usub = np.round(U[:8, :8].real * 4 * norm, 3)
#print(Usub)

U_dag = U.conj().T
np.savez("unitaries.npz", U = U, U_dag = U_dag)
