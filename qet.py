import numpy as np
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.quantum_info import Operator, Statevector
from qiskit.circuit.library import XGate, UnitaryGate

# import the block-encoding
data = np.load("unitaries.npz")
U = UnitaryGate(data["U"])
U_dag = UnitaryGate(data["U_dag"])

# read the phi-sequence
with open("phi_angles.txt", "r") as f:
    phi_set = [float(line.strip()) for line in f]

system = QuantumRegister(3)
anc = QuantumRegister(3)
anc2 = QuantumRegister(1)
c = ClassicalRegister(4)

qc = QuantumCircuit(system, anc, anc2, c)
mcx = XGate().control(3, ctrl_state = '000')

# --------- create the QSVT circuit ---------

qc.h(system[0])
qc.h(system[1])
qc.h(system[2])
qc.barrier()

qc.h(anc2[0])

for k in reversed(range(0, len(phi_set))):
    if k % 2 == 0:
        qc.append(U, [system[0], system[1], system[2], anc[0], anc[1], anc[2]])
        qc.append(mcx, [anc[0], anc[1], anc[2], anc2[0]])
        qc.rz(-2 * phi_set[k], anc2[0])
        qc.append(mcx, [anc[0], anc[1], anc[2], anc2[0]])
    else:   
        qc.append(U_dag, [system[0], system[1], system[2], anc[0], anc[1], anc[2]])
        qc.append(mcx, [anc[0], anc[1], anc[2], anc2[0]])
        qc.rz(-2 * phi_set[k], anc2[0])
        qc.append(mcx, [anc[0], anc[1], anc[2], anc2[0]])

qc.h(anc2[0])

qc.measure(anc[0],c[0])
qc.measure(anc[1], c[1])
qc.measure(anc[2], c[2])
qc.measure(anc2[0], c[3])

fig = qc.draw('mpl', initial_state = True, style = {"name":"clifford"})
plt.savefig("qsvt_circuit.png")

qc.remove_final_measurements()
statevector = Statevector.from_instruction(qc).data

# define a projector operator to |0><0|a
projector = np.zeros((128, 128), dtype = complex)
for i in range(0,8):
    projector[i, i] = 1

goal = projector @ statevector
for i in range(0, 8):
    print(round(goal[i].real * 2 * 7 / 4, 8))