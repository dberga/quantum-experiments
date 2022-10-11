import cirq
import qiskit
import math
import warnings
warnings.filterwarnings('ignore')

def _get_qubit_offsets(used_qubits):
    # for simulation we need an array of qubits, so this function
    # computes the offest into the array for each qubit type (in
    # Qiskit's Shor setup)
    qubit_offsets = {}
    prev_count = 0
    for q_name in ('up', 'down', 'aux'):
        qubit_offsets[q_name] = prev_count
        count = 0
        for q in used_qubits:
            if q_name in q:
                count += 1
        prev_count += count
    qubit_offsets['total'] = prev_count
    return qubit_offsets

def _get_qubit_subset(qubits, backend, qubit_offsets):
    out = []
    assert isinstance(backend, cirq.Circuit)
    for q in qubits:
        out.append(cirq.LineQubit(qubit_offsets[q.register.name]+q.index))
    return out

def _convert_and_apply_gate(g, backend, qubit_offsets):
    # based on observation, only the listed gates are used in Qiskit's Shor

    gate, qubits, _ = g  # always a 3-tuple
    if isinstance(gate, qiskit.circuit.library.standard_gates.p.MCPhaseGate):
        n_controls = gate.num_ctrl_qubits
        assert n_controls <= 2
        controls = _get_qubit_subset(qubits[:n_controls], backend, qubit_offsets)
        targets = _get_qubit_subset(qubits[n_controls:], backend, qubit_offsets)
        assert len(controls) + len(targets) <= 3
        assert len(gate.params) == 1
        angle = float(gate.params[0])
        if isinstance(backend, cirq.Circuit):
            #return cirq.CCZPowGate(exponent=angle/math.pi)(*(controls+targets))
            return cirq.ZPowGate(exponent=angle/math.pi)(*targets).controlled_by(*controls)
        else:
            assert False
    elif isinstance(gate, qiskit.circuit.library.standard_gates.swap.SwapGate):
        assert len(qubits) == 2
        targets = _get_qubit_subset(qubits, backend, qubit_offsets)
        if isinstance(backend, cirq.Circuit):
            return cirq.SWAP(*targets)
        else:
            assert False
    elif isinstance(gate, qiskit.circuit.library.standard_gates.x.XGate):
        assert len(qubits) == 1
        targets = _get_qubit_subset(qubits, backend, qubit_offsets)
        if isinstance(backend, cirq.Circuit):
            return cirq.X(*targets)
        else:
            assert False
    elif isinstance(gate, qiskit.circuit.library.standard_gates.h.HGate):
        assert len(qubits) == 1
        targets = _get_qubit_subset(qubits, backend, qubit_offsets)
        if isinstance(backend, cirq.Circuit):
            return cirq.H(*targets)
        else:
            assert False
    elif isinstance(gate, qiskit.circuit.library.standard_gates.p.CPhaseGate):
        n_controls = gate.num_ctrl_qubits
        assert n_controls == 1
        controls = _get_qubit_subset(qubits[:n_controls], backend, qubit_offsets)
        targets = _get_qubit_subset(qubits[n_controls:], backend, qubit_offsets)
        assert len(gate.params) == 1
        angle = float(gate.params[0])
        if isinstance(backend, cirq.Circuit):
            return cirq.CZPowGate(exponent=angle/math.pi)(*controls, *targets)
        else:
            assert False
    elif isinstance(gate, qiskit.circuit.library.standard_gates.p.PhaseGate):
        assert len(qubits) == 1
        targets = _get_qubit_subset(qubits, backend, qubit_offsets)
        assert len(gate.params) == 1
        angle = None
        if not gate.is_parameterized():
            angle = float(gate.params[0])
        if isinstance(backend, cirq.Circuit):
            if angle is None:
                import sympy
                arg = sympy.Symbol(f'q{list(gate.params[0].parameters)[0].index}')
                return cirq.ZPowGate(exponent=arg/math.pi)(*targets)
            return cirq.ZPowGate(exponent=angle/math.pi)(*targets)
        else:
            assert False
    elif isinstance(gate, qiskit.circuit.library.standard_gates.swap.CSwapGate):
        n_controls = gate.num_ctrl_qubits
        assert n_controls == 1
        controls = _get_qubit_subset(qubits[:n_controls], backend, qubit_offsets)
        targets = _get_qubit_subset(qubits[n_controls:], backend, qubit_offsets)
        assert len(targets) == 2
        if isinstance(backend, cirq.Circuit):
            return cirq.CSWAP(*controls, *targets)
        else:
            assert False
    elif isinstance(gate, qiskit.circuit.library.standard_gates.x.CXGate):
        n_controls = gate.num_ctrl_qubits
        assert n_controls == 1
        controls = _get_qubit_subset(qubits[:n_controls], backend, qubit_offsets)
        targets = _get_qubit_subset(qubits[n_controls:], backend, qubit_offsets)
        if isinstance(backend, cirq.Circuit):
            return cirq.CX(*controls, *targets)
        else:
            assert False
    elif isinstance(gate,
            (qiskit.circuit.library.standard_gates.u.UGate,
             qiskit.circuit.library.standard_gates.u3.U3Gate)):
        assert len(qubits) == 1
        targets = get_qubit_subset(qubits, backend, qubit_offsets)
        assert len(gate.params) == 3
        theta = float(gate.params[0])
        phi = float(gate.params[1])
        lam = float(gate.params[2])
        #logger.info(f"UGate: theta={theta}, phi={phi}, lam={lam}")
        if isinstance(backend, cirq.Circuit):
            # note: cannot use QasmUGate because it's not the same as U3 from Qiskit!
            return cirq.MatrixGate(U3_gate(theta, phi, lam, name='U3'))(*targets)
        else:
            assert False
    elif isinstance(gate, qiskit.circuit.library.standard_gates.u2.U2Gate):
        assert len(qubits) == 1
        targets = _get_qubit_subset(qubits, backend, qubit_offsets)
        assert len(gate.params) == 2
        if gate.is_parameterized():
            raise NotImplementedError
        phi = float(gate.params[0])
        lam = float(gate.params[1])
        if isinstance(backend, cirq.Circuit):
            # note: cannot use QasmUGate because it's not the same as U3 from Qiskit!
            return cirq.MatrixGate(U3_gate(0.5*math.pi, phi, lam, name='U2'))(*targets)
        else:
            assert False
    elif isinstance(gate, qiskit.circuit.library.standard_gates.rx.RXGate):
        assert len(qubits) == 1
        targets = _get_qubit_subset(qubits, backend, qubit_offsets)
        assert len(gate.params) == 1
        if gate.is_parameterized():
            raise NotImplementedError
        theta = float(gate.params[0])
        if isinstance(backend, cirq.Circuit):
            return cirq.Rx(rads=theta)(*targets)
        else:
            assert False
    elif isinstance(gate, qiskit.circuit.library.standard_gates.ry.RYGate):
        assert len(qubits) == 1
        targets = _get_qubit_subset(qubits, backend, qubit_offsets)
        assert len(gate.params) == 1
        if gate.is_parameterized():
            raise NotImplementedError
        theta = float(gate.params[0])
        if isinstance(backend, cirq.Circuit):
            return cirq.Ry(rads=theta)(*targets)
        else:
            assert False
    elif isinstance(gate, qiskit.circuit.library.standard_gates.rz.RZGate):
        assert len(qubits) == 1
        targets = _get_qubit_subset(qubits, backend, qubit_offsets)
        assert len(gate.params) == 1
        if gate.is_parameterized():
            raise NotImplementedError
        phi = float(gate.params[0])
        if isinstance(backend, cirq.Circuit):
            return cirq.Rz(rads=phi)(*targets)
        else:
            assert False
    else:
        raise NotImplementedError(f"unrecognized gate: {gate}")
        
def _get_gate_sequence(circuit, targets=None, recursive_level=0):
    # targets is propogated from the top level all the way down (readonly)
    gates = []
    for g in circuit:
        op, qubits, _ = g  # always a 3-tuple
        if targets is not None:
            unique_qreg = set([q.register for q in qubits])
            if len(unique_qreg) == 1:
                # if qubits have the same name, the qubit index is based on the upper level's qubit layout
                applied_qubits = [targets[q.index] for q in qubits]
            else:
                # it's a control circuit with two kinds of QuantumRegister
                assert len(unique_qreg) == 2  # "control" & "target"
                unique_qreg = list(unique_qreg)
                assert len(targets) == sum([qreg.size for qreg in unique_qreg])
                # TODO: check if this convention holds
                applied_qubits = []
                control_size = -1  # control first, and then target, so we need to record offset
                for q in qubits:
                    if q.register.name == "control":
                        applied_qubits.append(targets[q.index])
                        if control_size == -1:
                            control_size = q.register.size
                    elif q.register.name == "target":
                        applied_qubits.append(targets[q.index+control_size])
                    else:
                        assert False, "impossible"
        else:
            applied_qubits = qubits

        if isinstance(g[0], qiskit.circuit.Gate):
            if hasattr(qiskit.circuit.library.standard_gates, g[0].__class__.__name__) :
                # composite gates are applied to placeholder qubits, so
                # we apply it to the upper-level (actual) qubits instead.
                if targets is not None:
                    gates.append((op, applied_qubits, _))  # collect in the 3-tuple form for now
                else:
                    gates.append(g)
            else:
                # a composite gate inheriting qiskit.circuit.Gate,
                # let's walk further down
                group_gates = _get_gate_sequence(op.definition, applied_qubits, recursive_level+1)
                gates += group_gates
        elif isinstance(op, qiskit.circuit.Instruction):
            # also a composite gate, let's walk further down
            group_gates = _get_gate_sequence(op.definition, applied_qubits, recursive_level+1)
            gates += group_gates
        else:
            raise NotImplementedError(f"got {op} of type {type(op)}")

    return gates

def _get_qubit_list(gates):
    used_qubits = set()
    for g in gates:
        for q in g[1]:
            if q not in used_qubits:
                used_qubits.add(q)
    qubit_list = [str(q) for q in used_qubits]
    qubit_list.sort()
    return qubit_list

def convert(qiskit_circuit):
    gates = _get_gate_sequence(qiskit_circuit)
    qubits = _get_qubit_list(gates)
    qubit_offsets = _get_qubit_offsets(qubits)
    cirq_circuit = cirq.Circuit()
    for g in gates:
        cirq_circuit.append(_convert_and_apply_gate(g, cirq_circuit, qubit_offsets))
    return cirq_circuit, qubit_offsets
        
