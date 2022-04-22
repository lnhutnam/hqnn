import torch
import torch.nn as nn
from torch.autograd import Function
import numpy as np

from quantum_circuit import QuantumCircuit

use_cuda = torch.cuda.is_available()


class HybridFunction(Function):
    """ Hybrid quantum - classical function definition """

    @staticmethod
    def forward(ctx, inputs, quantum_circuit, shift):
        """ Forward pass computation """
        ctx.shift = shift
        ctx.quantum_circuit = quantum_circuit

        expectation_z = []
        for input in inputs:
            expectation_z.append(ctx.quantum_circuit.run(input.tolist()))

        if use_cuda:
            result = torch.tensor(expectation_z).cuda()
        else:
            result = torch.tensor(expectation_z)

        ctx.save_for_backward(inputs, result)

        return result

    @staticmethod
    def backward(ctx, grad_output):
        """ Backward pass computation """
        input, expectation_z = ctx.saved_tensors
        input_list = np.array(input.tolist())

        shift_right = input_list + np.ones(input_list.shape) * ctx.shift
        shift_left = input_list - np.ones(input_list.shape) * ctx.shift

        gradients = []
        for i in range(len(input_list)):
            expectation_right = ctx.quantum_circuit.run(shift_right[i])
            expectation_left = ctx.quantum_circuit.run(shift_left[i])

            if use_cuda:
                gradient = torch.tensor([expectation_right]).cuda(
                ) - torch.tensor([expectation_left]).cuda()
            else:
                gradient = torch.tensor(
                    [expectation_right]) - torch.tensor([expectation_left])
            gradients.append(gradient)

        if use_cuda:
            gradients = torch.tensor([gradients]).cuda()
            gradients = torch.transpose(gradients, 0, 1)
        else:
            gradients = torch.tensor([gradients])
            gradients = torch.transpose(gradients, 0, 1)

        return gradients.float() * grad_output.float(), None, None


class Hybrid(nn.Module):
    """ Hybrid quantum - classical layer definition """

    def __init__(self, backend, shots, shift):
        super(Hybrid, self).__init__()
        self.quantum_circuit = QuantumCircuit(1, backend, shots)
        self.shift = shift

    def forward(self, input):
        return HybridFunction.apply(input, self.quantum_circuit, self.shift)
