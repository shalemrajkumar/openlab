
import torch
import torch.nn as nn

from recurr_net import SNNGraph, LeakySurrogate
from visualize import *

DEVICE = 'cpu'
T = 25

FASHION_MNIST_CLASSES = ['T-shirt', 'Trouser', 'Pullover', 'Dress', 'Coat',
                          'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Boot']

def neuron(beta=0.9):
    return lambda size: LeakySurrogate(beta=beta)


def run(model, input_size, rate=0.3, weight_scale=4.0, seed=0, class_names=None):
    """Boost weights so an untrained net actually spikes, random input, one
    recorded forward pass, then open the 3D viewer."""
    with torch.no_grad():
        for p in model.parameters():
            p.mul_(weight_scale)

    g = torch.Generator().manual_seed(seed)
    x = (torch.rand(T, 1, input_size, generator=g) < rate).float().to(DEVICE)

    model.eval()
    with torch.no_grad():
        spk_dict = model(x, record_all=True)

    sample = {'idx': 0, 'target': 0, 'input': x, 'spk_dict': spk_dict}
    visualize_snn_graph(sample, model, class_names=class_names)


# =====================================================================
# 1. Linear chain: input -> h1 -> h2 -> out
# =====================================================================
def topology_linear_chain():
    specs = [
        ('h1',  {'inputs': ['input'], 'size': 64, 'neuron_factory': neuron()}),
        ('h2',  {'inputs': ['h1'],    'size': 64, 'neuron_factory': neuron()}),
        ('out', {'inputs': ['h2'],    'size': 10, 'neuron_factory': neuron()}),
    ]
    model = SNNGraph(specs, input_size=100, output_names=['out'], device=DEVICE).to(DEVICE)
    run(model, input_size=100, class_names=FASHION_MNIST_CLASSES)


# =====================================================================
# 2. Simple branch + merge: input -> A, input -> B, (A, B) -> C
# =====================================================================
def topology_branch_merge():
    specs = [
        ('A', {'inputs': ['input'],    'size': 50, 'neuron_factory': neuron()}),
        ('B', {'inputs': ['input'],    'size': 50, 'neuron_factory': neuron()}),
        ('C', {'inputs': ['A', 'B'],   'size': 10, 'neuron_factory': neuron()}),
    ]
    model = SNNGraph(specs, input_size=100, output_names=['C'], device=DEVICE).to(DEVICE)
    run(model, input_size=100, class_names=FASHION_MNIST_CLASSES)


# =====================================================================
# 3. Branch + skip connection: input -> A -> B, input -> C (skips A,B),
#    (A, B, C) -> D. A and C are true siblings at the same depth even
#    though C is defined after B in node_specs -- tests that layout
#    groups by real depth, not list position.
# =====================================================================
def topology_branch_skip():
    specs = [
        ('A', {'inputs': ['input'],       'size': 64, 'neuron_factory': neuron()}),
        ('B', {'inputs': ['A'],           'size': 64, 'neuron_factory': neuron()}),
        ('C', {'inputs': ['input'],       'size': 30, 'neuron_factory': neuron()}),  # skip: bypasses A, B
        ('D', {'inputs': ['A', 'B', 'C'], 'size': 10, 'neuron_factory': neuron()}),
    ]
    model = SNNGraph(specs, input_size=100, output_names=['D'], device=DEVICE).to(DEVICE)
    run(model, input_size=100, class_names=FASHION_MNIST_CLASSES)


# =====================================================================
# 4. Three-way fan-out + merge: input -> P1, P2, P3 (parallel, same
#    depth) -> Q -> out. Tests three siblings sharing one slab.
# =====================================================================
def topology_fan_merge():
    specs = [
        ('P1',  {'inputs': ['input'],       'size': 20, 'neuron_factory': neuron()}),
        ('P2',  {'inputs': ['input'],       'size': 20, 'neuron_factory': neuron()}),
        ('P3',  {'inputs': ['input'],       'size': 20, 'neuron_factory': neuron()}),
        ('Q',   {'inputs': ['P1', 'P2', 'P3'], 'size': 40, 'neuron_factory': neuron()}),
        ('out', {'inputs': ['Q'],           'size': 10, 'neuron_factory': neuron()}),
    ]
    model = SNNGraph(specs, input_size=100, output_names=['out'], device=DEVICE).to(DEVICE)
    run(model, input_size=100, class_names=FASHION_MNIST_CLASSES)


if __name__ == "__main__":
    # run ONE at a time -- mlab.show() blocks until you close the window
    # topology_linear_chain()
    # topology_branch_merge()
    # topology_branch_skip()
    topology_fan_merge()
