#!/home/rj/.pyenv/shims/python 

#-------------------- Imports and Environment Setup --------------------
import os
import time
import json
import copy
import numpy as np

import snntorch as snn
import torch

from snntorch import utils
from snntorch import spikegen

from torch.utils.data import DataLoader 
from torch import nn

import matplotlib.pyplot as plt
import snntorch.spikeplot as splt
from IPython.display import HTML, display

from load import *
from main import *

#-------------------- Update Leaky surrogate  --------------------


class LeakySurrogate(nn.Module):
    def __init__(self, beta, hidden_size=None, threshold=1.0):
        super(LeakySurrogate, self).__init__()

        # initialize decay rate beta and threshold
        self.beta = beta
        self.threshold = threshold
        self.spike_gradient = self.ATan.apply  # we can also use sigmoid, tanh or fast sigmoid functions here

    def init_state(self, shape, device):
        mem = torch.zeros(shape, device=device)
        return (mem,)  # state = (mem,) -- mem last (and only) element, by convention

    # the forward function is called each time we call Leaky
    def forward(self, input_, state):
        (mem,) = state
        spk = self.spike_gradient((mem - self.threshold))  # call the Heaviside function
        reset = (self.beta * spk * self.threshold).detach()  # remove reset from computational graph
        mem = self.beta * mem + input_ - reset  # update the membrane potential
        return spk, (mem,)

    # Forward pass: Heaviside function
    # Backward pass: Override Dirac Delta with the ArcTan function
    @staticmethod
    class ATan(torch.autograd.Function):
        @staticmethod
        def forward(ctx, mem):
            spk = (mem > 0).float()  # Heaviside on the forward pass
            ctx.save_for_backward(mem)  # store the membrane for use in the backward pass
            return spk

        @staticmethod
        def backward(ctx, grad_output):
            (mem,) = ctx.saved_tensors  # retrieve the membrane potential
            grad = 1 / (1 + (np.pi * mem).pow_(2)) * grad_output  # surrogate gradient using ArcTan function
            return grad

#-------------------- Recurrent Leaky neuron (no self-connections) --------------------

class RLeakySurrogate(nn.Module):
    def __init__(self, beta, hidden_size, threshold=1.0):
        super(RLeakySurrogate, self).__init__()

        self.beta = beta
        self.threshold = threshold
        self.hidden_size = hidden_size
        self.spike_gradient = self.ATan.apply

        # recurrent weight matrix: hidden_size x hidden_size
        self.V = nn.Parameter(torch.randn(hidden_size, hidden_size) * (1.0 / hidden_size**0.5))

        # fixed buffer (no gradient accumulation): 1 everywhere except diagonal -> blocks self-connections
        self.register_buffer('mask', 1 - torch.eye(hidden_size))

    def init_state(self, shape, device):
        spk = torch.zeros(shape, device=device)
        mem = torch.zeros(shape, device=device)
        return (spk, mem)  

    def forward(self, input_, state):
        spk, mem = state

        # mask diagonal every forward call so self-connections never contribute,
        # even if gradients push V's diagonal away from 0
        V_masked = self.V * self.mask

        rec_input = spk @ V_masked.T   # recurrent contribution from previous spikes
        spk_new = self.spike_gradient((mem - self.threshold))
        reset = (self.beta * spk_new * self.threshold).detach()
        mem_new = self.beta * mem + input_ + rec_input - reset

        return spk_new, (spk_new, mem_new)

    @staticmethod
    class ATan(torch.autograd.Function):
        @staticmethod
        def forward(ctx, mem):
            spk = (mem > 0).float()
            ctx.save_for_backward(mem)
            return spk

        @staticmethod
        def backward(ctx, grad_output):
            (mem,) = ctx.saved_tensors
            grad = 1 / (1 + (np.pi * mem).pow_(2)) * grad_output
            return grad

#-------------------- Node: one spiking layer with arbitrary fan-in --------------------

class LayerNode(nn.Module):
    """A single graph node > (one Linear projection per incoming edge [layer]) -> spiking neuron.
    Handles branching (many nodes reading the same source) and skip connections
    (a node reading from any earlier node, not just its immediate predecessor)
    automatically, since 'inputs' is just a list of source node names."""

    _seen_neuron_ids = set()  # class-level, tracked across all LayerNode instances

    def __init__(self, in_features_list, out_features, neuron_factory, combine='sum'):
        super().__init__()
        self.combine = combine
        self.out_features = out_features
        
        # operations over incoming projections
        if combine == 'sum':
            # the projection from each incoming layer node has separate weights.
            self.projections = nn.ModuleList([nn.Linear(f, out_features) for f in in_features_list])
        elif combine == 'concat':
            # all incoming projection weights are concatenated by adding dimensions, so only one projection is needed.
            self.projections = nn.ModuleList([nn.Linear(sum(in_features_list), out_features)])
        else:
            raise ValueError("combine must be 'sum' or 'concat'")

        self.neuron = neuron_factory(out_features)  # neuron_factory(size) -> neuron instance
        
        # guard: catches a factory that returns an already-used pre-built instance
        # (e.g. `lambda size: lif2` reused across two node_specs entries) instead of
        # constructing a fresh one -- that silently shares weights/gradients across nodes.
        if id(self.neuron) in LayerNode._seen_neuron_ids:
            raise ValueError(
                "neuron_factory returned an already-used neuron instance for this node — "
                "pass a constructor, not a pre-built object."
            )
        LayerNode._seen_neuron_ids.add(id(self.neuron))


    def init_state(self, batch_size, device):
        return self.neuron.init_state((batch_size, self.out_features), device)

    def forward(self, inputs, state):
        if self.combine == 'sum':
            # add their current contributions together, then feed to the neuron
            cur = sum(proj(x) for proj, x in zip(self.projections, inputs))
        else:
            # concatenate all incoming projections, then feed to the neuron
            cur = self.projections[0](torch.cat(inputs, dim=-1))
        spk, state = self.neuron(cur, state)
        return spk, state


#-------------------- Graph: arbitrary DAG of NeuronNodes --------------------

class SNNGraph(nn.Module):
    def __init__(self, node_specs, input_size, input_name='input', output_names=None, device='cpu'):
        """
        node_specs: list of (name, {'inputs': [names], 'size': int,
                                     'neuron_factory': fn(size)->neuron, 'combine': 'sum'|'concat'})
                    MUST be listed in topological order (a node's inputs must already
                    appear earlier in the list, or be `input_name`).
        input_name: key for the raw input at each timestep (treated as a virtual node)
        output_names: node names whose (spk, mem) you want returned
        """
        super().__init__()
        self.device = device
        self.input_name = input_name
        self.output_names = output_names
        self.node_order = [name for name, _ in node_specs]
        self.node_inputs = {name: spec['inputs'] for name, spec in node_specs}

        sizes = {input_name: input_size}
        nodes = {}
        for name, spec in node_specs:
            # read sizes of all input nodes, to create the Linear projections
            in_sizes = [sizes[src] for src in spec['inputs']]  # KeyError here = broken topological order
            # inti NeuronNode with custom neuron, size, and combine method (projection type)
            nodes[name] = LayerNode(in_sizes, spec['size'], spec['neuron_factory'], spec.get('combine', 'sum'))
            # store this node's size for future nodes to read
            sizes[name] = spec['size']
        self.nodes = nn.ModuleDict(nodes)
 
    def forward(self, x, record_all=False):
        num_steps, batch_size = x.size(0), x.size(1)
        states = {name: self.nodes[name].init_state(batch_size, self.device) for name in self.node_order}

        record_names = self.node_order if record_all else self.output_names
        spk_rec = {name: [] for name in record_names}
        mem_rec = {name: [] for name in record_names}

        for step in range(num_steps):
            spikes = {self.input_name: x[step]}
            for name in self.node_order:
                inputs = [spikes[src] for src in self.node_inputs[name]]
                spk, states[name] = self.nodes[name](inputs, states[name])
                spikes[name] = spk
                if name in record_names:
                    spk_rec[name].append(spk)
                    mem_rec[name].append(states[name][-1])

        out = {name: (torch.stack(spk_rec[name]), torch.stack(mem_rec[name])) for name in record_names}
        if not record_all and len(out) == 1:
            return next(iter(out.values()))
        return out

#%%%%%%%%%%%%%%%%%%%%%%% Version 2 %%%%%%%%%%%%%%%%%%%%%%%


class LayerNode2(nn.Module):
    """A single graph node > (one Linear projection per incoming edge [layer]) -> spiking neuron.
    Handles branching (many nodes reading the same source), skip connections
    (a node reading from any earlier node, not just its immediate predecessor),
    and recurrent/feedback edges (via delayed=True inputs, see SNNGraph2) automatically,
    since 'inputs' is just a list of source node names."""

    _seen_neuron_ids = set()  # class-level, tracked across all LayerNode2 instances

    def __init__(self, in_features_list, out_features, neuron_factory, combine='sum'):
        super().__init__()
        self.combine = combine
        self.out_features = out_features

        # operations over incoming projections
        if combine == 'sum':
            # the projection from each incoming layer node has separate weights.
            self.projections = nn.ModuleList([nn.Linear(f, out_features) for f in in_features_list])
        elif combine == 'concat':
            # all incoming projection weights are concatenated by adding dimensions, so only one projection is needed.
            self.projections = nn.ModuleList([nn.Linear(sum(in_features_list), out_features)])
        else:
            raise ValueError("combine must be 'sum' or 'concat'")

        self.neuron = neuron_factory(out_features)  # neuron_factory(size) -> neuron instance

        # guard: catches a factory that returns an already-used pre-built instance
        # (e.g. `lambda size: lif2` reused across two node_specs entries) instead of
        # constructing a fresh one -- that silently shares weights/gradients across nodes.
        if id(self.neuron) in LayerNode2._seen_neuron_ids:
            raise ValueError(
                "neuron_factory returned an already-used neuron instance for this node — "
                "pass a constructor, not a pre-built object."
            )
        LayerNode2._seen_neuron_ids.add(id(self.neuron))

    def init_state(self, batch_size, device):
        return self.neuron.init_state((batch_size, self.out_features), device)

    def forward(self, inputs, state):
        if self.combine == 'sum':
            # add their current contributions together, then feed to the neuron
            cur = sum(proj(x) for proj, x in zip(self.projections, inputs))
        else:
            # concatenate all incoming projections, then feed to the neuron
            cur = self.projections[0](torch.cat(inputs, dim=-1))
        spk, state = self.neuron(cur, state)
        return spk, state


#-------------------- Graph: arbitrary DAG (+ delayed feedback edges) of LayerNodes --------------------

class SNNGraph2(nn.Module):
    def __init__(self, node_specs, input_size, input_name='input', output_names=None, device='cpu'):
        """
        node_specs: list of (name, {'inputs': [names], 'size': int,
                                     'neuron_factory': fn(size)->neuron, 'combine': 'sum'|'concat',
                                     'delayed': [bool, ...]})  # one bool per entry in 'inputs'
                    MUST be listed in an order where every NON-delayed input already appears
                    earlier in the list (or is `input_name`). Delayed inputs may point to ANY
                    node in the graph, including ones defined later -- this is what makes
                    inter-layer feedback (e.g. h1 -> h2 -> h1) constructible: mark the edge
                    that closes the loop as delayed=True, and it reads that source's spike
                    from t-1 instead of t. At t=0, every delayed input is zero (nothing has
                    happened yet) -- this is a real modeling assumption, not just a default.
        input_size: width of the raw input at each timestep, needed to size the first layer(s).
        input_name: key for the raw input at each timestep (treated as a virtual node).
        output_names: node names whose (spk, mem) you want returned.
        """
        super().__init__()
        self.device = device
        self.input_name = input_name
        self.output_names = output_names
        self.node_order = [name for name, _ in node_specs]
        self.node_inputs = {name: spec['inputs'] for name, spec in node_specs}
        self.node_delayed = {name: spec.get('delayed', [False] * len(spec['inputs'])) for name, spec in node_specs}

        all_node_names = {input_name} | {name for name, _ in node_specs}

        # -------- validate lengths match, so zip() can't silently truncate a mismatched pair --------
        for name, spec in node_specs:
            if len(self.node_delayed[name]) != len(spec['inputs']):
                raise ValueError(f"Node '{name}': 'delayed' has {len(self.node_delayed[name])} entries "
                                  f"but 'inputs' has {len(spec['inputs'])} -- must match 1:1.")
            # -------- validate every source name actually exists somewhere in the graph --------
            for src in spec['inputs']:
                if src not in all_node_names:
                    raise ValueError(f"Node '{name}' references unknown source '{src}' "
                                      f"(not in node_specs and not input_name).")

        # -------- validate: every non-delayed edge must be forward-topological --------
        sizes = {input_name: input_size}
        seen = {input_name}
        nodes = {}
        for name, spec in node_specs:
            for src, delayed in zip(spec['inputs'], self.node_delayed[name]):
                if src not in seen and not delayed:
                    raise ValueError(
                        f"Node '{name}' reads '{src}' which hasn't been computed yet this timestep. "
                        f"Either reorder node_specs so '{src}' comes first, or mark this edge delayed=True "
                        f"to read '{src}''s previous-timestep output instead."
                    )
            # read sizes of all input nodes, to create the Linear projections
            # (safe for delayed edges too: their source's *size* is still known even if
            #  its *value* at this timestep isn't computed until later in node_order)
            in_sizes = [sizes[src] for src in spec['inputs']]
            nodes[name] = LayerNode2(in_sizes, spec['size'], spec['neuron_factory'], spec.get('combine', 'sum'))
            sizes[name] = spec['size']
            seen.add(name)
        self.nodes = nn.ModuleDict(nodes)

    def forward(self, x, record_all=False):
        num_steps, batch_size = x.size(0), x.size(1)
        states = {name: self.nodes[name].init_state(batch_size, self.device) for name in self.node_order}
        prev_spikes = {name: torch.zeros(batch_size, self.nodes[name].out_features, device=self.device)
                       for name in self.node_order}

        record_names = self.node_order if record_all else self.output_names
        spk_rec = {name: [] for name in record_names}
        mem_rec = {name: [] for name in record_names}

        for step in range(num_steps):
            cur_spikes = {self.input_name: x[step]}
            for name in self.node_order:
                inputs = [prev_spikes[src] if delayed else cur_spikes[src]
                          for src, delayed in zip(self.node_inputs[name], self.node_delayed[name])]
                spk, states[name] = self.nodes[name](inputs, states[name])
                cur_spikes[name] = spk
                if name in record_names:
                    spk_rec[name].append(spk)
                    mem_rec[name].append(states[name][-1])
            prev_spikes = cur_spikes

        out = {name: (torch.stack(spk_rec[name]), torch.stack(mem_rec[name])) for name in record_names}
        if not record_all and len(out) == 1:
            return next(iter(out.values()))
        return out



if __name__ == "__main__":

    #-------------- Training the Fashion-MNIST SNN Classifier --------------


    #-------------------- Hyperparameters and global variables --------------------

    # Training Parameters
    dtype = torch.float
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    data_path='./data'
    batch_size=128 # based on hardware constraints and dataset size, also adjust learning rate accordingly
    num_steps=100

    FASHION_MNIST_CLASSES = {
        0: 'T-shirt/top',
        1: 'Trouser',
        2: 'Pullover',
        3: 'Dress',
        4: 'Coat',
        5: 'Sandal',
        6: 'Shirt',
        7: 'Sneaker',
        8: 'Bag',
        9: 'Ankle boot',
    }

    # Network Architecture
    num_inputs = 28*28
    num_hidden = 1000
    num_outputs = 10

    # Temporal Dynamics
    num_steps = 25
    beta = 0.95

    # training parmeters 

    learning_rate = 1e-3
    num_epochs = 25

    #-------------------- Load Dataset --------------------

    train_loader, test_loader = load_dataset('fashion-mnist', root=data_path, batch_size=batch_size)

    #-------------------- Neuron Initialization -------------------

    lif1 = LeakySurrogate(beta=beta) 
    lif2 = RLeakySurrogate (beta=beta, hidden_size=num_hidden)

    #-------------------- Model Initialization --------------------
    
    snn_model = SNNGraph(
        node_specs=[
            ('hidden', {'inputs': ['input'], 'size': num_hidden, 'neuron_factory': lambda size: lif2}), #recurrent layer
            ('output', {'inputs': ['hidden'], 'size': num_outputs, 'neuron_factory': lambda size: lif1}), # output layer
        ],
        input_size=num_inputs,
        input_name='input',
        output_names=['output'],
        device=device
    ).to(device)

    #-------------------- Optimizer and Loss Function -------------------- 

    optimizer = torch.optim.Adam(snn_model.parameters(), lr=learning_rate)

    loss = SF_temporal_rate_CE(tau=25, device=device)

    #-------------------- Model Training --------------------

    snn_modeler = Modeler(snn_model, loss, optimizer, train_loader, test_loader, num_steps, device)

    snn_modeler.train(num_epochs=num_epochs, interm_evaluate=True, verbose=True, eval_every=50)

    final_test_acc, final_train_acc = snn_modeler.final_model_evaluation(train_accuracy=True)
    
    snn_modeler.save_results(final_test_acc=final_test_acc, final_train_acc=final_train_acc)


