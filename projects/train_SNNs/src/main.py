#!/home/rj/.pyenv/shims/python 

#-------------------- Imports and Environment Setup --------------------

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


#-------------------- Neuron definition --------------------

# Leaky neuron model, overriding the backward pass with a custom function
class LeakySurrogate(nn.Module):
  def __init__(self, beta, threshold=1.0):
      super(LeakySurrogate, self).__init__()

      # initialize decay rate beta and threshold
      self.beta = beta
      self.threshold = threshold
      self.spike_gradient = self.ATan.apply ## we can also use sigmoid, tanh or fast sigmoid functions here
  
  # the forward function is called each time we call Leaky
  def forward(self, input_, mem):
    spk = self.spike_gradient((mem-self.threshold))  # call the Heaviside function
    reset = (self.beta * spk * self.threshold).detach() # remove reset from computational graph
    mem = self.beta * mem + input_ - reset # update the membrane potential
    return spk, mem

  # Forward pass: Heaviside function
  # Backward pass: Override Dirac Delta with the ArcTan function
  @staticmethod
  class ATan(torch.autograd.Function):
      @staticmethod
      def forward(ctx, mem):
          spk = (mem > 0).float() # Heaviside on the forward pass
          ctx.save_for_backward(mem)  # store the membrane for use in the backward pass
          return spk

      @staticmethod
      def backward(ctx, grad_output):
          (mem,) = ctx.saved_tensors  # retrieve the membrane potential 
          grad = 1 / (1 + (np.pi * mem).pow_(2)) * grad_output # surrogate gradient using ArcTan function
          return grad


#-------------------- Network definition --------------------

# Define Network
class Net(nn.Module):
    def __init__(self, neuron, num_inputs, num_hidden, num_outputs, beta=0.95, device='cpu'):
        super().__init__()

        self.num_inputs = num_inputs
        self.num_hidden = num_hidden 
        self.num_outputs = num_outputs 
        self.device = device

        # Initialize layers
        self.fc1 = nn.Linear(num_inputs, num_hidden)
        self.lif1 = neuron(beta=beta)
        self.fc2 = nn.Linear(num_hidden, num_outputs)
        self.lif2 = neuron(beta=beta)

    def forward(self, x):
        
        self.num_steps = x.size(0)
        self.batch_size = x.size(1)

        # Initialize hidden states at t=0
        mem1 = torch.zeros((self.batch_size, self.num_hidden), device=self.device) # membrane potential
        mem2 = torch.zeros((self.batch_size, self.num_outputs), device=self.device)
        
        # Record the final layer
        spk2_rec = []
        mem2_rec = []

        # computing current for inputs 
        cur1_all = self.fc1(x)

        for step in range(self.num_steps):
            spk1, mem1 = self.lif1(cur1_all[step], mem1)
            cur2 = self.fc2(spk1)
            spk2, mem2 = self.lif2(cur2, mem2)
            spk2_rec.append(spk2)
            mem2_rec.append(mem2)

        return torch.stack(spk2_rec, dim=0), torch.stack(mem2_rec, dim=0)

#-------------------- Performance Metrics --------------------

#-------------------- Training --------------------

class Modeler:
    def __init__(self, model, loss, optimizer, train_loader, test_loader, num_steps, device, path='./data', *kwargs):
        self.net = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.num_steps = num_steps
        self.device = device
        self.loss = loss
        self.optimizer = optimizer
        self._test_iter = None
        self.decoder = loss.decoder 


    def data_loader(self, loader):
        for data, targets in loader:
            data, targets = data.to(self.device), targets.to(self.device)
            spike_data = spikegen.latency(data, num_steps=self.num_steps, normalize=True, clip=True)
            spike_data = spike_data.view(self.num_steps, data.size(0), -1)
            yield spike_data, targets


    def train(self, num_epochs, interm_evaluate=False, verbose=False, eval_every=50):

    
        self.counter = 0
        self.loss_hist = []
        self.test_loss_hist = []

        for epoch in range(num_epochs):
            self.iter_counter = 0
            self.net.train()
            for data, targets in self.data_loader(self.train_loader):
                # forward pass
                spk_rec, mem_rec = self.net(data)

                # initialize the loss & sum over time
                loss_val = self.loss(spk_rec, targets)

                # Gradient calculation + weight update
                self.optimizer.zero_grad()
                loss_val.backward()
                self.optimizer.step()

                # Store loss history for future plotting
                self.loss_hist.append(loss_val.item())

                self.counter += 1
                self.iter_counter += 1

                # print loss and accuracy every eval_every iterations
                if verbose and self.counter % eval_every == 0:
                    if interm_evaluate:
                        test_data, test_targets = self.interm_evaluate(epoch)
                        self._printer(data, targets, epoch, test_data, test_targets)
                    else:
                        self._printer(data, targets, epoch)


    def _get_test_batch(self):
        if self._test_iter is None:
            self._test_iter = iter(self.data_loader(self.test_loader))
        try:
            return next(self._test_iter)
        except StopIteration:
            # exhausted the test set - start a new pass
            self._test_iter = iter(self.data_loader(self.test_loader))
            return next(self._test_iter)

    def interm_evaluate(self, epoch):
        with torch.no_grad():
            self.net.eval()
            test_data, test_targets = self._get_test_batch()
            test_spk, test_mem = self.net(test_data)
            test_loss = self.loss(test_spk, test_targets)
            self.test_loss_hist.append(test_loss.item())
            return test_data, test_targets  

    def _printer(self, data, targets, epoch, test_data=None, test_targets=None):
        print(f"Epoch {epoch}, Iteration {self.iter_counter}")
        print(f"Train | Loss: {self.loss_hist[-1]:.2f}", end="  ")
        self._print_batch_accuracy(data, targets, train=True)

        if test_data is not None:
            print(f"Test  | Loss: {self.test_loss_hist[-1]:.2f}", end="  ")
            self._print_batch_accuracy(test_data, test_targets, train=False)
        print()

    def final_model_evaluation(self, train_accuracy=False):
        # Final evaluation of the model on the test set
        self.net.eval()
        with torch.no_grad():
            total_correct = 0
            total_samples = 0
            
            for test_data, test_targets in self.data_loader(self.test_loader):
                output, _ = self.net(test_data)
                _, predicted = self.decoder(output).max(1)
                total_correct += (predicted == test_targets).sum().item()
                total_samples += test_targets.size(0)
            
            final_accuracy = total_correct / total_samples
            print(f"Final Test Set Accuracy: {final_accuracy*100:.2f}%")

            if train_accuracy:
                total_correct = 0
                total_samples = 0
                
                for train_data, train_targets in self.data_loader(self.train_loader):
                    output, _ = self.net(train_data)
                    _, predicted = self.decoder(output).max(1)
                    total_correct += (predicted == train_targets).sum().item()
                    total_samples += train_targets.size(0)
                
                final_train_accuracy = total_correct / total_samples
                print(f"Final Train Set Accuracy: {final_train_accuracy*100:.2f}%")
        



    def _print_batch_accuracy(self, data, targets, train=False):
        
        output, _ = self.net(data)
        _, idx = self.decoder(output).max(1)
        acc = np.mean((targets == idx).detach().cpu().numpy())

        if train:
            print(f"Train set accuracy for a single minibatch: {acc*100:.2f}%")
        else:
            print(f"Test set accuracy for a single minibatch: {acc*100:.2f}%")

                

class SF_temporal_rate_CE(nn.Module):
    def __init__(self, tau=1, device='cpu'):
        super().__init__()
        self.device = device
        self.ce = nn.CrossEntropyLoss()
        self.tau = tau

    def decoder(self, spk_rec):
        # Compute the temporal rate code by applying an exponential decay to the spike trains 
        S = torch.einsum('t,tbc->bc', torch.exp(-torch.arange(0, spk_rec.size(0), device=self.device)/self.tau), spk_rec)
        return S

    def forward(self, spk_rec, targets):
        # Decode the spike trains to obtain the temporal rate code
        S = self.decoder(spk_rec)

        # Compute the cross-entropy loss between the decoded spike trains and the targets
        return self.ce(S, targets)




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
    num_epochs = 1

    #-------------------- Load Dataset --------------------

    train_loader, test_loader = load_dataset('fashion-mnist', root=data_path, batch_size=batch_size)

    #-------------------- Neuron Initialization -------------------

    neuron = LeakySurrogate

    #-------------------- Model Initialization --------------------
    
    snn_model = Net(neuron, num_inputs, num_hidden, num_outputs, beta=beta, device=device).to(device)

    #-------------------- Optimizer and Loss Function -------------------- 

    optimizer = torch.optim.Adam(snn_model.parameters(), lr=learning_rate)

    loss = SF_temporal_rate_CE(tau=17, device=device)

    #-------------------- Model Training --------------------

    snn_modeler = Modeler(snn_model, loss, optimizer, train_loader, test_loader, num_steps, device)

    snn_modeler.train(num_epochs=num_epochs, interm_evaluate=True, verbose=True, eval_every=50)

    snn_modeler.final_model_evaluation(train_accuracy=True)
    
    # <clean>
    # # Iterate through minibatches
    # data = iter(train_loader)
    # data_it, targets_it = next(data)
    # spike_data = spikegen.latency(data_it, num_steps=num_steps, normalize=True, clip=True)
    # spike_data = spike_data.to(device)
    # spike_data = spike_data.view(num_steps, batch_size, -1)

    # snn_classifier = Net(num_inputs, num_hidden, num_outputs, beta, device).to(device)
    # output = snn_classifier(spike_data)
    # spike_data_sample = output[0][:, 1, :].detach().cpu() # get the spikes for the second sample in the batch
    # print(FASHION_MNIST_CLASSES[targets_it[1].item()])
    # print(spike_data_sample)
    # fig = plt.figure(facecolor="w", figsize=(10, 5))
    # ax = fig.add_subplot(111)
    # splt.raster(spike_data_sample, ax)
    # plt.show()
    # #display(HTML(anim.to_html5_video())
    # <clean>
