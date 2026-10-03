"""

- [ ] downsample model
- [ ] templates
- [ ] ...
"""

def selectncode(dataset, net, num_steps, device, idx=None):
    """
    Picks a random (or specific) sample from `dataset` (must have spikegen.latency-style
    access to raw data), encodes it, runs it through `net`, and returns everything
    you need for visualization in one bundle.
    """
    if idx is None:
        idx = np.random.randint(0, len(dataset) - 1)

    raw_data, target = dataset[idx]
    raw_data = raw_data.to(device).unsqueeze(0)  # add batch dim -> (1, 1, 28, 28) or similar

    spike_data = spikegen.latency(raw_data, num_steps=num_steps, normalize=True, clip=True)
    spike_data = spike_data.view(num_steps, 1, -1)  # (num_steps, batch=1, 784)

    net.eval()
    with torch.no_grad():
        spk_dict = net(spike_data, record_all=True)

    return {
        'idx': idx,
        'target': target,
        'input': spike_data,   # (num_steps, 1, 784)
        'spk_dict': spk_dict,       # {layer_name: (spk, mem)}
    }

"""
encode whole dataset 
"""

"""
classify and encode each class 
"""


