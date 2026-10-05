"""
visualize_snn_graph.py

Generic 3D visualizer for an SNNGraph model of arbitrary feedforward
topology: any number of nodes, branching, skip connections, merges.
Reads the graph structure straight off the model.

Layout uses each node's real DEPTH (longest path from input, sibling branches share one y-slab and sit
side-by-side which is helpful to visualize complex linear and branching topologies).

"""

import numpy as np
import matplotlib
from mayavi import mlab

import os 
import tempfile
import imageio.v2 as imageio
from pyface.api import GUI
import time


def default_viz_pars(**kwargs):
    pars = {}
 
    # ---- animation / simulation-facing ----
    pars['spiking'] = True          # True: color by spikes. False: color by membrane voltage
    pars['delay'] = 170             # ms between animation frames
    pars['weight_threshold'] = 0.3  # |weight| above which an edge is drawn
 
    # ---- layout ----
    pars['layer_spacing'] = 30     # distance between depth-slabs
    pars['gap'] = 3.0              # space between sibling nodes at the same depth
    pars['io_spacing_mult'] = 2.0  # internal neuron spacing for input/output nodes (>1 = more spread out)
    pars['hidden_jitter'] = 1.0    # 0..1 multiplier on hidden-node jitter; 0 disables it
    pars['io_jitter'] = 0.0        # same, for input/output nodes -- 0 by default (clean, no jitter)
    pars['node_layout'] = None     # optional {name: 'row' | 'grid'} to override auto layout per node
 
    # ---- glyphs ----
    pars['io_glyph'] = 'cube'
    pars['hidden_glyph'] = 'sphere'
    pars['io_point_size'] = 0.5
    pars['hidden_point_size'] = 0.35
    pars['point_opacity'] = 0.85
    pars['line_width'] = 1
    pars['line_opacity'] = 0.15
 
    # ---- color ----
    pars['cmap_name'] = 'inferno'
    pars['voltage_gamma'] = 1.3
    pars['bg_bottom'] = (0, 0, 0)
    pars['bg_top'] = (10 / 255, 40 / 255, 120 / 255)
    pars['bg_botton'] = (5 / 255, 5 / 255, 10 / 255)
    pars['bg_gamma'] = 1.3
 
    # ---- figure ----
    pars['figure_size'] = (1000, 650)
 
    # external overrides -- unknown keys raise, so a typo'd pars name fails
    # loudly here instead of silently sitting unread
    for k in kwargs:
        if k not in pars:
            raise KeyError(f"Unknown visualization parameter '{k}' -- typo? "
                            f"Valid keys: {sorted(pars)}")
        pars[k] = kwargs[k]
 
    return pars



_ACTIVE_ANIMATORS = []  # keeps animator objects alive; Scene is a Traits
                         # object and won't accept an arbitrary new attribute


def _voltage_lut(n=256, gamma=1.3, cmap_name='inferno'):
    """RGBA lookup table sampled from matplotlib's `cmap_name` (inferno by
    default). Maps continuously from 0 to 1. Applying gamma > 1 eases the 
    walk through the rest of the colormap (through purple/red/orange, out 
    to inferno's bright yellow-white top) instead of a linear jump."""
    try:
        cmap = matplotlib.colormaps[cmap_name]
    except AttributeError:  # older matplotlib without the colormaps registry
        cmap = matplotlib.cm.get_cmap(cmap_name)
 
    t = np.linspace(0, 1, n)
    t_eased = t ** gamma
 
    colors = cmap(t_eased)  # (n, 4) floats in [0, 1]
    return (colors * 255).astype(np.uint8)


def _grid_layout(n, max_row_for_line=12, force=None, spacing_mult=1.0):
    """(row, col) coordinates for n points, centered at 0, scaled by
    `spacing_mult`. `force`: None = auto (single row for small n, grid for
    large n); 'row' forces a single line; 'grid' forces the square-ish grid."""
    use_row = (force == 'row') or (force is None and n <= max_row_for_line)
    if use_row:# lay out as a single row 
        cc = (np.arange(n, dtype=float) - n / 2) * spacing_mult
        rr = np.zeros(n, dtype=float)
        return rr, cc
    cols = int(np.ceil(np.sqrt(n)))# grid breakdown
    rows = int(np.ceil(n / cols))
    rr, cc = np.mgrid[-rows / 2:rows / 2:1, -cols / 2:cols / 2:1] # create a grid of row and column indices with -rows/2 to rows/2 and -cols/2 to cols/2
    return rr.ravel()[:n] * spacing_mult, cc.ravel()[:n] * spacing_mult



def _build_scene(sample, model, pars, class_names=None):
    """Everything from data extraction through drawing the actors --
    exactly what visualize_snn_graph used to do inline, extracted so
    save_rotation_gif can build the identical scene without duplicating it."""
    T = sample['input'].shape[0]
    input_size = sample['input'].shape[-1]
 
    mlab.close(all=True)  # release any windows/GPU resources left over from previous runs
 
    # --------------------- load per-node spike sequences or voltages from model --------------------
 
    """
    - N/A
    """
 
    seqs = {model.input_name: sample['input'][:, 0, :].detach().cpu().numpy()}
    sizes = {model.input_name: input_size}
    for name in model.node_order:
        if name not in sample['spk_dict']:
            raise KeyError(
                f"'{name}' not in sample['spk_dict'] "
                f"(available: {list(sample['spk_dict'].keys())}) -- "
                "make sure selectncode() calls the model with record_all=True."
            )
        spk_or_mem = sample['spk_dict'][name][0 if pars['spiking'] else 1]  # (T, 1, size)
        seqs[name] = spk_or_mem[:, 0, :].detach().cpu().numpy()
        sizes[name] = model.nodes[name].out_features
 
    # ---- which nodes count as "io" (input + every output node) ----
    output_names = model.output_names if model.output_names else [model.node_order[-1]]
    io_names = {model.input_name} | set(output_names)
 
    #----------------- graph depth based (longest path from input), node layout: one layer-slab or grid or linear column per topological position
 
    """
    - add control over grid, slab or linear column layout for each node based on its depth in the graph
    """
 
    depth = {model.input_name: 0}  # init primary input depth
    for name in model.node_order:
        depth[name] = max(depth[src] for src in model.node_inputs[name]) + 1  # build depth from inputs for each node, depth = longest path from input node
 
    all_names = [model.input_name] + model.node_order
    by_depth = {}
    for name in all_names:
        by_depth.setdefault(depth[name], []).append(name)
 
    # ---- layout: one y-slab per depth. Siblings at the same depth sit side-by-side along x, each getting its own block + gap ----
 
    """
    - add control over the gap b/w sibling nodes, layer spacing, width of each node's block
    - add control over the random jitter applied to node at different axes
    """
 
    order = []
    x_all, y_all, z_all, seg_ranges = [], [], [], {}
    cursor = 0  # book keeping for the global neuron index across all nodes at seg_ranges
    for d in sorted(by_depth):
        x_cursor = 0.0  # spacial book keeping for the x position of the next sibling node at this depth
        row_items = []
        for name in by_depth[d]:
            n = sizes[name]  # number of neurons in this node
 
            is_io = name in io_names
            jitter_mult = pars['io_jitter'] if is_io else pars['hidden_jitter']
            spacing_mult = pars['io_spacing_mult'] if is_io else 1.0
            force = (pars['node_layout'] or {}).get(name)
 
            rr, cc = _grid_layout(n, force=force, spacing_mult=spacing_mult)  # get grid layout for this node's neurons
            width = (cc.max() - cc.min() + 1) if n > 1 else spacing_mult  # width of the block
            xs = cc - cc.min() + x_cursor + np.random.rand(n) * 0.6 * jitter_mult  # x positions with zero as the left edge of the block + x_cursor offset + jitter
            zs = rr + np.random.rand(n) * 0.6 * jitter_mult
            ys = np.full(n, d * pars['layer_spacing'], dtype=float) + \
                np.random.rand(n) * (pars['layer_spacing'] * 0.15) * jitter_mult  # node depth * layer_spacing + jitter
            row_items.append((name, xs, zs, ys, n))
            x_cursor += width + pars['gap']

     
        row_width = x_cursor - pars['gap']          # drop the trailing gap from the total
        center_shift = row_width / 2.0
        for name, xs, zs, ys, n in row_items:
            x_all.append(xs - center_shift)         # now centered on x=0
            y_all.append(ys); z_all.append(zs)
            seg_ranges[name] = (cursor, cursor + n)
            cursor += n
            order.append(name)
     
    x = np.hstack(x_all); y = np.hstack(y_all); z = np.hstack(z_all)
    scalars0 = np.hstack([seqs[name][0] for name in order])
    io_mask = np.concatenate([np.full(sizes[name], name in io_names, dtype=bool) for name in order])
 
    # ------- connections: follow the real edges (node_inputs), incl. skip connections -------
    """
    - N/A
    """
 
    edges_from, edges_to = [], []
    for name in model.node_order:
        node = model.nodes[name]  # node obj
        in_names = model.node_inputs[name]
        in_sizes = [sizes[src] for src in in_names]
        to_start, _ = seg_ranges[name]
 
        for i, src in enumerate(in_names):
 
            if node.combine == 'sum':  # sources -> target: w1.T@x1 + w2.T@x2 + ...
                w = node.projections[i].weight.detach().cpu().numpy()  # extract each source's weight matrix for this node
            else:  # source -> target: W_all.T@torch.cat([x1, x2, ...], dim=-1)
                offset = sum(in_sizes[:i])  # neurons index offset for this source
                w = node.projections[0].weight.detach().cpu().numpy()[:, offset:offset + in_sizes[i]]  # extract the weight matrix for this source from the combined weight matrix
 
            fr_local, to_local = (np.abs(w.T) > pars['weight_threshold']).nonzero()  # rows, cols of the weight matrix that exceed the threshold
            fr_start, _ = seg_ranges[src]  # get the global neuron index range
            edges_from.append(fr_local + fr_start)
            edges_to.append(to_local + to_start)
 
    edges_from = np.concatenate(edges_from) if edges_from else np.array([], dtype=int)
    edges_to = np.concatenate(edges_to) if edges_to else np.array([], dtype=int)
 
    # ---- draw on Mayavi ----
    """
    - could update the colormap, point size, point shape, line width, opacity, background color, figure size, etc.
    - add control over the view angle, distance, focal point, etc.
    - add control over the text size, position, etc.
    - add control over the animation speed, delay, etc.
    - add control over the animation loop, step size, etc.
    - add suggested smoothing function
    """
 
    # background
    fig = mlab.figure(bgcolor=pars['bg_bottom'], size=pars['figure_size'])
    fig.scene.renderer.background2 = pars['bg_top']
    fig.scene.renderer.gradient_background = True
 
    # data and if spiking, max and mins
    all_vals = np.concatenate([seqs[name].ravel() for name in order])
    vmin, vmax = (0, 1) if pars['spiking'] else (float(all_vals.min()), float(all_vals.max()))
 
    # draw neurons as cubes/spheres, color by initial spike / voltage
    acts_io = mlab.points3d(x[io_mask], y[io_mask], -z[io_mask], scalars0[io_mask],
                             mode=pars['io_glyph'], scale_factor=pars['io_point_size'], scale_mode='none',
                             colormap='hot', vmin=vmin, vmax=vmax, opacity=pars['point_opacity'])
 
    acts_hidden = mlab.points3d(x[~io_mask], y[~io_mask], -z[~io_mask], scalars0[~io_mask],
                                 mode=pars['hidden_glyph'], scale_factor=pars['hidden_point_size'],
                                 scale_mode='none', colormap='hot', vmin=vmin, vmax=vmax,
                                 opacity=pars['point_opacity'])
 
    # draw connections as lines, color by initial spike / voltage via VTK's scalar_scatter and stripper pipelines
    src_pipe = mlab.pipeline.scalar_scatter(x, y, -z, scalars0)  # redraw point clouds (deliberate duplication), same object cannot be reused, because lines and glyphs are visually separate actors in VTK
    src_pipe.mlab_source.dataset.lines = np.vstack((edges_from, edges_to)).T  # reshapes and transpose according to [[from0,to0], [from1,to1], ...] pair-per-row format VTK convention
    src_pipe.update()
    lines = mlab.pipeline.stripper(src_pipe)  # VTK optimization pass which merges adjacent line segments into longer continuous strips where possible, which renders faster.
 
    connections = mlab.pipeline.surface(lines, colormap='hot', line_width=pars['line_width'],
                                         opacity=pars['line_opacity'], vmin=vmin, vmax=vmax)  # actually draw the lines, color by initial spike / voltage
 
    custom_lut = _voltage_lut(gamma=pars['voltage_gamma'], cmap_name=pars['cmap_name'])
 
    # voltage gradients
    acts_io.module_manager.scalar_lut_manager.lut.table = custom_lut
    acts_hidden.module_manager.scalar_lut_manager.lut.table = custom_lut
    connections.module_manager.scalar_lut_manager.lut.table = custom_lut
 
    # labeling the output neurons with class names if provided
    out_name = model.output_names[0] if model.output_names else model.node_order[-1]  # final layer
    out_start, out_end = seg_ranges[out_name]
    if class_names:
        for i, cname in enumerate(class_names[:out_end - out_start]):
            mlab.text3d(x=x[out_start + i], y=y[out_start + i] + 3, z=-z[out_start + i],
                         text=cname, scale=0.5)  # label, offset in y for visibility
 
    label = class_names[int(sample['target'])] if class_names else str(sample['target'])
    mlab.text(0.01, 0.9, f"sample {sample['idx']}  |  true class: {label}", width=0.3)  # whole scene label, top left corner
 
    # camera view: using spherical coordinates
    n_layers = len(by_depth)
    focalpoint = [(x.min()+x.max())/2, (y.min()+y.max())/2, (z.min()+z.max())/2]
    mlab.view(azimuth=0, elevation=65, distance=pars['layer_spacing'] * n_layers * 1.5,
              focalpoint=focalpoint, reset_roll=False)
 
    return {
        'fig': fig, 'acts_io': acts_io, 'acts_hidden': acts_hidden, 'connections': connections,
    'seqs': seqs, 'order': order, 'io_mask': io_mask, 'T': T, 'n_layers': n_layers, 'focalpoint': focalpoint
    }
 
 
def visualize_snn_graph(sample, model, class_names=None, pars=None):
    if pars is None:
        pars = default_viz_pars()
 
    scene = _build_scene(sample, model, pars, class_names)
    acts_io, acts_hidden = scene['acts_io'], scene['acts_hidden']
    connections, seqs, order, io_mask, T = (
        scene['connections'], scene['seqs'], scene['order'], scene['io_mask'], scene['T'])
 
    # animation: update the scalars of the points and lines at each time step, looping over T
    @mlab.animate(delay=pars['delay'], ui=False)
    def anim():
        step = 0
        while True:
            s = np.hstack([seqs[name][step % T] for name in order])
            acts_io.mlab_source.scalars = s[io_mask]
            acts_hidden.mlab_source.scalars = s[~io_mask]
            connections.mlab_source.scalars = s
            step += 1
            yield
 
    animator = anim()          # keep a reference -- an unassigned animator can be
    _ACTIVE_ANIMATORS.append(animator)  # garbage-collected right after creation,
                                         # silently killing the timer before it fires
    mlab.show()
    return scene['fig']
 
 
def save_rotation_gif(sample, model, class_names=None, pars=None,
                       output_path='rotation.gif', n_frames=120, fps=6,
                       spin_cycles=1, offscreen=False):
    """
    Full 360-degree camera rotation, synced with `spin_cycles` loops

    offscreen=False by default: renders in a real (visible) window rather
    than Mayavi's offscreen mode, since offscreen rendering needs working
    headless-GL support that's less reliable. A software-rendering driver stack handling it unusually) -- offscreen
    sidesteps the window manager entirely. Set False to go back to
    visible-window capture if offscreen mode fails outright on your setup.
 
    """ 

    if pars is None:
        pars = default_viz_pars()
 
    was_offscreen = mlab.options.offscreen
    mlab.options.offscreen = offscreen
    try:
        scene = _build_scene(sample, model, pars, class_names)
        fig = scene['fig']
        T, n_layers = scene['T'], scene['n_layers']
        acts_io, acts_hidden, connections = scene['acts_io'], scene['acts_hidden'], scene['connections']
        seqs, order, io_mask = scene['seqs'], scene['order'], scene['io_mask']
 
        frames = []
        with tempfile.TemporaryDirectory() as tmpdir:
            for i in range(n_frames):
                azimuth = (360.0 * spin_cycles * i / n_frames) % 360
                step = int(i * T * spin_cycles / n_frames) % T
 
                s = np.hstack([seqs[name][step] for name in order])
                acts_io.mlab_source.scalars = s[io_mask]
                acts_hidden.mlab_source.scalars = s[~io_mask]
                connections.mlab_source.scalars = s
 
                mlab.view(azimuth=azimuth, elevation=65,
                          distance=pars['layer_spacing'] * n_layers * 1.5,
                          focalpoint=scene['focalpoint'],
                          reset_roll=False)
 
                mlab.process_ui_events()  # flush pending render/event queue before capturing
 
                frame_path = os.path.join(tmpdir, f'frame_{i:04d}.png')
                mlab.savefig(frame_path, figure=fig)
                frames.append(imageio.imread(frame_path))
                time.sleep(0.08)
 
        imageio.mimsave(output_path, frames, fps=fps, loop=0)
        mlab.close(fig)
    finally:
        mlab.options.offscreen = was_offscreen
 
    return output_path


