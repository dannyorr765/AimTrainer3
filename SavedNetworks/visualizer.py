"""
Neural Network Visualizer for the Aim Trainer Neuroevolution Project.

Reads saved network JSON files and draws an interactive graph showing
nodes, connections (colored by weight), and generation progression.

Usage:
    python visualizer.py                          # defaults to ./SavedNetworks
    python visualizer.py /path/to/SavedNetworks
    python visualizer.py --gen 5                  # jump to generation 5
    python visualizer.py --rank 0                 # show best network (default)

Controls:
    Left/Right arrows  — previous/next generation
    Up/Down arrows     — previous/next rank (network within a generation)
    R                  — toggle connection rendering (all / positive only / negative only)
    T                  — toggle thickness scaling
    Q / Escape         — quit
"""

import json
import os
import sys
import argparse
import math
from pathlib import Path

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.lines import Line2D
import numpy as np


# ── Data Loading ──────────────────────────────────────────────────────

def discover_generations(base_dir: str) -> list[str]:
    """Return sorted list of Gen_XXXX folder paths."""
    gens = []
    for name in sorted(os.listdir(base_dir)):
        full = os.path.join(base_dir, name)
        if os.path.isdir(full) and name.startswith("Gen_"):
            gens.append(full)
    return gens


def discover_networks(gen_dir: str) -> list[str]:
    """Return sorted list of network JSON paths within a generation folder."""
    nets = []
    for name in sorted(os.listdir(gen_dir)):
        if name.startswith("network_") and name.endswith(".json"):
            nets.append(os.path.join(gen_dir, name))
    return nets


def load_network(path: str) -> dict:
    """Load a single network JSON file."""
    with open(path, "r") as f:
        return json.load(f)


# ── Weight Reconstruction ────────────────────────────────────────────

def reconstruct_weights(data: dict) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """
    Unflattens the weight array back into per-layer weight matrices and bias vectors.
    Matches NeuralNetwork.cs Flatten/Unflatten order:
        for each layer connection l:
            for each destination neuron j:
                for each source neuron i:
                    weights[l][j][i]
                biases[l][j]
    Returns (weight_matrices, bias_vectors) where:
        weight_matrices[l] has shape (out_size, in_size)
        bias_vectors[l]    has shape (out_size,)
    """
    topology = data["topology"]
    flat = np.array(data["weights"], dtype=np.float32)

    weight_matrices = []
    bias_vectors = []
    idx = 0

    for l in range(len(topology) - 1):
        in_size = topology[l]
        out_size = topology[l + 1]

        W = np.zeros((out_size, in_size), dtype=np.float32)
        b = np.zeros(out_size, dtype=np.float32)

        for j in range(out_size):
            for i in range(in_size):
                W[j, i] = flat[idx]
                idx += 1
            b[j] = flat[idx]
            idx += 1

        weight_matrices.append(W)
        bias_vectors.append(b)

    return weight_matrices, bias_vectors


# ── Drawing ──────────────────────────────────────────────────────────

INPUT_LABELS = [
    "LOS", "Dist", "Dir X", "Dir Z",
    *[f"Ray {i}" for i in range(12)],
    *[f"Mv{t}{'X' if ax == 0 else 'Z'}" for t in range(4) for ax in range(2)]
]
OUTPUT_LABELS = ["Move X", "Move Z"]


def draw_network(ax, data: dict, weight_matrices: list[np.ndarray],
                 bias_vectors: list[np.ndarray], filter_mode: str = "all",
                 thickness_on: bool = True):
    """Draw the neural network graph on the given axes."""
    ax.clear()
    topology = data["topology"]
    n_layers = len(topology)

    # Layout: x positions evenly spaced, y positions centered per layer
    layer_x = np.linspace(0.1, 0.9, n_layers)
    max_neurons = max(topology)

    # Compute node positions
    positions = []  # positions[l] = list of (x, y) for each neuron in layer l
    for l, size in enumerate(topology):
        x = layer_x[l]
        if size <= 20:
            ys = np.linspace(0.9, 0.1, size)
        else:
            # For large layers, compress spacing
            margin = 0.05
            ys = np.linspace(0.95 - margin, 0.05 + margin, size)
        positions.append([(x, y) for y in ys])

    # Find global weight range for color normalization
    all_weights = np.concatenate([W.flatten() for W in weight_matrices])
    w_absmax = max(abs(all_weights.min()), abs(all_weights.max()), 1e-6)

    # Draw connections
    for l in range(n_layers - 1):
        W = weight_matrices[l]
        out_size, in_size = W.shape

        for j in range(out_size):
            for i in range(in_size):
                w = W[j, i]

                if filter_mode == "pos" and w < 0:
                    continue
                if filter_mode == "neg" and w > 0:
                    continue

                # Color: blue for positive, red for negative
                intensity = abs(w) / w_absmax
                if w > 0:
                    color = (0.1, 0.3, 0.9, 0.05 + 0.6 * intensity)
                else:
                    color = (0.9, 0.15, 0.15, 0.05 + 0.6 * intensity)

                lw = 0.3 + 2.5 * intensity if thickness_on else 0.5

                x0, y0 = positions[l][i]
                x1, y1 = positions[l + 1][j]
                ax.plot([x0, x1], [y0, y1], color=color, linewidth=lw,
                        solid_capstyle="round", zorder=1)

    # Draw nodes
    for l, layer_positions in enumerate(positions):
        size = topology[l]
        for n_idx, (x, y) in enumerate(layer_positions):
            # Node color by layer
            if l == 0:
                fc = "#2196F3"       # input: blue
            elif l == n_layers - 1:
                fc = "#4CAF50"       # output: green
            else:
                fc = "#FF9800"       # hidden: orange

            node_size = 8 if size <= 20 else max(3, 8 - (size - 20) * 0.1)
            circle = plt.Circle((x, y), 0.008 * (node_size / 8),
                                facecolor=fc, edgecolor="white",
                                linewidth=0.5, zorder=3)
            ax.add_patch(circle)

            # Labels for input/output layers
            if l == 0 and n_idx < len(INPUT_LABELS):
                ax.text(x - 0.02, y, INPUT_LABELS[n_idx],
                        ha="right", va="center", fontsize=5, color="#666")
            elif l == n_layers - 1 and n_idx < len(OUTPUT_LABELS):
                ax.text(x + 0.02, y, OUTPUT_LABELS[n_idx],
                        ha="left", va="center", fontsize=7, color="#333",
                        fontweight="bold")

    # Layer size labels at top
    for l, size in enumerate(topology):
        x = layer_x[l]
        if l == 0:
            label = f"Input\n{size}"
        elif l == n_layers - 1:
            label = f"Output\n{size}"
        else:
            label = f"Hidden {l}\n{size}"
        ax.text(x, 0.98, label, ha="center", va="top", fontsize=8,
                color="#333", fontweight="bold")

    # Info
    gen = data.get("generation", "?")
    rank = data.get("rank", "?")
    fitness = data.get("fitness", 0)
    total_weights = sum(W.size for W in weight_matrices)
    total_biases = sum(b.size for b in bias_vectors)

    info = (f"Generation {gen}  |  Rank #{rank}  |  "
            f"Fitness: {fitness:.2f}s  |  "
            f"Params: {total_weights + total_biases:,} "
            f"({total_weights:,}W + {total_biases:,}B)")
    ax.set_title(info, fontsize=10, pad=12)

    filter_label = {"all": "All", "pos": "Positive only", "neg": "Negative only"}
    thick_label = "Thickness ON" if thickness_on else "Thickness OFF"
    ax.text(0.5, 0.01,
            f"[← → gen]  [↑ ↓ rank]  "
            f"[R: filter={filter_label[filter_mode]}]  "
            f"[T: {thick_label}]  [Q: quit]",
            ha="center", va="bottom", fontsize=7, color="#999",
            transform=ax.transAxes)

    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.02, 1.05)
    ax.set_aspect("equal")
    ax.axis("off")


# ── Weight Distribution Histogram ────────────────────────────────────

def draw_histogram(ax, weight_matrices: list[np.ndarray], data: dict):
    """Draw a histogram of all weights."""
    ax.clear()
    all_w = np.concatenate([W.flatten() for W in weight_matrices])

    ax.hist(all_w, bins=80, color="#2196F3", alpha=0.7, edgecolor="none")
    ax.axvline(0, color="#333", linewidth=0.5, linestyle="--")
    ax.set_xlabel("Weight value", fontsize=8)
    ax.set_ylabel("Count", fontsize=8)

    std = np.std(all_w)
    mean = np.mean(all_w)
    ax.set_title(f"Weight Distribution (Gen {data.get('generation', '?')})  —  "
                 f"μ={mean:.3f}  σ={std:.3f}", fontsize=9)
    ax.tick_params(labelsize=7)


# ── Fitness Progression Chart ────────────────────────────────────────

def draw_fitness_progression(ax, all_gens: list[str]):
    """Plot best and average fitness across generations."""
    ax.clear()
    gens = []
    bests = []
    avgs = []

    for gen_dir in all_gens:
        nets = discover_networks(gen_dir)
        if not nets:
            continue
        fitnesses = []
        gen_num = None
        for net_path in nets:
            d = load_network(net_path)
            fitnesses.append(d.get("fitness", 0))
            if gen_num is None:
                gen_num = d.get("generation", 0)
        if gen_num is not None:
            gens.append(gen_num)
            bests.append(max(fitnesses))
            avgs.append(sum(fitnesses) / len(fitnesses))

    if gens:
        ax.plot(gens, bests, "o-", color="#4CAF50", markersize=3,
                linewidth=1.5, label="Best")
        ax.plot(gens, avgs, "s--", color="#FF9800", markersize=2,
                linewidth=1, alpha=0.7, label="Avg (top 5)")
        ax.legend(fontsize=7)

    ax.set_xlabel("Generation", fontsize=8)
    ax.set_ylabel("Fitness (seconds survived)", fontsize=8)
    ax.set_title("Fitness Progression", fontsize=9)
    ax.tick_params(labelsize=7)
    ax.grid(True, alpha=0.3)


# ── Main ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Neuroevolution Network Visualizer")
    parser.add_argument("directory", nargs="?", default="SavedNetworks",
                        help="Path to SavedNetworks folder")
    parser.add_argument("--gen", type=int, default=None,
                        help="Starting generation number")
    parser.add_argument("--rank", type=int, default=0,
                        help="Starting rank (0 = best)")
    args = parser.parse_args()

    base_dir = args.directory
    if not os.path.isdir(base_dir):
        print(f"Directory not found: {base_dir}")
        print("Run the aim trainer for at least one generation to create saved networks.")
        sys.exit(1)

    all_gens = discover_generations(base_dir)
    if not all_gens:
        print(f"No Gen_XXXX folders found in {base_dir}")
        sys.exit(1)

    # State
    gen_idx = 0
    if args.gen is not None:
        # Find the generation folder matching the requested number
        for i, g in enumerate(all_gens):
            folder_name = os.path.basename(g)
            num = int(folder_name.split("_")[1])
            if num == args.gen:
                gen_idx = i
                break

    rank_idx = args.rank
    filter_mode = "all"    # "all", "pos", "neg"
    thickness_on = True

    # Set up figure
    fig = plt.figure(figsize=(14, 8), facecolor="#fafafa")
    fig.canvas.manager.set_window_title("Neuroevolution Visualizer")

    # Layout: big network graph on top, histogram + fitness below
    ax_net = fig.add_axes([0.02, 0.3, 0.96, 0.68])
    ax_hist = fig.add_axes([0.06, 0.06, 0.38, 0.2])
    ax_fit = fig.add_axes([0.56, 0.06, 0.38, 0.2])

    def refresh():
        nets = discover_networks(all_gens[gen_idx])
        if not nets:
            return

        r = min(rank_idx, len(nets) - 1)
        data = load_network(nets[r])
        W, b = reconstruct_weights(data)

        draw_network(ax_net, data, W, b, filter_mode, thickness_on)
        draw_histogram(ax_hist, W, data)
        draw_fitness_progression(ax_fit, all_gens)

        # Highlight current gen on fitness chart
        gen_num = data.get("generation", 0)
        for line in ax_fit.get_lines():
            pass  # fitness chart redraws each time, no highlight needed
        ax_fit.axvline(gen_num, color="#2196F3", alpha=0.4, linewidth=2)

        fig.canvas.draw_idle()

    def on_key(event):
        nonlocal gen_idx, rank_idx, filter_mode, thickness_on

        if event.key in ("q", "escape"):
            plt.close(fig)
            return

        if event.key == "right":
            gen_idx = min(gen_idx + 1, len(all_gens) - 1)
        elif event.key == "left":
            gen_idx = max(gen_idx - 1, 0)
        elif event.key == "up":
            rank_idx = max(rank_idx - 1, 0)
        elif event.key == "down":
            rank_idx += 1
        elif event.key == "r":
            cycle = {"all": "pos", "pos": "neg", "neg": "all"}
            filter_mode = cycle[filter_mode]
        elif event.key == "t":
            thickness_on = not thickness_on
        else:
            return

        refresh()

    fig.canvas.mpl_connect("key_press_event", on_key)
    refresh()
    plt.show()


if __name__ == "__main__":
    main()
