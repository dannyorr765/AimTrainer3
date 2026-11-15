"""
Neural Network Visualizer for the Aim Trainer Neuroevolution Project.

Reads saved network JSON files and draws an interactive graph showing
nodes, connections (colored by weight), and generation progression.

Supports both legacy (fixed-topology) and NEAT (dynamic-topology) formats.

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


def is_neat_format(data: dict) -> bool:
    """Check if a network uses the NEAT format (has 'nodes' key)."""
    return "nodes" in data


# ── Weight Reconstruction (legacy format) ───────────────────────────

def reconstruct_weights(data: dict) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """
    Unflattens the weight array back into per-layer weight matrices and bias vectors.
    Only works with legacy (fixed-topology) format.
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
    "LOS", "Dist", "Dir X", "Dir Y", "Dir Z",
    *[f"Ray {i}" for i in range(12)],
    *[f"Mv{t}{'XYZ'[ax]}" for t in range(4) for ax in range(3)]
]
OUTPUT_LABELS = ["Move X", "Move Y", "Move Z"]


def draw_network_legacy(ax, data: dict, weight_matrices: list[np.ndarray],
                        bias_vectors: list[np.ndarray], filter_mode: str = "all",
                        thickness_on: bool = True):
    """Draw a legacy fixed-topology network graph."""
    ax.clear()
    topology = data["topology"]
    n_layers = len(topology)

    layer_x = np.linspace(0.1, 0.9, n_layers)

    positions = []
    for l, size in enumerate(topology):
        x = layer_x[l]
        if size <= 20:
            ys = np.linspace(0.9, 0.1, size)
        else:
            margin = 0.05
            ys = np.linspace(0.95 - margin, 0.05 + margin, size)
        positions.append([(x, y) for y in ys])

    all_weights = np.concatenate([W.flatten() for W in weight_matrices])
    w_absmax = max(abs(all_weights.min()), abs(all_weights.max()), 1e-6)

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

    for l, layer_positions in enumerate(positions):
        size = topology[l]
        for n_idx, (x, y) in enumerate(layer_positions):
            if l == 0:
                fc = "#2196F3"
            elif l == n_layers - 1:
                fc = "#4CAF50"
            else:
                fc = "#FF9800"
            node_size = 8 if size <= 20 else max(3, 8 - (size - 20) * 0.1)
            circle = plt.Circle((x, y), 0.008 * (node_size / 8),
                                facecolor=fc, edgecolor="white",
                                linewidth=0.5, zorder=3)
            ax.add_patch(circle)
            if l == 0 and n_idx < len(INPUT_LABELS):
                ax.text(x - 0.02, y, INPUT_LABELS[n_idx],
                        ha="right", va="center", fontsize=5, color="#666")
            elif l == n_layers - 1 and n_idx < len(OUTPUT_LABELS):
                ax.text(x + 0.02, y, OUTPUT_LABELS[n_idx],
                        ha="left", va="center", fontsize=7, color="#333",
                        fontweight="bold")

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

    gen = data.get("generation", "?")
    rank = data.get("rank", "?")
    fitness = data.get("fitness", 0)
    total_weights = sum(W.size for W in weight_matrices)
    total_biases = sum(b.size for b in bias_vectors)

    info = (f"Generation {gen}  |  Rank #{rank}  |  "
            f"Fitness: {fitness:.2f}s  |  "
            f"Params: {total_weights + total_biases:,} "
            f"({total_weights:,}W + {total_biases:,}B)  [LEGACY]")
    ax.set_title(info, fontsize=10, pad=12)
    _draw_controls(ax)
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.02, 1.05)
    ax.set_aspect("equal")
    ax.axis("off")


def draw_network_neat(ax, data: dict, filter_mode: str = "all",
                      thickness_on: bool = True):
    """Draw a NEAT dynamic-topology network graph."""
    ax.clear()
    nodes = data["nodes"]
    connections = data["connections"]

    # Group nodes by depth
    depth_groups = {}
    node_map = {}  # id -> node dict
    for n in nodes:
        d = n["depth"]
        if d not in depth_groups:
            depth_groups[d] = []
        depth_groups[d].append(n)
        node_map[n["id"]] = n

    sorted_depths = sorted(depth_groups.keys())
    n_depths = len(sorted_depths)

    # Assign x positions based on depth, y positions spread within group
    x_positions = {}
    y_positions = {}
    depth_to_x = {}

    for di, depth in enumerate(sorted_depths):
        x = 0.1 + 0.8 * (di / max(n_depths - 1, 1))
        depth_to_x[depth] = x
        group = depth_groups[depth]

        n_nodes = len(group)
        if n_nodes <= 20:
            ys = np.linspace(0.9, 0.1, n_nodes) if n_nodes > 1 else [0.5]
        else:
            ys = np.linspace(0.95, 0.05, n_nodes)

        for ni, node in enumerate(group):
            x_positions[node["id"]] = x
            y_positions[node["id"]] = ys[ni]

    # Find weight range for color normalization
    enabled_weights = [c["weight"] for c in connections if c["enabled"]]
    if enabled_weights:
        w_absmax = max(abs(min(enabled_weights)), abs(max(enabled_weights)), 1e-6)
    else:
        w_absmax = 1.0

    # Draw connections
    for c in connections:
        if not c["enabled"]:
            continue
        w = c["weight"]
        if filter_mode == "pos" and w < 0:
            continue
        if filter_mode == "neg" and w > 0:
            continue

        from_id = c["fromNode"]
        to_id = c["toNode"]
        if from_id not in x_positions or to_id not in x_positions:
            continue

        intensity = abs(w) / w_absmax
        if w > 0:
            color = (0.1, 0.3, 0.9, 0.05 + 0.6 * intensity)
        else:
            color = (0.9, 0.15, 0.15, 0.05 + 0.6 * intensity)
        lw = 0.3 + 2.5 * intensity if thickness_on else 0.5

        ax.plot([x_positions[from_id], x_positions[to_id]],
                [y_positions[from_id], y_positions[to_id]],
                color=color, linewidth=lw, solid_capstyle="round", zorder=1)

    # Draw nodes
    # Build index maps for input/output labeling
    input_nodes = sorted([n for n in nodes if n["type"] == 0], key=lambda n: n["id"])
    output_nodes = sorted([n for n in nodes if n["type"] == 2], key=lambda n: n["id"])

    for node in nodes:
        nid = node["id"]
        x = x_positions[nid]
        y = y_positions[nid]

        if node["type"] == 0:
            fc = "#2196F3"   # input
        elif node["type"] == 2:
            fc = "#4CAF50"   # output
        else:
            fc = "#FF9800"   # hidden

        circle = plt.Circle((x, y), 0.008,
                            facecolor=fc, edgecolor="white",
                            linewidth=0.5, zorder=3)
        ax.add_patch(circle)

        # Labels
        if node["type"] == 0:
            in_idx = next((i for i, n in enumerate(input_nodes) if n["id"] == nid), None)
            if in_idx is not None and in_idx < len(INPUT_LABELS):
                ax.text(x - 0.02, y, INPUT_LABELS[in_idx],
                        ha="right", va="center", fontsize=5, color="#666")
        elif node["type"] == 2:
            out_idx = next((i for i, n in enumerate(output_nodes) if n["id"] == nid), None)
            if out_idx is not None and out_idx < len(OUTPUT_LABELS):
                ax.text(x + 0.02, y, OUTPUT_LABELS[out_idx],
                        ha="left", va="center", fontsize=7, color="#333",
                        fontweight="bold")

    # Depth labels at top
    for depth in sorted_depths:
        x = depth_to_x[depth]
        n_count = len(depth_groups[depth])
        if depth == 0.0:
            label = f"Input\n{n_count}"
        elif depth == 1.0:
            label = f"Output\n{n_count}"
        else:
            label = f"d={depth:.2f}\n{n_count}"
        ax.text(x, 0.98, label, ha="center", va="top", fontsize=8,
                color="#333", fontweight="bold")

    # Info
    gen = data.get("generation", "?")
    rank = data.get("rank", "?")
    fitness = data.get("fitness", 0)
    n_hidden = sum(1 for n in nodes if n["type"] == 1)
    n_enabled = sum(1 for c in connections if c["enabled"])
    n_total = len(connections)
    n_params = n_enabled + sum(1 for n in nodes if n["type"] != 0)  # weights + biases

    info = (f"Generation {gen}  |  Rank #{rank}  |  "
            f"Fitness: {fitness:.2f}s  |  "
            f"Hidden: {n_hidden}  |  "
            f"Connections: {n_enabled}/{n_total}  |  "
            f"~{n_params} params")
    ax.set_title(info, fontsize=10, pad=12)
    _draw_controls(ax)
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.02, 1.05)
    ax.set_aspect("equal")
    ax.axis("off")


def _draw_controls(ax):
    """Draw control hints at bottom of network panel."""
    filter_labels = {"all": "All", "pos": "Positive only", "neg": "Negative only"}
    # We can't easily get the current filter_mode here, so this draws a static hint
    ax.text(0.5, 0.01,
            "[← → gen]  [↑ ↓ rank]  "
            "[R: cycle filter]  "
            "[T: toggle thickness]  [Q: quit]",
            ha="center", va="bottom", fontsize=7, color="#999",
            transform=ax.transAxes)


# ── Weight Distribution Histogram ────────────────────────────────────

def draw_histogram(ax, data: dict, weight_matrices: list[np.ndarray] | None = None):
    """Draw a histogram of all weights. Works with both formats."""
    ax.clear()

    if is_neat_format(data):
        # NEAT: collect weights from connection genes
        weights = [c["weight"] for c in data["connections"] if c["enabled"]]
        if not weights:
            ax.text(0.5, 0.5, "No enabled connections",
                    ha="center", va="center", fontsize=9, color="#999",
                    transform=ax.transAxes)
            ax.set_title("Weight Distribution", fontsize=9)
            return
        all_w = np.array(weights, dtype=np.float32)
    else:
        # Legacy
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


# ── Generation Summary Loading ───────────────────────────────────────

def load_generation_summary(gen_dir: str) -> dict | None:
    """Load generation_summary.json from a generation folder, or None."""
    path = os.path.join(gen_dir, "generation_summary.json")
    if not os.path.exists(path):
        return None
    with open(path, "r") as f:
        return json.load(f)


# ── Fitness Progression Chart ────────────────────────────────────────

def draw_fitness_progression(ax, all_gens: list[str]):
    """Plot best, average, median, and worst fitness across generations.
    Uses generation_summary.json when available. For fallback (top-N only data),
    only plots best and avg — median/worst would be misleading."""
    ax.clear()
    gens = []
    bests = []
    avgs = []

    # Separate tracks for summary-backed data (which has full-population stats)
    summary_gens = []
    medians = []
    worsts = []

    for gen_dir in all_gens:
        summary = load_generation_summary(gen_dir)
        if summary is not None:
            g = summary["generation"]
            gens.append(g)
            bests.append(summary["fitnessBest"])
            avgs.append(summary["fitnessAvg"])
            # Full-population stats — these are accurate
            summary_gens.append(g)
            medians.append(summary["fitnessMedian"])
            worsts.append(summary["fitnessWorst"])
        else:
            # Fallback: read from top-N network files
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
                # Avg of top-N only — label it accordingly
                avgs.append(sum(fitnesses) / len(fitnesses))
                # Do NOT compute median/worst from top-N — they'd be misleading

    if gens:
        # Fill between worst and best only where we have summary data
        if summary_gens:
            ax.fill_between(summary_gens, worsts, [bests[gens.index(g)] for g in summary_gens],
                            alpha=0.1, color="#4CAF50")

        ax.plot(gens, bests, "o-", color="#4CAF50", markersize=3,
                linewidth=1.5, label="Best")

        # Label avg based on whether we have any non-summary gens
        has_fallback = len(gens) > len(summary_gens)
        avg_label = "Avg (top-N pre-summary)" if has_fallback and not summary_gens else "Avg"
        ax.plot(gens, avgs, "s-", color="#FF9800", markersize=2,
                linewidth=1.2, alpha=0.8, label=avg_label)

        # Median and worst only for summary-backed generations
        if summary_gens:
            ax.plot(summary_gens, medians, "^--", color="#9C27B0", markersize=2,
                    linewidth=0.8, alpha=0.6, label="Median")
            ax.plot(summary_gens, worsts, "v--", color="#F44336", markersize=2,
                    linewidth=0.8, alpha=0.5, label="Worst")

        ax.legend(fontsize=6, ncol=2)

    ax.set_xlabel("Generation", fontsize=8)
    ax.set_ylabel("Fitness (s)", fontsize=8)
    ax.set_title("Fitness Progression", fontsize=9)
    ax.tick_params(labelsize=7)
    ax.grid(True, alpha=0.3)


# ── Generation Stats Panel ──────────────────────────────────────────

def draw_gen_stats(ax, all_gens: list[str], current_gen_idx: int):
    """Draw camping ratio and engagement ratio across generations."""
    ax.clear()

    gen_nums = []
    stationary_ratios = []
    engagement_ratios = []

    for gen_dir in all_gens:
        summary = load_generation_summary(gen_dir)
        if summary is None:
            continue
        gen_nums.append(summary["generation"])
        stationary_ratios.append(summary.get("stationaryRatio", 0) * 100)
        engagement_ratios.append(summary.get("avgEngagementRatio", 0) * 100)

    if not gen_nums:
        ax.text(0.5, 0.5, "No summary data yet\n(generation_summary.json)",
                ha="center", va="center", fontsize=9, color="#999",
                transform=ax.transAxes)
        ax.set_title("Engagement Stats", fontsize=9)
        return

    color1 = "#E91E63"
    color2 = "#4CAF50"

    ax.plot(gen_nums, stationary_ratios, "o-", color=color1, markersize=3,
            linewidth=1.2, label="Camp ratio %")
    ax.plot(gen_nums, engagement_ratios, "s-", color=color2, markersize=3,
            linewidth=1.2, label="Engagement %")
    ax.set_ylabel("Ratio %", fontsize=7)
    ax.tick_params(axis="y", labelsize=6)

    if 0 <= current_gen_idx < len(all_gens):
        folder_name = os.path.basename(all_gens[current_gen_idx])
        num = int(folder_name.split("_")[1])
        ax.axvline(num, color="#999", alpha=0.3, linewidth=2)

    ax.set_xlabel("Generation", fontsize=7)
    ax.set_title("Camping vs Engagement", fontsize=9)
    ax.tick_params(axis="x", labelsize=6)
    ax.grid(True, alpha=0.2)
    ax.legend(fontsize=5, loc="upper right")


# ── Complexity Growth Panel ─────────────────────────────────────────

def draw_complexity(ax, all_gens: list[str], current_gen_idx: int):
    """Draw network complexity growth (hidden nodes & connections) over generations."""
    ax.clear()

    gen_nums = []
    avg_hidden = []
    avg_conns = []

    for gen_dir in all_gens:
        summary = load_generation_summary(gen_dir)
        if summary is None:
            continue
        if "avgHiddenNodes" not in summary:
            continue
        gen_nums.append(summary["generation"])
        avg_hidden.append(summary.get("avgHiddenNodes", 0))
        avg_conns.append(summary.get("avgEnabledConnections", 0))

    if not gen_nums:
        ax.text(0.5, 0.5, "No complexity data yet\n(NEAT summaries needed)",
                ha="center", va="center", fontsize=9, color="#999",
                transform=ax.transAxes)
        ax.set_title("Network Complexity", fontsize=9)
        return

    color1 = "#FF9800"
    color2 = "#2196F3"

    ax.plot(gen_nums, avg_hidden, "o-", color=color1, markersize=3,
            linewidth=1.2, label="Avg hidden nodes")
    ax.set_ylabel("Hidden nodes", fontsize=7, color=color1)
    ax.tick_params(axis="y", labelcolor=color1, labelsize=6)

    ax2 = ax.twinx()
    ax2.plot(gen_nums, avg_conns, "s-", color=color2, markersize=3,
             linewidth=1.2, label="Avg connections")
    ax2.set_ylabel("Connections", fontsize=7, color=color2)
    ax2.tick_params(axis="y", labelcolor=color2, labelsize=6)

    if 0 <= current_gen_idx < len(all_gens):
        folder_name = os.path.basename(all_gens[current_gen_idx])
        num = int(folder_name.split("_")[1])
        ax.axvline(num, color="#999", alpha=0.3, linewidth=2)

    ax.set_xlabel("Generation", fontsize=7)
    ax.set_title("Network Complexity", fontsize=9)
    ax.tick_params(axis="x", labelsize=6)
    ax.grid(True, alpha=0.2)

    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, fontsize=5, loc="upper left")


# ── IO Averages ─────────────────────────────────────────────────────

def load_io_averages(gen_dir: str) -> dict | None:
    """Load io_averages.json from a generation folder, or None."""
    path = os.path.join(gen_dir, "io_averages.json")
    if not os.path.exists(path):
        return None
    with open(path, "r") as f:
        return json.load(f)


def draw_io_averages(ax, all_gens: list[str], current_gen_idx: int):
    """Draw IO average values across generations as line plots."""
    ax.clear()

    all_io = []
    gen_nums = []
    for gen_dir in all_gens:
        io = load_io_averages(gen_dir)
        if io is not None:
            all_io.append(io)
            gen_nums.append(io.get("generation", 0))

    if not all_io:
        ax.text(0.5, 0.5, "No IO data yet\n(io_averages.json)",
                ha="center", va="center", fontsize=9, color="#999",
                transform=ax.transAxes)
        ax.set_title("IO Averages", fontsize=9)
        return

    n_inputs = len(all_io[0]["avgInputs"])
    n_outputs = len(all_io[0]["avgOutputs"])

    key_input_indices = [0, 1, 2, 3, 4]
    key_input_labels = ["LOS", "Dist", "Dir X", "Dir Y", "Dir Z"]
    input_colors = ["#2196F3", "#FF9800", "#E91E63", "#9C27B0", "#00BCD4"]

    output_labels = OUTPUT_LABELS[:n_outputs]
    output_colors = ["#4CAF50", "#8BC34A", "#CDDC39"]

    for k, (idx, label) in enumerate(zip(key_input_indices, key_input_labels)):
        if idx < n_inputs:
            vals = [io["avgInputs"][idx] for io in all_io]
            ax.plot(gen_nums, vals, "-", color=input_colors[k % len(input_colors)],
                    linewidth=1.2, label=f"In: {label}", alpha=0.8, markersize=2)

    for k in range(min(n_outputs, len(output_labels))):
        vals = [io["avgOutputs"][k] for io in all_io]
        ax.plot(gen_nums, vals, "o-", color=output_colors[k % len(output_colors)],
                linewidth=1.5, label=f"Out: {output_labels[k]}",
                markersize=3, alpha=0.9)

    if 0 <= current_gen_idx < len(all_gens):
        folder_name = os.path.basename(all_gens[current_gen_idx])
        num = int(folder_name.split("_")[1])
        ax.axvline(num, color="#2196F3", alpha=0.3, linewidth=2)

    ax.set_xlabel("Generation", fontsize=7)
    ax.set_ylabel("Avg Value", fontsize=7)
    ax.set_title("IO Averages (key inputs + outputs)", fontsize=9)
    ax.tick_params(labelsize=6)
    ax.legend(fontsize=5, ncol=2, loc="upper right")
    ax.grid(True, alpha=0.3)


def draw_io_bar(ax, gen_dir: str):
    """Draw a bar chart of current generation's IO averages."""
    ax.clear()
    io = load_io_averages(gen_dir)
    if io is None:
        ax.text(0.5, 0.5, "No IO data for this gen",
                ha="center", va="center", fontsize=9, color="#999",
                transform=ax.transAxes)
        ax.set_title("IO Snapshot", fontsize=9)
        return

    avg_in = io["avgInputs"]
    avg_out = io["avgOutputs"]
    gen = io.get("generation", "?")

    n_in = len(avg_in)
    in_labels = INPUT_LABELS[:n_in] if n_in <= len(INPUT_LABELS) else [str(i) for i in range(n_in)]
    out_labels = OUTPUT_LABELS[:len(avg_out)]

    all_labels = in_labels + ["│"] + out_labels
    all_vals = list(avg_in) + [0] + list(avg_out)
    colors = (["#2196F3"] * n_in) + ["#fafafa"] + (["#4CAF50"] * len(avg_out))

    x = np.arange(len(all_vals))
    ax.bar(x, all_vals, color=colors, edgecolor="none", width=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(all_labels, fontsize=4, rotation=60, ha="right")
    ax.axhline(0, color="#333", linewidth=0.5)
    ax.set_title(f"IO Snapshot — Gen {gen} (blue=inputs, green=outputs)", fontsize=8)
    ax.tick_params(axis="y", labelsize=6)


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
        for i, g in enumerate(all_gens):
            folder_name = os.path.basename(g)
            num = int(folder_name.split("_")[1])
            if num == args.gen:
                gen_idx = i
                break

    rank_idx = args.rank
    filter_mode = "all"
    thickness_on = True

    # 7-panel layout: network on top, 6 panels on bottom
    fig = plt.figure(figsize=(20, 10), facecolor="#fafafa")
    fig.canvas.manager.set_window_title("Neuroevolution Visualizer")

    ax_net   = fig.add_axes([0.02, 0.42, 0.96, 0.56])       # network graph
    ax_hist  = fig.add_axes([0.03, 0.05, 0.14, 0.30])       # weight distribution
    ax_fit   = fig.add_axes([0.20, 0.05, 0.14, 0.30])       # fitness progression
    ax_stats = fig.add_axes([0.37, 0.05, 0.14, 0.30])       # camping & diversity
    ax_cplx  = fig.add_axes([0.54, 0.05, 0.14, 0.30])       # complexity growth
    ax_io    = fig.add_axes([0.71, 0.05, 0.14, 0.30])       # IO progression
    ax_bar   = fig.add_axes([0.88, 0.05, 0.10, 0.30])       # IO snapshot bar

    def refresh():
        nets = discover_networks(all_gens[gen_idx])
        if not nets:
            return

        r = min(rank_idx, len(nets) - 1)
        data = load_network(nets[r])

        if is_neat_format(data):
            draw_network_neat(ax_net, data, filter_mode, thickness_on)
            draw_histogram(ax_hist, data)
        else:
            W, b = reconstruct_weights(data)
            draw_network_legacy(ax_net, data, W, b, filter_mode, thickness_on)
            draw_histogram(ax_hist, data, W)

        draw_fitness_progression(ax_fit, all_gens)
        draw_gen_stats(ax_stats, all_gens, gen_idx)
        draw_complexity(ax_cplx, all_gens, gen_idx)
        draw_io_averages(ax_io, all_gens, gen_idx)
        draw_io_bar(ax_bar, all_gens[gen_idx])

        # Highlight current gen on fitness chart
        gen_num = data.get("generation", 0)
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