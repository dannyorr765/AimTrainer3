using System;
using System.Collections.Generic;
using System.Linq;

public class NodeGene
{
    public int id;
    public int type; // 0=Input, 1=Hidden, 2=Output
    public float bias;
    public float depth; // 0.0=input layer, 1.0=output layer, between=hidden

    public NodeGene(int id, int type, float bias, float depth)
    {
        this.id = id;
        this.type = type;
        this.bias = bias;
        this.depth = depth;
    }

    public NodeGene Clone()
    {
        return new NodeGene(id, type, bias, depth);
    }
}

public class ConnectionGene
{
    public int innovation;
    public int fromNode;
    public int toNode;
    public float weight;
    public bool enabled;

    public ConnectionGene(int innovation, int fromNode, int toNode, float weight, bool enabled)
    {
        this.innovation = innovation;
        this.fromNode = fromNode;
        this.toNode = toNode;
        this.weight = weight;
        this.enabled = enabled;
    }

    public ConnectionGene Clone()
    {
        return new ConnectionGene(innovation, fromNode, toNode, weight, enabled);
    }
}

public class NeuralNetwork
{
    private int inputCount;
    private int outputCount;
    private List<NodeGene> nodes;
    private List<ConnectionGene> connections;

    // Pre-cached forward pass structures
    private float[] nodeValues;
    private NodeGene[] evalOrder;
    private ConnectionGene[][] incomingByNode;
    private Dictionary<int, int> nodeIdToIndex;
    private bool dirty = true;

    // Global counters
    private static int nextInnovation = 0;
    private static int nextNodeId = 0;
    private static Dictionary<(int, int), int> innovationCache = new Dictionary<(int, int), int>();
    private static Random rng = new Random();

    /// <summary>
    /// Creates a minimal NEAT network with direct input→output connections.
    /// All networks share the same node IDs: inputs 0..(inputs-1), outputs inputs..(inputs+outputs-1).
    /// This is essential for NEAT crossover — innovation numbers only align when node IDs match.
    /// </summary>
    public NeuralNetwork(int inputs, int outputs)
    {
        inputCount = inputs;
        outputCount = outputs;
        nodes = new List<NodeGene>();
        connections = new List<ConnectionGene>();

        // Fixed node IDs shared across all networks in the population
        // Inputs: 0 to inputs-1, Outputs: inputs to inputs+outputs-1
        for (int i = 0; i < inputs; i++)
            nodes.Add(new NodeGene(i, 0, 0f, 0f));

        for (int i = 0; i < outputs; i++)
            nodes.Add(new NodeGene(inputs + i, 2, 0f, 1f));

        // Ensure nextNodeId is past all input/output IDs so structural
        // mutations allocate hidden node IDs that don't collide
        if (nextNodeId < inputs + outputs)
            nextNodeId = inputs + outputs;

        // Connect every input to every output with He-initialized weights
        // Innovation cache ensures all networks get the same innovation numbers
        // for the same structural connection (e.g. node 0→29 is always the same innovation)
        float heStd = (float)Math.Sqrt(2.0 / inputs);
        for (int i = 0; i < inputs; i++)
        {
            for (int o = 0; o < outputs; o++)
            {
                int fromId = i;
                int toId = inputs + o;
                int innov = GetInnovation(fromId, toId);
                float w = (float)Gaussian() * heStd;
                connections.Add(new ConnectionGene(innov, fromId, toId, w, true));
            }
        }

        dirty = true;
    }

    /// <summary>
    /// Private constructor for Clone/Crossover/deserialization.
    /// </summary>
    private NeuralNetwork(int inputs, int outputs, List<NodeGene> nodes, List<ConnectionGene> connections)
    {
        this.inputCount = inputs;
        this.outputCount = outputs;
        this.nodes = nodes;
        this.connections = connections;
        this.dirty = true;
    }

    // ─── Static counter management ───

    public static void ResetInnovationCache()
    {
        innovationCache.Clear();
    }

    public static (int nextInnov, int nextNode) GetCounters()
    {
        return (nextInnovation, nextNodeId);
    }

    public static void SetCounters(int innov, int node)
    {
        nextInnovation = innov;
        nextNodeId = node;
    }

    public static int GetInnovation(int fromNode, int toNode)
    {
        var key = (fromNode, toNode);
        if (innovationCache.TryGetValue(key, out int cached))
            return cached;
        int innov = nextInnovation++;
        innovationCache[key] = innov;
        return innov;
    }

    private static int AllocateNodeId()
    {
        return nextNodeId++;
    }

    // ─── Forward pass ───

    private void RebuildCache()
    {
        // Map node ids to dense indices
        nodeIdToIndex = new Dictionary<int, int>();
        for (int i = 0; i < nodes.Count; i++)
            nodeIdToIndex[nodes[i].id] = i;

        nodeValues = new float[nodes.Count];

        // Sort non-input nodes by depth for forward eval order
        var nonInputs = new List<NodeGene>();
        for (int i = 0; i < nodes.Count; i++)
        {
            if (nodes[i].type != 0) // not input
                nonInputs.Add(nodes[i]);
        }
        nonInputs.Sort((a, b) => a.depth.CompareTo(b.depth));
        evalOrder = nonInputs.ToArray();

        // Build incoming connections per node (only enabled, only if both endpoints exist)
        var incoming = new Dictionary<int, List<ConnectionGene>>();
        for (int i = 0; i < connections.Count; i++)
        {
            var c = connections[i];
            if (!c.enabled) continue;
            if (!nodeIdToIndex.ContainsKey(c.fromNode) || !nodeIdToIndex.ContainsKey(c.toNode)) continue;
            if (!incoming.ContainsKey(c.toNode))
                incoming[c.toNode] = new List<ConnectionGene>();
            incoming[c.toNode].Add(c);
        }

        incomingByNode = new ConnectionGene[nodes.Count][];
        for (int i = 0; i < nodes.Count; i++)
        {
            int nodeId = nodes[i].id;
            if (incoming.ContainsKey(nodeId))
                incomingByNode[i] = incoming[nodeId].ToArray();
            else
                incomingByNode[i] = Array.Empty<ConnectionGene>();
        }

        dirty = false;
    }

    public float[] Forward(float[] inputs)
    {
        if (inputs.Length != inputCount)
            throw new ArgumentException($"Expected {inputCount} inputs, got {inputs.Length}");

        if (dirty) RebuildCache();

        // Clear all node values
        Array.Clear(nodeValues, 0, nodeValues.Length);

        // Set input values
        for (int i = 0; i < inputCount; i++)
            nodeValues[nodeIdToIndex[nodes[i].id]] = inputs[i];

        // Evaluate in depth order
        for (int i = 0; i < evalOrder.Length; i++)
        {
            var node = evalOrder[i];
            int idx = nodeIdToIndex[node.id];
            float sum = node.bias;

            var inc = incomingByNode[idx];
            for (int j = 0; j < inc.Length; j++)
            {
                int fromIdx = nodeIdToIndex[inc[j].fromNode];
                sum += nodeValues[fromIdx] * inc[j].weight;
            }

            nodeValues[idx] = (float)Math.Tanh(sum);
        }

        // Collect outputs
        float[] result = new float[outputCount];
        for (int i = 0; i < outputCount; i++)
            result[i] = nodeValues[nodeIdToIndex[nodes[inputCount + i].id]];

        return result;
    }

    // ─── Mutation ───

    public void Mutate(float weightRate, float weightMagnitude, float addNodeRate = 0f,
        float addConnectionRate = 0f, float weightResetRate = 0.1f, float pruneRate = 0f, float pruneThreshold = 0.05f)
    {
        float heStd = (float)Math.Sqrt(2.0 / inputCount);

        // Weight perturbation with chance of full reset
        for (int i = 0; i < connections.Count; i++)
        {
            double roll = rng.NextDouble();
            if (roll < weightResetRate)
            {
                // Full weight reset — reroll from scratch to escape local optima
                connections[i].weight = (float)Gaussian() * heStd;
                dirty = true;
            }
            else if (roll < weightResetRate + weightRate)
            {
                // Normal perturbation
                connections[i].weight += (float)Gaussian() * weightMagnitude;
                dirty = true;
            }
        }

        // Bias perturbation with chance of reset
        for (int i = 0; i < nodes.Count; i++)
        {
            if (nodes[i].type == 0) continue; // don't mutate input biases
            double roll = rng.NextDouble();
            if (roll < weightResetRate)
            {
                nodes[i].bias = (float)Gaussian() * 0.1f;
                dirty = true;
            }
            else if (roll < weightResetRate + weightRate)
            {
                nodes[i].bias += (float)Gaussian() * weightMagnitude;
                dirty = true;
            }
        }

        // Pruning — disable weak connections (only on networks with hidden nodes to avoid gutting minimal topology)
        if (pruneRate > 0 && GetHiddenNodeCount() > 0)
        {
            for (int i = 0; i < connections.Count; i++)
            {
                if (connections[i].enabled && Math.Abs(connections[i].weight) < pruneThreshold)
                {
                    if (rng.NextDouble() < pruneRate)
                    {
                        connections[i].enabled = false;
                        dirty = true;
                    }
                }
            }

            // Prune dead hidden nodes — no enabled inputs OR no enabled outputs means the node does nothing
            PruneDeadNodes();
        }

        // Structural mutations
        if (rng.NextDouble() < addNodeRate)
            AddNodeMutation();

        if (rng.NextDouble() < addConnectionRate)
            AddConnectionMutation();
    }

    private void AddNodeMutation()
    {
        // Pick a random enabled connection
        var enabled = new List<int>();
        for (int i = 0; i < connections.Count; i++)
        {
            if (connections[i].enabled) enabled.Add(i);
        }
        if (enabled.Count == 0) return;

        int idx = enabled[rng.Next(enabled.Count)];
        var conn = connections[idx];

        // Disable the original connection
        conn.enabled = false;

        // Find from/to node depths
        float fromDepth = 0f, toDepth = 1f;
        for (int i = 0; i < nodes.Count; i++)
        {
            if (nodes[i].id == conn.fromNode) fromDepth = nodes[i].depth;
            if (nodes[i].id == conn.toNode) toDepth = nodes[i].depth;
        }

        // Create new hidden node at midpoint depth
        int newNodeId = AllocateNodeId();
        float newDepth = (fromDepth + toDepth) / 2f;
        nodes.Add(new NodeGene(newNodeId, 1, 0f, newDepth));

        // Add from→new with weight 1.0 (preserves signal)
        int innov1 = GetInnovation(conn.fromNode, newNodeId);
        connections.Add(new ConnectionGene(innov1, conn.fromNode, newNodeId, 1f, true));

        // Add new→to with original weight (preserves signal)
        int innov2 = GetInnovation(newNodeId, conn.toNode);
        connections.Add(new ConnectionGene(innov2, newNodeId, conn.toNode, conn.weight, true));

        dirty = true;
    }

    private void AddConnectionMutation()
    {
        // Try up to 30 times to find two unconnected nodes where from.depth < to.depth
        var existingConns = new HashSet<(int, int)>();
        for (int i = 0; i < connections.Count; i++)
            existingConns.Add((connections[i].fromNode, connections[i].toNode));

        for (int attempt = 0; attempt < 30; attempt++)
        {
            var from = nodes[rng.Next(nodes.Count)];
            var to = nodes[rng.Next(nodes.Count)];

            if (from.id == to.id) continue;
            if (from.depth >= to.depth) continue;
            if (existingConns.Contains((from.id, to.id))) continue;

            // He init based on fan-in of target node
            int fanIn = 0;
            for (int i = 0; i < connections.Count; i++)
            {
                if (connections[i].toNode == to.id && connections[i].enabled) fanIn++;
            }
            float heStd = (float)Math.Sqrt(2.0 / Math.Max(fanIn, 1));
            float w = (float)Gaussian() * heStd;

            int innov = GetInnovation(from.id, to.id);
            connections.Add(new ConnectionGene(innov, from.id, to.id, w, true));
            dirty = true;
            return;
        }
    }

    private void PruneDeadNodes()
    {
        // Find hidden nodes with no enabled inbound OR no enabled outbound connections
        var deadNodes = new HashSet<int>();
        for (int i = 0; i < nodes.Count; i++)
        {
            if (nodes[i].type != 1) continue; // only prune hidden nodes

            int nodeId = nodes[i].id;
            bool hasEnabledIn = false;
            bool hasEnabledOut = false;

            for (int j = 0; j < connections.Count; j++)
            {
                if (!connections[j].enabled) continue;
                if (connections[j].toNode == nodeId) hasEnabledIn = true;
                if (connections[j].fromNode == nodeId) hasEnabledOut = true;
                if (hasEnabledIn && hasEnabledOut) break;
            }

            if (!hasEnabledIn || !hasEnabledOut)
                deadNodes.Add(nodeId);
        }

        if (deadNodes.Count == 0) return;

        // Disable all connections to/from dead nodes
        for (int i = 0; i < connections.Count; i++)
        {
            if (deadNodes.Contains(connections[i].fromNode) || deadNodes.Contains(connections[i].toNode))
                connections[i].enabled = false;
        }

        // Remove the dead nodes from the genome
        nodes.RemoveAll(n => deadNodes.Contains(n.id));
        dirty = true;
    }

    // ─── Crossover ───

    public static NeuralNetwork Crossover(NeuralNetwork a, NeuralNetwork b)
    {
        // Build innovation→connection maps
        var aConns = new Dictionary<int, ConnectionGene>();
        for (int i = 0; i < a.connections.Count; i++)
            aConns[a.connections[i].innovation] = a.connections[i];

        var bConns = new Dictionary<int, ConnectionGene>();
        for (int i = 0; i < b.connections.Count; i++)
            bConns[b.connections[i].innovation] = b.connections[i];

        // Collect all innovation numbers
        var allInnovations = new HashSet<int>();
        foreach (var k in aConns.Keys) allInnovations.Add(k);
        foreach (var k in bConns.Keys) allInnovations.Add(k);

        var childConns = new List<ConnectionGene>();
        var neededNodeIds = new HashSet<int>();

        foreach (int innov in allInnovations)
        {
            bool inA = aConns.ContainsKey(innov);
            bool inB = bConns.ContainsKey(innov);
            ConnectionGene chosen;

            if (inA && inB)
            {
                // Matching gene — random pick
                chosen = (rng.NextDouble() < 0.5) ? aConns[innov].Clone() : bConns[innov].Clone();
                // Re-enable with 25% chance if disabled in either parent
                if (!aConns[innov].enabled || !bConns[innov].enabled)
                {
                    if (rng.NextDouble() < 0.25) chosen.enabled = true;
                }
            }
            else if (inA)
            {
                // Disjoint/excess from A — include (equal fitness variant)
                chosen = aConns[innov].Clone();
            }
            else
            {
                // Disjoint/excess from B — include (equal fitness variant)
                chosen = bConns[innov].Clone();
            }

            childConns.Add(chosen);
            neededNodeIds.Add(chosen.fromNode);
            neededNodeIds.Add(chosen.toNode);
        }

        // Build node set from both parents
        var aNodes = new Dictionary<int, NodeGene>();
        for (int i = 0; i < a.nodes.Count; i++)
            aNodes[a.nodes[i].id] = a.nodes[i];

        var bNodes = new Dictionary<int, NodeGene>();
        for (int i = 0; i < b.nodes.Count; i++)
            bNodes[b.nodes[i].id] = b.nodes[i];

        var childNodes = new List<NodeGene>();
        foreach (int nodeId in neededNodeIds)
        {
            bool inA = aNodes.ContainsKey(nodeId);
            bool inB = bNodes.ContainsKey(nodeId);

            if (inA && inB)
            {
                // Shared node — random pick
                childNodes.Add((rng.NextDouble() < 0.5) ? aNodes[nodeId].Clone() : bNodes[nodeId].Clone());
            }
            else if (inA)
            {
                childNodes.Add(aNodes[nodeId].Clone());
            }
            else if (inB)
            {
                childNodes.Add(bNodes[nodeId].Clone());
            }
        }

        // Ensure all input/output nodes are present even if no connections reference them
        for (int i = 0; i < a.nodes.Count; i++)
        {
            var n = a.nodes[i];
            if (n.type != 1 && !neededNodeIds.Contains(n.id)) // input or output
            {
                childNodes.Add(n.Clone());
            }
        }

        // Remove connections that reference nodes neither parent still has (pruned nodes)
        var childNodeIds = new HashSet<int>();
        for (int i = 0; i < childNodes.Count; i++)
            childNodeIds.Add(childNodes[i].id);

        childConns.RemoveAll(c => !childNodeIds.Contains(c.fromNode) || !childNodeIds.Contains(c.toNode));

        // Sort nodes: inputs first, then by depth, then outputs
        childNodes.Sort((x, y) =>
        {
            if (x.type != y.type)
            {
                // Input=0, Hidden=1, Output=2
                return x.type.CompareTo(y.type);
            }
            return x.depth.CompareTo(y.depth);
        });

        return new NeuralNetwork(a.inputCount, a.outputCount, childNodes, childConns);
    }

    // ─── Clone ───

    public NeuralNetwork Clone()
    {
        var clonedNodes = new List<NodeGene>(nodes.Count);
        for (int i = 0; i < nodes.Count; i++)
            clonedNodes.Add(nodes[i].Clone());

        var clonedConns = new List<ConnectionGene>(connections.Count);
        for (int i = 0; i < connections.Count; i++)
            clonedConns.Add(connections[i].Clone());

        return new NeuralNetwork(inputCount, outputCount, clonedNodes, clonedConns);
    }

    // ─── Info ───

    public int GetHiddenNodeCount()
    {
        int count = 0;
        for (int i = 0; i < nodes.Count; i++)
            if (nodes[i].type == 1) count++;
        return count;
    }

    public int GetEnabledConnectionCount()
    {
        int count = 0;
        for (int i = 0; i < connections.Count; i++)
            if (connections[i].enabled) count++;
        return count;
    }

    public int GetTotalConnectionCount()
    {
        return connections.Count;
    }

    public int GetInputCount() => inputCount;
    public int GetOutputCount() => outputCount;

    // ─── Serialization access ───

    public List<NodeGene> GetNodes() => nodes;
    public List<ConnectionGene> GetConnections() => connections;

    /// <summary>
    /// Reconstruct a network from serialized data.
    /// </summary>
    public static NeuralNetwork FromData(int inputs, int outputs, List<NodeGene> nodes, List<ConnectionGene> connections)
    {
        return new NeuralNetwork(inputs, outputs, nodes, connections);
    }

    // ─── Utility ───

    private static float Gaussian()
    {
        double u1 = 1.0 - rng.NextDouble();
        double u2 = 1.0 - rng.NextDouble();
        return (float)(Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Sin(2.0 * Math.PI * u2));
    }
}