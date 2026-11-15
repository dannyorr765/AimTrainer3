using UnityEngine;
using System.Collections.Generic;
using System.Linq;
using System.IO;

/// <summary>
/// Spawns a population of bot prefabs, tracks their survival,
/// and evolves the neural networks between generations.
/// Position this object where you want bots to spawn around.
///
/// Fitness formula (engagement-based):
///   engagementRatio = timeInRange / survivalTime
///   fitness = survivalTime * (survivalWeight + engagementRatio * engagementWeight)
///           - stationaryTime * campingPenaltyRate
///
/// Rewards bots that survive long AND stay near the player.
/// Penalizes camping. Cover usage emerges naturally — bots that
/// break LOS while actively moving survive longer.
/// </summary>
public class BotSpawner : MonoBehaviour
{
    [Header("References")]
    [SerializeField] private GameObject botPrefab;
    [SerializeField] private Transform player;

    [Header("Spawning")]
    [SerializeField] private int botsPerGeneration = 30;
    [SerializeField] private float spawnRadius = 15f;
    [SerializeField] private float respawnDelay = 2f;

    [Header("Evolution")]
    [SerializeField] private int eliteCount = 5;
    [SerializeField] private float mutationRate = 0.4f;
    [SerializeField] private float mutationMagnitude = 0.3f;

    [Header("Fitness")]
    [SerializeField] private float survivalWeight = 0.3f;
    [SerializeField] private float engagementWeight = 1.0f;
    [SerializeField] private float campingPenaltyRate = 0.5f;

    [Header("Structural Mutation (NEAT)")]
    [SerializeField] private float addNodeRate = 0.03f;
    [SerializeField] private float addConnectionRate = 0.05f;
    [SerializeField] private float weightResetRate = 0.1f;

    [Header("Pruning")]
    [SerializeField] private float pruneRate = 0.15f;
    [SerializeField] private float pruneThreshold = 0.05f;

    [Header("Saving")]
    [SerializeField] private int saveTopN = 5;
    [SerializeField] private string saveFolder = "SavedNetworks";

    [Header("Status")]
    [SerializeField] private int generation = 1;
    [SerializeField] private float bestFitnessEver = 0f;
    [SerializeField] private int botsAlive;

    private List<NeuralNetwork> population;
    private List<BotAgent> activeBots = new List<BotAgent>();
    private NeuralNetwork bestNetwork;
    private string savePath;
    private int inputSize;
    private int outputSize;
    private bool shuttingDown = false;

    void Start()
    {
        // Save path lives next to Assets/ in the project root
        savePath = Path.Combine(Application.dataPath, "..", saveFolder);
        Debug.Log($"[Evolution] Networks will be saved to: {Path.GetFullPath(savePath)}");

        // Read input/output dimensions from the prefab
        var template = botPrefab.GetComponent<BotAgent>();
        inputSize = template.InputSize;
        outputSize = template.OutputSize;

        Debug.Log($"[Evolution] NEAT — Inputs: {inputSize}, Outputs: {outputSize}, Pop: {botsPerGeneration}");

        // Try to resume from saved networks
        if (TryLoadLatestGeneration())
        {
            Debug.Log($"[Evolution] Resumed from generation {generation - 1}, starting gen {generation}");
        }
        else
        {
            population = new List<NeuralNetwork>();
            for (int i = 0; i < botsPerGeneration; i++)
                population.Add(new NeuralNetwork(inputSize, outputSize));
            Debug.Log("[Evolution] No saved networks found, starting fresh");
        }

        SpawnGeneration();
    }

    void OnApplicationQuit()
    {
        shuttingDown = true;
        CancelInvoke();
    }

    void OnDestroy()
    {
        shuttingDown = true;
        CancelInvoke();
    }

    private void SpawnGeneration()
    {
        // Clean up stragglers from previous generation
        foreach (var bot in activeBots)
            if (bot != null) Destroy(bot.gameObject);
        activeBots.Clear();

        botsAlive = botsPerGeneration;

        for (int i = 0; i < botsPerGeneration; i++)
        {
            Vector2 offset = Random.insideUnitCircle * spawnRadius;
            Vector3 pos = transform.position + new Vector3(offset.x, 0f, offset.y);

            GameObject obj = Instantiate(botPrefab, pos, Quaternion.identity);
            BotAgent agent = obj.GetComponent<BotAgent>();
            agent.Initialize(population[i], this, player);
            activeBots.Add(agent);
        }

        Debug.Log($"[Gen {generation}] Spawned {botsPerGeneration} bots");
    }

    /// <summary>
    /// Called by each BotAgent when it's killed.
    /// Fitness = survivalTime * (survivalWeight + engagementRatio * engagementWeight)
    ///         - stationaryTime * campingPenaltyRate
    /// </summary>
    public void OnBotKilled(BotAgent bot, float survivalTime)
    {
        if (shuttingDown) return;

        // Engagement ratio: what fraction of its life was spent near the player?
        float engagementRatio = survivalTime > 0.1f
            ? bot.TimeInRange / survivalTime : 0f;

        float fitness = survivalTime * (survivalWeight + engagementRatio * engagementWeight)
                      - bot.StationaryTime * campingPenaltyRate;
        fitness = Mathf.Max(0f, fitness);

        bot.SetFitness(fitness, engagementRatio);
        botsAlive--;

        if (botsAlive <= 0)
        {
            float[] fitnesses = new float[activeBots.Count];
            for (int i = 0; i < activeBots.Count; i++)
                fitnesses[i] = activeBots[i].Fitness;

            Evolve(fitnesses);
            Invoke(nameof(SpawnGeneration), respawnDelay);
        }
    }

    private void Evolve(float[] fitnesses)
    {
        var ranked = Enumerable.Range(0, fitnesses.Length)
            .OrderByDescending(i => fitnesses[i])
            .ToList();

        float genBest = fitnesses[ranked[0]];
        float genAvg = fitnesses.Average();

        if (genBest > bestFitnessEver)
        {
            bestFitnessEver = genBest;
            bestNetwork = population[ranked[0]].Clone();
        }

        // Complexity stats from bots
        float avgHidden = 0f, avgConns = 0f, avgOutputMag = 0f;
        int maxHidden = 0, maxConns = 0, zeroFitnessCount = 0;
        foreach (var bot in activeBots)
        {
            if (bot == null) continue;
            int h = bot.HiddenNodeCount;
            int c = bot.EnabledConnectionCount;
            avgHidden += h;
            avgConns += c;
            avgOutputMag += bot.AverageOutputMagnitude;
            if (h > maxHidden) maxHidden = h;
            if (c > maxConns) maxConns = c;
            if (bot.Fitness <= 0f) zeroFitnessCount++;
        }
        if (activeBots.Count > 0)
        {
            avgHidden /= activeBots.Count;
            avgConns /= activeBots.Count;
            avgOutputMag /= activeBots.Count;
        }

        // Engagement stats
        float avgEngagement = activeBots.Where(b => b != null).Average(b => b.EngagementRatio);
        float avgStationaryRatio = activeBots.Where(b => b != null)
            .Average(b => b.SurvivalTime > 0 ? b.StationaryTime / b.SurvivalTime : 0f);

        Debug.Log($"[Gen {generation}] Fitness — Best: {genBest:F2} | Avg: {genAvg:F2} | All-time: {bestFitnessEver:F2} | Zero-fitness: {zeroFitnessCount}/{fitnesses.Length}");
        Debug.Log($"[Gen {generation}] Engagement — Avg ratio: {avgEngagement:F2} | Avg stationary ratio: {avgStationaryRatio:F2} | Avg output mag: {avgOutputMag:F3}");

        // Save before building new population
        SaveTopNetworks(fitnesses, ranked);
        SaveIOAverages();
        SaveGenerationSummary(fitnesses, ranked, avgHidden, avgConns, maxHidden, maxConns);

        // Reset innovation cache for this generation's breeding
        NeuralNetwork.ResetInnovationCache();

        var newPop = new List<NeuralNetwork>();

        // Elitism — fitness-proportionate selection (no hard cutoff)
        var eliteIndices = WeightedSampleWithoutReplacement(fitnesses, eliteCount);
        foreach (int idx in eliteIndices)
            newPop.Add(population[idx].Clone());

        // Fill the rest via tournament selection → crossover → mutation
        while (newPop.Count < botsPerGeneration)
        {
            var parentA = TournamentSelect(fitnesses);
            var parentB = TournamentSelect(fitnesses);
            var child = NeuralNetwork.Crossover(parentA, parentB);
            child.Mutate(mutationRate, mutationMagnitude, addNodeRate, addConnectionRate, weightResetRate, pruneRate, pruneThreshold);
            newPop.Add(child);
        }

        population = newPop;
        generation++;
    }

    /// <summary>
    /// Selects `count` unique indices, weighted by fitness.
    /// Higher fitness = higher chance, but nothing is guaranteed.
    /// </summary>
    private List<int> WeightedSampleWithoutReplacement(float[] fitnesses, int count)
    {
        // Shift so the worst bot still has a small positive weight
        float minFit = fitnesses.Min();
        float shift = (minFit < 0f) ? -minFit + 0.1f : 0.1f;

        float[] weights = new float[fitnesses.Length];
        for (int i = 0; i < fitnesses.Length; i++)
            weights[i] = fitnesses[i] + shift;

        var selected = new List<int>();
        bool[] taken = new bool[fitnesses.Length];

        for (int s = 0; s < count && s < fitnesses.Length; s++)
        {
            // Sum available weights
            float totalWeight = 0f;
            for (int i = 0; i < weights.Length; i++)
                if (!taken[i]) totalWeight += weights[i];

            // Roulette spin
            float spin = Random.Range(0f, totalWeight);
            float cumulative = 0f;
            int chosen = -1;

            for (int i = 0; i < weights.Length; i++)
            {
                if (taken[i]) continue;
                cumulative += weights[i];
                if (spin <= cumulative)
                {
                    chosen = i;
                    break;
                }
            }

            // Fallback (float precision edge case)
            if (chosen == -1)
            {
                for (int i = weights.Length - 1; i >= 0; i--)
                    if (!taken[i]) { chosen = i; break; }
            }

            selected.Add(chosen);
            taken[chosen] = true;
        }

        return selected;
    }

    private NeuralNetwork TournamentSelect(float[] fitnesses)
    {
        int best = Random.Range(0, fitnesses.Length);
        for (int i = 0; i < 3; i++)
        {
            int candidate = Random.Range(0, fitnesses.Length);
            if (fitnesses[candidate] > fitnesses[best])
                best = candidate;
        }
        return population[best];
    }

    // --- Saving ---

    private void SaveTopNetworks(float[] fitnesses, List<int> ranked)
    {
        string genDir = Path.Combine(savePath, $"Gen_{generation:D4}");
        Directory.CreateDirectory(genDir);

        int count = Mathf.Min(saveTopN, ranked.Count);
        for (int rank = 0; rank < count; rank++)
        {
            int idx = ranked[rank];
            var net = population[idx];

            // Serialize nodes
            var nodes = net.GetNodes();
            var nodeDataList = new NodeData[nodes.Count];
            for (int i = 0; i < nodes.Count; i++)
            {
                nodeDataList[i] = new NodeData
                {
                    id = nodes[i].id,
                    type = nodes[i].type,
                    bias = nodes[i].bias,
                    depth = nodes[i].depth
                };
            }

            // Serialize connections
            var conns = net.GetConnections();
            var connDataList = new ConnData[conns.Count];
            for (int i = 0; i < conns.Count; i++)
            {
                connDataList[i] = new ConnData
                {
                    innovation = conns[i].innovation,
                    fromNode = conns[i].fromNode,
                    toNode = conns[i].toNode,
                    weight = conns[i].weight,
                    enabled = conns[i].enabled
                };
            }

            var data = new SaveData
            {
                inputs = net.GetInputCount(),
                outputs = net.GetOutputCount(),
                nodes = nodeDataList,
                connections = connDataList,
                generation = generation,
                fitness = fitnesses[idx],
                rank = rank
            };

            string json = JsonUtility.ToJson(data, true);
            string path = Path.Combine(genDir, $"network_{rank}.json");
            File.WriteAllText(path, json);
        }

        // Save global counters so resuming preserves innovation/node IDs
        var (nextInnov, nextNode) = NeuralNetwork.GetCounters();
        var globalState = new GlobalState { nextInnovation = nextInnov, nextNodeId = nextNode };
        string globalJson = JsonUtility.ToJson(globalState, true);
        File.WriteAllText(Path.Combine(genDir, "global_state.json"), globalJson);

        Debug.Log($"[Gen {generation}] Saved top {count} networks to {genDir}");
    }

    // --- Resume ---

    /// <summary>
    /// Scans the save folder for the highest generation, loads its networks,
    /// and breeds a full population from them. Returns false if nothing to load.
    /// </summary>
    private bool TryLoadLatestGeneration()
    {
        if (!Directory.Exists(savePath)) return false;

        // Find all Gen_XXXX folders and pick the highest number
        int latestGen = -1;
        string latestDir = null;

        foreach (string dir in Directory.GetDirectories(savePath, "Gen_*"))
        {
            string folderName = Path.GetFileName(dir);
            if (folderName.Length > 4 && int.TryParse(folderName.Substring(4), out int num))
            {
                if (num > latestGen)
                {
                    latestGen = num;
                    latestDir = dir;
                }
            }
        }

        if (latestDir == null) return false;

        // Restore global counters if available
        string globalPath = Path.Combine(latestDir, "global_state.json");
        if (File.Exists(globalPath))
        {
            try
            {
                string globalJson = File.ReadAllText(globalPath);
                var gs = JsonUtility.FromJson<GlobalState>(globalJson);
                NeuralNetwork.SetCounters(gs.nextInnovation, gs.nextNodeId);
                Debug.Log($"[Evolution] Restored global state: nextInnov={gs.nextInnovation}, nextNode={gs.nextNodeId}");
            }
            catch (System.Exception e)
            {
                Debug.LogWarning($"[Evolution] Failed to load global_state.json: {e.Message}");
            }
        }

        // Load all network files from that generation
        var loaded = new List<NeuralNetwork>();
        float loadedBestFitness = 0f;

        string[] files = Directory.GetFiles(latestDir, "network_*.json");
        System.Array.Sort(files);

        foreach (string file in files)
        {
            try
            {
                string json = File.ReadAllText(file);

                // Detect format: old (has "topology") vs new (has "nodes")
                if (json.Contains("\"topology\""))
                {
                    Debug.LogWarning($"[Evolution] Skipping old-format network: {file} (incompatible with NEAT)");
                    continue;
                }

                var data = JsonUtility.FromJson<SaveData>(json);

                // Reconstruct nodes
                var nodeList = new List<NodeGene>();
                for (int i = 0; i < data.nodes.Length; i++)
                {
                    var nd = data.nodes[i];
                    nodeList.Add(new NodeGene(nd.id, nd.type, nd.bias, nd.depth));
                }

                // Reconstruct connections
                var connList = new List<ConnectionGene>();
                for (int i = 0; i < data.connections.Length; i++)
                {
                    var cd = data.connections[i];
                    connList.Add(new ConnectionGene(cd.innovation, cd.fromNode, cd.toNode, cd.weight, cd.enabled));
                }

                var net = NeuralNetwork.FromData(data.inputs, data.outputs, nodeList, connList);
                loaded.Add(net);

                if (data.fitness > loadedBestFitness)
                {
                    loadedBestFitness = data.fitness;
                    bestNetwork = net.Clone();
                }
            }
            catch (System.Exception e)
            {
                Debug.LogWarning($"[Evolution] Failed to load {file}: {e.Message}");
            }
        }

        if (loaded.Count == 0)
        {
            Debug.LogWarning("[Evolution] No compatible networks found. Starting fresh.");
            return false;
        }

        // Set generation to one past the loaded one
        generation = latestGen + 1;
        bestFitnessEver = loadedBestFitness;

        // Reset innovation cache for breeding
        NeuralNetwork.ResetInnovationCache();

        // Build population: loaded elites first, then breed from them
        population = new List<NeuralNetwork>();

        foreach (var net in loaded)
            population.Add(net.Clone());

        // Fill remaining slots by crossover + mutation of loaded networks
        while (population.Count < botsPerGeneration)
        {
            var parentA = loaded[Random.Range(0, loaded.Count)];
            var parentB = loaded[Random.Range(0, loaded.Count)];
            var child = NeuralNetwork.Crossover(parentA, parentB);
            child.Mutate(mutationRate, mutationMagnitude, addNodeRate, addConnectionRate, weightResetRate, pruneRate, pruneThreshold);
            population.Add(child);
        }

        Debug.Log($"[Evolution] Loaded {loaded.Count} networks from Gen {latestGen} (best fitness: {loadedBestFitness:F2})");
        return true;
    }

    // --- IO Averages ---

    private void SaveIOAverages()
    {
        if (activeBots.Count == 0) return;

        int inSize = activeBots[0].InputSize;
        int outSize = activeBots[0].OutputSize;

        float[] avgInputs = new float[inSize];
        float[] avgOutputs = new float[outSize];
        int botCount = 0;

        foreach (var bot in activeBots)
        {
            if (bot == null) continue;
            float[] botInputs = bot.GetAverageInputs();
            float[] botOutputs = bot.GetAverageOutputs();
            for (int i = 0; i < inSize; i++)
                avgInputs[i] += botInputs[i];
            for (int i = 0; i < outSize; i++)
                avgOutputs[i] += botOutputs[i];
            botCount++;
        }

        if (botCount == 0) return;

        for (int i = 0; i < inSize; i++)
            avgInputs[i] /= botCount;
        for (int i = 0; i < outSize; i++)
            avgOutputs[i] /= botCount;

        var ioData = new IOAverageData
        {
            generation = generation,
            avgInputs = avgInputs,
            avgOutputs = avgOutputs,
            botCount = botCount
        };

        string genDir = Path.Combine(savePath, $"Gen_{generation:D4}");
        Directory.CreateDirectory(genDir);
        string json = JsonUtility.ToJson(ioData, true);
        File.WriteAllText(Path.Combine(genDir, "io_averages.json"), json);

        string inputStr = string.Join(", ", avgInputs.Select(v => v.ToString("F3")));
        string outputStr = string.Join(", ", avgOutputs.Select(v => v.ToString("F3")));
        Debug.Log($"[Gen {generation}] IO Averages — Inputs: [{inputStr}]");
        Debug.Log($"[Gen {generation}] IO Averages — Outputs: [{outputStr}]");
    }

    // --- Generation Summary ---

    private void SaveGenerationSummary(float[] fitnesses, List<int> ranked,
        float avgHidden, float avgConns, int maxHidden, int maxConns)
    {
        // Fitness stats
        float best = fitnesses[ranked[0]];
        float worst = fitnesses[ranked[ranked.Count - 1]];
        float avg = fitnesses.Average();
        float median = fitnesses.Length % 2 == 0
            ? (fitnesses.OrderBy(f => f).ElementAt(fitnesses.Length / 2 - 1)
             + fitnesses.OrderBy(f => f).ElementAt(fitnesses.Length / 2)) / 2f
            : fitnesses.OrderBy(f => f).ElementAt(fitnesses.Length / 2);

        // Standard deviation
        float sumSqDiff = 0f;
        for (int i = 0; i < fitnesses.Length; i++)
            sumSqDiff += (fitnesses[i] - avg) * (fitnesses[i] - avg);
        float stdDev = Mathf.Sqrt(sumSqDiff / fitnesses.Length);

        // Per-bot stats
        float avgSurvival = 0f, avgStationary = 0f, maxSurvival = 0f;
        float avgTimeInRange = 0f, avgEngagement = 0f;
        float avgOutputMag = 0f;
        int zeroFitnessCount = 0;
        int botCount = 0;

        foreach (var bot in activeBots)
        {
            if (bot == null) continue;
            avgSurvival += bot.SurvivalTime;
            avgStationary += bot.StationaryTime;
            avgTimeInRange += bot.TimeInRange;
            avgEngagement += bot.EngagementRatio;
            avgOutputMag += bot.AverageOutputMagnitude;
            if (bot.SurvivalTime > maxSurvival)
                maxSurvival = bot.SurvivalTime;
            if (bot.Fitness <= 0f)
                zeroFitnessCount++;
            botCount++;
        }
        if (botCount > 0)
        {
            avgSurvival /= botCount;
            avgStationary /= botCount;
            avgTimeInRange /= botCount;
            avgEngagement /= botCount;
            avgOutputMag /= botCount;
        }

        var summary = new GenerationSummary
        {
            generation = generation,
            populationSize = fitnesses.Length,
            fitnessBest = best,
            fitnessWorst = worst,
            fitnessAvg = avg,
            fitnessMedian = median,
            fitnessStdDev = stdDev,
            avgSurvivalTime = avgSurvival,
            maxSurvivalTime = maxSurvival,
            avgStationaryTime = avgStationary,
            stationaryRatio = avgSurvival > 0 ? avgStationary / avgSurvival : 0f,
            avgTimeInRange = avgTimeInRange,
            avgEngagementRatio = avgEngagement,
            avgOutputMagnitude = avgOutputMag,
            zeroFitnessCount = zeroFitnessCount,
            eliteCount = eliteCount,
            survivalWeight = survivalWeight,
            engagementWeight = engagementWeight,
            campingPenaltyRate = campingPenaltyRate,
            mutationRate = mutationRate,
            mutationMagnitude = mutationMagnitude,
            addNodeRate = addNodeRate,
            addConnectionRate = addConnectionRate,
            weightResetRate = weightResetRate,
            pruneRate = pruneRate,
            pruneThreshold = pruneThreshold,
            avgHiddenNodes = avgHidden,
            avgEnabledConnections = avgConns,
            maxHiddenNodes = maxHidden,
            maxEnabledConnections = maxConns
        };

        string genDir = Path.Combine(savePath, $"Gen_{generation:D4}");
        Directory.CreateDirectory(genDir);
        string json = JsonUtility.ToJson(summary, true);
        File.WriteAllText(Path.Combine(genDir, "generation_summary.json"), json);
    }

    // --- Data Classes ---

    [System.Serializable]
    private class NodeData
    {
        public int id;
        public int type;
        public float bias;
        public float depth;
    }

    [System.Serializable]
    private class ConnData
    {
        public int innovation;
        public int fromNode;
        public int toNode;
        public float weight;
        public bool enabled;
    }

    [System.Serializable]
    private class SaveData
    {
        public int inputs;
        public int outputs;
        public NodeData[] nodes;
        public ConnData[] connections;
        public int generation;
        public float fitness;
        public int rank;
    }

    [System.Serializable]
    private class GlobalState
    {
        public int nextInnovation;
        public int nextNodeId;
    }

    [System.Serializable]
    private class IOAverageData
    {
        public int generation;
        public float[] avgInputs;
        public float[] avgOutputs;
        public int botCount;
    }

    [System.Serializable]
    private class GenerationSummary
    {
        public int generation;
        public int populationSize;
        public float fitnessBest;
        public float fitnessWorst;
        public float fitnessAvg;
        public float fitnessMedian;
        public float fitnessStdDev;
        public float avgSurvivalTime;
        public float maxSurvivalTime;
        public float avgStationaryTime;
        public float stationaryRatio;
        public float avgTimeInRange;
        public float avgEngagementRatio;
        public float avgOutputMagnitude;
        public int zeroFitnessCount;
        public int eliteCount;
        public float survivalWeight;
        public float engagementWeight;
        public float campingPenaltyRate;
        public float mutationRate;
        public float mutationMagnitude;
        public float addNodeRate;
        public float addConnectionRate;
        public float weightResetRate;
        public float pruneRate;
        public float pruneThreshold;
        public float avgHiddenNodes;
        public float avgEnabledConnections;
        public int maxHiddenNodes;
        public int maxEnabledConnections;
    }
}