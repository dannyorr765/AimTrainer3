using UnityEngine;
using System.Collections.Generic;
using System.Linq;
using System.IO;

/// <summary>
/// Spawns a population of bot prefabs, tracks their survival,
/// and evolves the neural networks between generations.
/// Position this object where you want bots to spawn around.
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

    [Header("Network")]
    [SerializeField] private int[] hiddenLayers = { 32, 24 };

    [Header("Evolution")]
    [SerializeField] private int eliteCount = 5;
    [SerializeField] private float mutationRate = 0.15f;
    [SerializeField] private float mutationMagnitude = 0.3f;
    [SerializeField] private float campingPenaltyRate = 0.5f;

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
    private float lastKillTime;
    private string savePath;
    private int[] topology;
    private bool shuttingDown = false;

    void Start()
    {
        // Save path lives next to Assets/ in the project root
        savePath = Path.Combine(Application.dataPath, "..", saveFolder);
        Debug.Log($"[Evolution] Networks will be saved to: {Path.GetFullPath(savePath)}");

        // Read input/output dimensions from the prefab
        var template = botPrefab.GetComponent<BotAgent>();

        // Build network topology: input → hidden → output
        topology = new int[hiddenLayers.Length + 2];
        topology[0] = template.InputSize;
        for (int i = 0; i < hiddenLayers.Length; i++)
            topology[i + 1] = hiddenLayers[i];
        topology[topology.Length - 1] = template.OutputSize;

        Debug.Log($"[Evolution] Topology: [{string.Join(", ", topology)}], Pop: {botsPerGeneration}");

        // Try to resume from saved networks
        if (TryLoadLatestGeneration())
        {
            Debug.Log($"[Evolution] Resumed from generation {generation - 1}, starting gen {generation}");
        }
        else
        {
            population = new List<NeuralNetwork>();
            for (int i = 0; i < botsPerGeneration; i++)
                population.Add(new NeuralNetwork(topology));
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
        lastKillTime = Time.time;

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
    /// Fitness = time since the player's previous kill (isolates dodging skill).
    /// Triggers evolution once all bots are dead.
    /// </summary>
    public void OnBotKilled(BotAgent bot, float survivalTime)
    {
        if (shuttingDown) return;

        float now = Time.time;
        float timeSinceLastKill = now - lastKillTime;
        lastKillTime = now;

        float campingPenalty = bot.StationaryTime * campingPenaltyRate;
        float adjustedFitness = Mathf.Max(0f, timeSinceLastKill - campingPenalty);
        bot.SetFitness(adjustedFitness);
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

        Debug.Log($"[Gen {generation}] Best: {genBest:F2}s | Avg: {genAvg:F2}s | All-time: {bestFitnessEver:F2}s");

        // Save top networks before building new population
        SaveTopNetworks(fitnesses, ranked);

        var newPop = new List<NeuralNetwork>();

        // Elitism — top performers pass through unchanged
        for (int i = 0; i < eliteCount && i < population.Count; i++)
            newPop.Add(population[ranked[i]].Clone());

        // Fill the rest via tournament selection → crossover → mutation
        while (newPop.Count < botsPerGeneration)
        {
            var parentA = TournamentSelect(fitnesses);
            var parentB = TournamentSelect(fitnesses);
            var child = NeuralNetwork.Crossover(parentA, parentB);
            child.Mutate(mutationRate, mutationMagnitude);
            newPop.Add(child);
        }

        population = newPop;
        generation++;
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

            var data = new SaveData
            {
                topology = net.GetTopology(),
                weights = net.Flatten(),
                generation = generation,
                fitness = fitnesses[idx],
                rank = rank
            };

            string json = JsonUtility.ToJson(data, true);  // pretty-print
            string path = Path.Combine(genDir, $"network_{rank}.json");
            File.WriteAllText(path, json);
        }

        Debug.Log($"[Gen {generation}] Saved top {count} networks to {genDir}");
    }

    // --- Load ---

    /// <summary>
    /// Load a saved network and inject it into the population at index 0.
    /// Call before the next generation spawns.
    /// </summary>
    public void LoadNetwork(string jsonPath)
    {
        string json = File.ReadAllText(jsonPath);
        var data = JsonUtility.FromJson<SaveData>(json);
        var net = new NeuralNetwork(data.topology);
        net.Unflatten(data.weights);

        if (population != null && population.Count > 0)
        {
            population[0] = net;
            Debug.Log($"[Evolution] Loaded network from gen {data.generation} (fitness: {data.fitness:F2}) into slot 0");
        }
        else
        {
            Debug.LogWarning("[Evolution] Population not initialized yet. Load after Start().");
        }
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

        // Load all network files from that generation
        var loaded = new List<NeuralNetwork>();
        float loadedBestFitness = 0f;

        string[] files = Directory.GetFiles(latestDir, "network_*.json");
        System.Array.Sort(files);  // network_0, network_1, ...

        foreach (string file in files)
        {
            try
            {
                string json = File.ReadAllText(file);
                var data = JsonUtility.FromJson<SaveData>(json);
                var net = new NeuralNetwork(data.topology);
                net.Unflatten(data.weights);
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

        if (loaded.Count == 0) return false;

        // Set generation to one past the loaded one
        generation = latestGen + 1;
        bestFitnessEver = loadedBestFitness;

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
            child.Mutate(mutationRate, mutationMagnitude);
            population.Add(child);
        }

        Debug.Log($"[Evolution] Loaded {loaded.Count} networks from Gen {latestGen} (best fitness: {loadedBestFitness:F2}s)");
        return true;
    }

    [System.Serializable]
    private class SaveData
    {
        public int[] topology;
        public float[] weights;
        public int generation;
        public float fitness;
        public int rank;
    }
}