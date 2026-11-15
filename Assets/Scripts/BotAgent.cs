using UnityEngine;
using System.Collections.Generic;

/// <summary>
/// Goes on the bot prefab. Handles sensing, network evaluation,
/// movement, and death. Initialized at runtime by BotSpawner.
///
/// Prefab setup:
///   - Rigidbody: useGravity OFF, all rotation frozen
///   - Collider (sphere or capsule)
/// </summary>
public class BotAgent : MonoBehaviour
{
    [Header("Movement")]
    [SerializeField] private float moveSpeed = 6f;

    [Header("Sensing")]
    [SerializeField] private int numRaycasts = 12;
    [SerializeField] private float raycastRange = 20f;
    [SerializeField] private float maxPlayerDistance = 50f;
    [SerializeField] private int moveMemorySize = 4;
    [SerializeField] private LayerMask environmentMask;

    [Header("Anti-Camping")]
    [SerializeField] private float emaAlpha = 0.02f;
    [SerializeField] private float campingThreshold = 1.5f;

    private NeuralNetwork network;
    private BotSpawner spawner;
    private Transform player;
    private Queue<float[]> moveHistory;
    private float spawnTime;
    private bool alive;
    private Rigidbody rb;

    // Anti-camping: EMA of position + accumulated stationary time
    private Vector3 positionEMA;
    private float stationaryTime;

    // 1 (LOS) + 1 (distance) + 3 (direction XYZ) + raycasts + memory*3
    public int InputSize => 5 + numRaycasts + moveMemorySize * 3;
    public int OutputSize => 3;  // move X, Y, Z
    public float SurvivalTime { get; private set; }
    public float Fitness { get; private set; }
    public float StationaryTime => stationaryTime;

    public void SetFitness(float fitness)
    {
        Fitness = fitness;
    }

    /// <summary>
    /// Called by BotSpawner right after Instantiate.
    /// </summary>
    public void Initialize(NeuralNetwork network, BotSpawner spawner, Transform player)
    {
        this.network = network;
        this.spawner = spawner;
        this.player = player;
        rb = GetComponent<Rigidbody>();

        spawnTime = Time.time;
        alive = true;
        SurvivalTime = 0f;
        stationaryTime = 0f;
        positionEMA = transform.position;
        ResetMoveHistory();
    }

    void FixedUpdate()
    {
        if (!alive || network == null) return;

        // Anti-camping: update EMA and check if stationary
        positionEMA = Vector3.Lerp(positionEMA, transform.position, emaAlpha);
        float distFromEMA = Vector3.Distance(transform.position, positionEMA);
        if (distFromEMA < campingThreshold)
            stationaryTime += Time.fixedDeltaTime;

        float[] observations = GatherObservations();
        float[] actions = network.Forward(observations);

        ApplyMovement(actions);
        RecordMove(actions);
    }

    // ----------------------------------------------------------
    // Observations
    // ----------------------------------------------------------

    private float[] GatherObservations()
    {
        float[] obs = new float[InputSize];
        int idx = 0;

        Vector3 toPlayer = player.position - transform.position;

        // 1. Line of sight (1 = player visible, 0 = blocked)
        bool hasLOS = !Physics.Linecast(
            transform.position, player.position, environmentMask);
        obs[idx++] = hasLOS ? 1f : 0f;

        // 2. Distance to player (normalized 0-1)
        obs[idx++] = Mathf.Clamp01(toPlayer.magnitude / maxPlayerDistance);

        // 3. Direction to player (full 3D)
        Vector3 dirNorm = toPlayer.normalized;
        obs[idx++] = dirNorm.x;
        obs[idx++] = dirNorm.y;
        obs[idx++] = dirNorm.z;

        // 4. Environmental raycasts — evenly spaced horizontally around the bot
        for (int i = 0; i < numRaycasts; i++)
        {
            float angle = i * (360f / numRaycasts);
            Vector3 dir = Quaternion.Euler(0f, angle, 0f) * Vector3.forward;

            if (Physics.Raycast(transform.position, dir, out RaycastHit hit,
                    raycastRange, environmentMask))
                obs[idx++] = hit.distance / raycastRange;
            else
                obs[idx++] = 1f; // open space
        }

        // 5. Previous moves — pseudo-memory for movement patterns
        foreach (float[] prev in moveHistory)
        {
            obs[idx++] = prev[0];
            obs[idx++] = prev[1];
            obs[idx++] = prev[2];
        }

        return obs;
    }

    // ----------------------------------------------------------
    // Movement (3D — no gravity)
    // ----------------------------------------------------------

    private void ApplyMovement(float[] actions)
    {
        Vector3 move = new Vector3(actions[0], actions[1], actions[2]);
        if (move.sqrMagnitude > 1f)
            move.Normalize();

        rb.linearVelocity = move * moveSpeed;
    }

    private void RecordMove(float[] actions)
    {
        moveHistory.Enqueue(new float[] { actions[0], actions[1], actions[2] });
        if (moveHistory.Count > moveMemorySize)
            moveHistory.Dequeue();
    }

    private void ResetMoveHistory()
    {
        moveHistory = new Queue<float[]>();
        for (int i = 0; i < moveMemorySize; i++)
            moveHistory.Enqueue(new float[] { 0f, 0f, 0f });
    }

    // ----------------------------------------------------------
    // Hit — call this from your shooting script
    // ----------------------------------------------------------

    /// <summary>
    /// One-shot kill. Records survival time and notifies the spawner.
    /// </summary>
    public void OnHit()
    {
        if (!alive) return;

        alive = false;
        SurvivalTime = Time.time - spawnTime;
        gameObject.SetActive(false);
        spawner.OnBotKilled(this, SurvivalTime);
    }

    // ----------------------------------------------------------
    // Debug
    // ----------------------------------------------------------

    void OnDrawGizmosSelected()
    {
        if (player != null)
        {
            Gizmos.color = Color.red;
            Gizmos.DrawLine(transform.position, player.position);
        }

        for (int i = 0; i < numRaycasts; i++)
        {
            float angle = i * (360f / numRaycasts);
            Vector3 dir = Quaternion.Euler(0f, angle, 0f) * Vector3.forward;
            Gizmos.color = Color.yellow;
            Gizmos.DrawRay(transform.position, dir * raycastRange);
        }

        // Show EMA position and camping threshold
        if (Application.isPlaying && alive)
        {
            Gizmos.color = stationaryTime > 0.5f ? Color.red : Color.green;
            Gizmos.DrawWireSphere(positionEMA, campingThreshold);
        }
    }
}