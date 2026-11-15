using UnityEngine;
using System.Collections.Generic;

/// <summary>
/// Goes on the bot prefab. Handles sensing, network evaluation,
/// movement, and death. Initialized at runtime by BotSpawner.
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

    private NeuralNetwork network;
    private BotSpawner spawner;
    private Transform player;
    private Queue<float[]> moveHistory;
    private float spawnTime;
    private bool alive;
    private Rigidbody ball;

    // 1 (LOS) + 1 (distance) + 2 (direction) + raycasts + memory*2
    public int InputSize => 4 + numRaycasts + moveMemorySize * 2;
    public int OutputSize => 2;
    public float SurvivalTime { get; private set; }
    public float Fitness { get; private set; }

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
        ball = GetComponent<Rigidbody>();

        spawnTime = Time.time;
        alive = true;
        SurvivalTime = 0f;
        ResetMoveHistory();
    }

    void FixedUpdate()
    {
        if (!alive || network == null) return;

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
            transform.position + Vector3.up * 0.5f,
            player.position + Vector3.up * 0.5f,
            environmentMask);
        obs[idx++] = hasLOS ? 1f : 0f;

        // 2. Distance to player (normalized 0-1)
        obs[idx++] = Mathf.Clamp01(toPlayer.magnitude / maxPlayerDistance);

        // 3. Direction to player (horizontal)
        Vector3 dirFlat = new Vector3(toPlayer.x, 0f, toPlayer.z).normalized;
        obs[idx++] = dirFlat.x;
        obs[idx++] = dirFlat.z;

        // 4. Environmental raycasts — evenly spaced around the bot
        Vector3 rayOrigin = transform.position + Vector3.up * 0.5f;
        for (int i = 0; i < numRaycasts; i++)
        {
            float angle = i * (360f / numRaycasts);
            Vector3 dir = Quaternion.Euler(0f, angle, 0f) * Vector3.forward;

            if (Physics.Raycast(rayOrigin, dir, out RaycastHit hit, raycastRange, environmentMask))
                obs[idx++] = hit.distance / raycastRange;
            else
                obs[idx++] = 1f; // open space
        }

        // 5. Previous moves — pseudo-memory for strafing patterns
        foreach (float[] prev in moveHistory)
        {
            obs[idx++] = prev[0];
            obs[idx++] = prev[1];
        }

        return obs;
    }

    // ----------------------------------------------------------
    // Movement
    // ----------------------------------------------------------

    private void ApplyMovement(float[] actions)
    {
        Vector3 move = new Vector3(actions[0], 0f, actions[1]);
        if (move.sqrMagnitude > 1f)
            move.Normalize();

        ball.linearVelocity = new Vector3(
            move.x * moveSpeed,
            ball.linearVelocity.y,  // preserve gravity
            move.z * moveSpeed
        );
    }

    private void RecordMove(float[] actions)
    {
        moveHistory.Enqueue(new float[] { actions[0], actions[1] });
        if (moveHistory.Count > moveMemorySize)
            moveHistory.Dequeue();
    }

    private void ResetMoveHistory()
    {
        moveHistory = new Queue<float[]>();
        for (int i = 0; i < moveMemorySize; i++)
            moveHistory.Enqueue(new float[] { 0f, 0f });
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
        Vector3 origin = transform.position + Vector3.up * 0.5f;

        if (player != null)
        {
            Gizmos.color = Color.red;
            Gizmos.DrawLine(origin, player.position);
        }

        for (int i = 0; i < numRaycasts; i++)
        {
            float angle = i * (360f / numRaycasts);
            Vector3 dir = Quaternion.Euler(0f, angle, 0f) * Vector3.forward;
            Gizmos.color = Color.yellow;
            Gizmos.DrawRay(origin, dir * raycastRange);
        }
    }
}
