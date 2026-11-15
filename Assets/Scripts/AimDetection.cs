using UnityEngine;

public class AimDetection : MonoBehaviour
{
    [SerializeField] private GameObject camera;
    public bool isAimingAtBall = false;
    private BotAgent lastHitBot;
    public RaycastHit hit;

    void FixedUpdate()
    {
        Vector3 camOrigin = camera.transform.position;
        Vector3 camDirection = camera.transform.forward;

        if (Physics.Raycast(camOrigin, camDirection, out hit))
        {
            lastHitBot = hit.collider.GetComponent<BotAgent>();
            isAimingAtBall = lastHitBot != null;
        }
        else
        {
            isAimingAtBall = false;
            lastHitBot = null;
        }
    }

    public BotAgent GetLastHitBot() => lastHitBot;
}