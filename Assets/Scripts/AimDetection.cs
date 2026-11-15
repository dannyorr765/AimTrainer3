using UnityEngine;

public class AimDetection : MonoBehaviour
{
    [SerializeField] private GameObject camera;
    public bool isAimingAtBall = false;
    private BotAgent lastHitBot;
    public out hit;

    void FixedUpdate()
    {
        Vector3 camOrigin = camera.transform.position;
        Vector3 camDirection = camera.transform.forward;

        RaycastHit hit;

        if (Physics.Raycast(camOrigin, camDirection, hit))
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