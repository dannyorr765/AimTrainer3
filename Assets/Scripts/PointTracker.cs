using UnityEngine;
using TMPro;
using UnityEngine.InputSystem;

public class PointTracker : MonoBehaviour
{
    [SerializeField] private TextMeshProUGUI scoreText;
    [SerializeField] public AimDetection aimDetector;
    [SerializeField] public BulletTracer bulletTracer;
    private int score = 0;
    [SerializeField]public FPSRecoil recoil;

    // Start is called once before the first execution of Update after the MonoBehaviour is created
    void Start()
    {
        
    }

    // Update is called once per frame
    void Update()
    {
        if (aimDetector.isAimingAtBall && Mouse.current.leftButton.wasPressedThisFrame)
        {
            score++;
            aimDetector.GetLastHitBot()?.OnHit();
        }
        scoreText.text = "Points: " + score;

        if(Mouse.current.leftButton.wasPressedThisFrame)
        {
            Shoot();
        }
    }

    void Shoot()
    {
        // Hitscan and bullet tracer.
        bulletTracer.Fire();

        // Camera kick.
        recoil.ApplyRecoil();
    }
}
