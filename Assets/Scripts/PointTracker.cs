using UnityEngine;
using TMPro;
using UnityEngine.InputSystem;

public class PointTracker : MonoBehaviour
{
    [SerializeField] private TextMeshProUGUI scoreText;
    [SerializeField] public AimDetection aimDetector;
    private int score = 0;
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
    }
}
