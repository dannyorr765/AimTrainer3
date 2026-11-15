
using UnityEngine;
using Unity.Cinemachine;

public class FPSRecoil : MonoBehaviour
{
    public CinemachineRecomposer recomposer;

    public float verticalRecoil = 5f;
    public float horizontalRecoil = 0f;
    public float snappiness = 18f;
    public float returnSpeed = 5f;

    private Vector2 targetRecoil;
    private Vector2 currentRecoil;

    public void ApplyRecoil()
    {
        Debug.Log("RECOIL ACTIVATED!");

        targetRecoil += new Vector2(
            verticalRecoil,
            Random.Range(-horizontalRecoil, horizontalRecoil)
        );
    }

    void LateUpdate()
    {
        if (recomposer == null) return;

        float dt = Time.deltaTime;

        targetRecoil = Vector2.Lerp(
            targetRecoil,
            Vector2.zero,
            returnSpeed * dt
        );

        currentRecoil = Vector2.Lerp(
            currentRecoil,
            targetRecoil,
            snappiness * dt
        );

        recomposer.Tilt = currentRecoil.x;
        recomposer.Pan = currentRecoil.y;
    }
}
