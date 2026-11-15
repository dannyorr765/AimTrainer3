
using UnityEngine;
using System.Collections;

public class BulletTracer : MonoBehaviour
{
    [Header("References")]
    public Transform muzzlePoint;
    public Camera playerCamera;

    [Header("Settings")]
    public float range = 100f;
    public float tracerDuration = 0.00000001f;
    public float tracerWidth = 0.005f;
    public LayerMask hitLayers = ~0;

    private Material tracerMaterial;

    void Awake()
    {
        tracerMaterial = new Material(
            Shader.Find("Sprites/Default")
        );
    }

    public void Fire()
    {
        // Raycast from the center of the screen.
        Ray ray = playerCamera.ViewportPointToRay(
            new Vector3(0.5f, 0.5f, 0f)
        );

        Vector3 hitPoint;

        if (Physics.Raycast(ray, out RaycastHit hit,
            range, hitLayers))
        {
            hitPoint = hit.point;
            Debug.Log("Hit: " + hit.collider.name);
        }
        else
        {
            hitPoint = ray.GetPoint(range);
        }

        // Capture the muzzle's position at firing time.
        Vector3 muzzlePosition = muzzlePoint.position;

        // Create a separate tracer in world space.
        GameObject tracerObject =
            new GameObject("BulletTracer");

        LineRenderer line =
            tracerObject.AddComponent<LineRenderer>();

        line.material = tracerMaterial;
        line.positionCount = 2;
        line.useWorldSpace = true;

        line.startWidth = tracerWidth;
        line.endWidth = tracerWidth * 0.5f;

        line.startColor = Color.white;
        line.endColor = Color.white;

        line.SetPosition(0, muzzlePosition);
        line.SetPosition(1, hitPoint);

        // Remove the tracer after a short time.
        Destroy(tracerObject, tracerDuration);
    }

    void OnDestroy()
    {
        if (tracerMaterial != null)
            Destroy(tracerMaterial);
    }
}
