using UnityEngine;

public class QuadOverlayDrawerOfficial : MonoBehaviour
{
    [Header("Source")]
    public BlazeFaceOfficialOnQuad detector;
    public Transform quadTransform;

    [Header("Materials")]
    public Material boxMat;       // Line.mat
    public Material keypointMat;  // Keypoint.mat (也可以用同一个)

    [Header("Box Style")]
    public float boxLineWidth = 0.02f;
    public float zOffset = -0.05f;

    [Header("Keypoint Style")]
    public float keypointSize = 0.03f;
    public float keypointWidth = 0.03f;

    LineRenderer boxLR;
    LineRenderer[] kpLR = new LineRenderer[6];

    void Start()
    {
        if (quadTransform == null) quadTransform = transform;

        // Box LR
        boxLR = CreateLine("FaceBox", boxMat, boxLineWidth);
        boxLR.positionCount = 5;

        // Keypoints LR (6)
        for (int i = 0; i < kpLR.Length; i++)
        {
            var lr = CreateLine("KP_" + i, keypointMat != null ? keypointMat : boxMat, keypointWidth);
            lr.positionCount = 2; // draw a point as a "dot line"
            lr.numCapVertices = 8; // make it rounder
            kpLR[i] = lr;
        }

        HideAll();
    }

    void Update()
    {
        if (detector == null) { HideAll(); return; }
        if (!detector.HasFace) { HideAll(); return; }

        // draw 1 box + 6 points
        DrawBox(detector.FaceRect01);
        DrawKeypoints(detector.Keypoints01);
    }

    void HideAll()
    {
        if (boxLR != null) boxLR.gameObject.SetActive(false);
        for (int i = 0; i < kpLR.Length; i++)
            if (kpLR[i] != null) kpLR[i].gameObject.SetActive(false);
    }

    LineRenderer CreateLine(string name, Material mat, float width)
    {
        var go = new GameObject(name);
        go.transform.SetParent(quadTransform, false);
        go.transform.localPosition = new Vector3(0, 0, zOffset);

        var lr = go.AddComponent<LineRenderer>();
        lr.useWorldSpace = true;
        lr.loop = false;
        lr.widthMultiplier = width;
        lr.material = mat;
        lr.shadowCastingMode = UnityEngine.Rendering.ShadowCastingMode.Off;
        lr.receiveShadows = false;

        return lr;
    }

    void DrawBox(Rect r01)
    {
        boxLR.gameObject.SetActive(true);

        float xmin = r01.xMin;
        float yminTop = r01.yMin;
        float xmax = r01.xMax;
        float ymaxTop = r01.yMax;

        Vector3 p0 = Quad01ToWorld(new Vector2(xmin, yminTop));
        Vector3 p1 = Quad01ToWorld(new Vector2(xmax, yminTop));
        Vector3 p2 = Quad01ToWorld(new Vector2(xmax, ymaxTop));
        Vector3 p3 = Quad01ToWorld(new Vector2(xmin, ymaxTop));

        boxLR.SetPosition(0, p0);
        boxLR.SetPosition(1, p1);
        boxLR.SetPosition(2, p2);
        boxLR.SetPosition(3, p3);
        boxLR.SetPosition(4, p0);
    }

    void DrawKeypoints(Vector2[] kps01)
    {
        if (kps01 == null || kps01.Length < 6) return;

        for (int i = 0; i < 6; i++)
        {
            var lr = kpLR[i];
            lr.gameObject.SetActive(true);

            Vector3 p = Quad01ToWorld(kps01[i]);

            // draw as tiny segment (round caps look like dot)
            Vector3 dx = quadTransform.right * (keypointSize * 0.5f);
            lr.SetPosition(0, p - dx);
            lr.SetPosition(1, p + dx);
        }
    }

    // p01: x=0..1 left->right, y=0..1 top->bottom
    Vector3 Quad01ToWorld(Vector2 p01)
    {
        float lx = p01.x - 0.5f;
        float ly = 0.5f - p01.y; // y=0 at top
        return quadTransform.TransformPoint(new Vector3(lx, ly, zOffset));
    }
}
