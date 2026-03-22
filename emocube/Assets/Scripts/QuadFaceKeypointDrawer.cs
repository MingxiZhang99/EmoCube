using System.Collections.Generic;
using UnityEngine;

public class QuadFaceKeypointDrawer : MonoBehaviour
{
    public BlazeFaceOnQuad detector;
    public Transform quadTransform;

    public Material lineMaterial;
    public float pointSize = 0.02f;   // 点的大小（Quad 本地单位）
    public float lineWidth = 0.01f;
    public float zOffset = -0.05f;

    List<LineRenderer> pool = new List<LineRenderer>();

    void Start()
    {
        if (quadTransform == null) quadTransform = transform;
    }

    void Update()
    {
        if (detector == null) return;
        var all = detector.keypoints;

        // 只画第一张脸（你也可以扩展）
        int need = (all.Count > 0) ? all[0].Length : 0;
        EnsurePool(need);

        for (int i = 0; i < pool.Count; i++)
            pool[i].gameObject.SetActive(i < need);

        if (need == 0) return;

        var kps = all[0];
        for (int i = 0; i < kps.Length; i++)
            DrawCross(pool[i], kps[i]);
    }

    void EnsurePool(int n)
    {
        while (pool.Count < n)
        {
            var go = new GameObject("KP");
            go.transform.SetParent(quadTransform, false);
            go.transform.localPosition = new Vector3(0, 0, zOffset);

            var lr = go.AddComponent<LineRenderer>();
            lr.material = lineMaterial;
            lr.useWorldSpace = true;
            lr.widthMultiplier = lineWidth;
            lr.positionCount = 4; // 两条线段
            pool.Add(lr);
        }
    }

    void DrawCross(LineRenderer lr, Vector2 p01)
    {
        Vector3 w = Quad01ToWorld(p01);
        Vector3 dx = quadTransform.right * pointSize * 0.5f;
        Vector3 dy = quadTransform.up * pointSize * 0.5f;

        // 画一个“X”形的十字（两条线）
        lr.SetPosition(0, w - dx);
        lr.SetPosition(1, w + dx);
        lr.SetPosition(2, w - dy);
        lr.SetPosition(3, w + dy);
    }

    Vector3 Quad01ToWorld(Vector2 p01)
    {
        float lx = p01.x - 0.5f;
        float ly = 0.5f - p01.y; // y=0顶部
        return quadTransform.TransformPoint(new Vector3(lx, ly, zOffset));
    }
}
