using System.Collections.Generic;
//using System.Diagnostics;
using UnityEngine;

public class QuadFaceBoxDrawer : MonoBehaviour
{
    [Header("Source")]
    public BlazeFaceOnQuad detector;
    public Transform quadTransform;

    [Header("Line")]
    public Material lineMaterial;
    public float lineWidth = 0.05f;
    public float zOffset = -0.2f;
    public int maxBoxes = 5;

    readonly List<LineRenderer> pool = new List<LineRenderer>();
    bool createdOnce = false;

    void Start()
    {
        if (quadTransform == null) quadTransform = transform;

        Debug.Log("[BoxDrawer] Start() called. quadTransform=" + quadTransform.name);

        // 关键：一启动就创建 pool，用来验证“会不会生成子物体”
        EnsurePool(maxBoxes);
        createdOnce = true;
        Debug.Log("[BoxDrawer] Pool created: " + pool.Count);
    }

    void Update()
    {
        Debug.Log("[BoxDrawer] Update running");

        if (detector == null)
        {
            Debug.LogWarning("[BoxDrawer] detector is NULL (Inspector 里拖 BlazeFaceOnQuad 组件实例)");
            return;
        }

        var rects = detector.results;
        Debug.Log("[BoxDrawer] detector ok, rects=" + (rects == null ? -1 : rects.Count));

        // 没检测到也没关系，先把所有线显示出来做测试
        for (int i = 0; i < pool.Count; i++)
            pool[i].gameObject.SetActive(true);

        // 如果有框才画框；没有框就画一个固定框验证可见性
        if (rects != null && rects.Count > 0)
        {
            for (int i = 0; i < pool.Count; i++)
            {
                if (i < rects.Count) DrawRectOnQuad(pool[i], rects[i]);
            }
        }
        else
        {
            // 画一个固定框（居中 50% 大小）验证你一定能看到线
            DrawRectOnQuad(pool[0], new Rect(0.25f, 0.25f, 0.5f, 0.5f));
        }
    }

    void EnsurePool(int n)
    {
        while (pool.Count < n)
        {
            var go = new GameObject("FaceBoxLine_" + pool.Count);
            go.transform.SetParent(quadTransform, false);
            go.transform.localPosition = new Vector3(0, 0, zOffset);

            var lr = go.AddComponent<LineRenderer>();
            lr.positionCount = 5;
            lr.loop = false;
            lr.useWorldSpace = true;
            lr.widthMultiplier = lineWidth;

            if (lineMaterial != null) lr.material = lineMaterial;
            else Debug.LogWarning("[BoxDrawer] lineMaterial is NULL");

            pool.Add(lr);
        }
    }

    void DrawRectOnQuad(LineRenderer lr, Rect r)
    {
        float xmin = r.xMin;
        float yminTop = r.yMin;
        float xmax = r.xMax;
        float ymaxTop = r.yMax;

        Vector3 p0 = QuadLocalToWorld(xmin, yminTop);
        Vector3 p1 = QuadLocalToWorld(xmax, yminTop);
        Vector3 p2 = QuadLocalToWorld(xmax, ymaxTop);
        Vector3 p3 = QuadLocalToWorld(xmin, ymaxTop);

        lr.SetPosition(0, p0);
        lr.SetPosition(1, p1);
        lr.SetPosition(2, p2);
        lr.SetPosition(3, p3);
        lr.SetPosition(4, p0);
    }

    Vector3 QuadLocalToWorld(float x01, float yTop01)
    {
        float lx = (x01 - 0.5f);
        float ly = (0.5f - yTop01);
        return quadTransform.TransformPoint(new Vector3(lx, ly, zOffset));
    }
}
