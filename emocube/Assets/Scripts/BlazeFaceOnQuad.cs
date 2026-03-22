using System;
using System.Collections.Generic;
//using System.Diagnostics;

//using System.Diagnostics;
using UnityEngine;

public class BlazeFaceOnQuad : MonoBehaviour
{
    [Header("Model")]
    public Unity.InferenceEngine.ModelAsset onnxModel;
    public Unity.InferenceEngine.BackendType backend = Unity.InferenceEngine.BackendType.GPUCompute;

    [Header("Anchors")]
    public TextAsset anchorsCsv;

    [Header("Input")]
    public int inputSize = 128;
    public float scoreThreshold = 0.30f;
    public float iouThreshold = 0.30f;
    public int maxFaces = 2;
    public float inferInterval = 0.05f;

    [Header("Decode (BlazeFace)")]
    public bool inputIsMinus1To1 = true;        // 你现在用的是 -1..1
    public bool useAnchorSizeForWH = true;      // 常见：w,h 乘 anchor 的 w,h
    public bool useAnchorSizeForXY = true;      // 常见：x,y 偏移也按 anchor 尺寸缩放
    public float xScale = 128f;
    public float yScale = 128f;
    public float wScale = 128f;
    public float hScale = 128f;

    [Header("Debug")]
    public bool enableLogs = true;
    public int logEveryN = 10;

    // 输出：脸框（0..1）
    public readonly List<Rect> results = new List<Rect>();
    // 输出：每张脸 6 个关键点（0..1）
    public readonly List<Vector2[]> keypoints = new List<Vector2[]>();

    Unity.InferenceEngine.Model runtimeModel;
    Unity.InferenceEngine.Worker worker;

    WebCamTexture camTex;
    Texture2D cpuTex;

    // 为了避免每帧 GC alloc
    Color32[] camPixels;
    Color32[] dstPixels;

    struct Anchor { public float x, y, w, h; }
    Anchor[] anchors;

    float timer;
    int inferCount = 0;
    bool printedWebcamReady = false;

    const string INPUT_NAME = "input";
    const string BOXES_NAME = "regressors";       // (1,896,16)
    const string SCORES_NAME = "classificators";  // (1,896,1)

    void Start()
    {
        if (onnxModel == null) { Debug.LogError("[BlazeFace] onnxModel is null"); enabled = false; return; }
        if (anchorsCsv == null) { Debug.LogError("[BlazeFace] anchorsCsv is null"); enabled = false; return; }

        camTex = (WebcamManager.Instance != null) ? WebcamManager.Instance.CamTex : null;
        if (camTex == null) { Debug.LogError("[BlazeFace] WebcamManager.Instance.CamTex is null"); enabled = false; return; }

        anchors = LoadAnchors(anchorsCsv.text);
        if (anchors == null || anchors.Length == 0) { Debug.LogError("[BlazeFace] anchors load failed"); enabled = false; return; }

        cpuTex = new Texture2D(inputSize, inputSize, TextureFormat.RGBA32, false);
        dstPixels = new Color32[inputSize * inputSize];

        runtimeModel = Unity.InferenceEngine.ModelLoader.Load(onnxModel);
        worker = new Unity.InferenceEngine.Worker(runtimeModel, backend);

        results.Clear();
        keypoints.Clear();

        if (enableLogs)
        {
            Debug.Log($"[BlazeFace] Model={onnxModel.name}, backend={backend}");
            Debug.Log($"[BlazeFace] Anchors={anchors.Length} (expect 896)");
            Debug.Log($"[BlazeFace] Outputs: {BOXES_NAME} (1,896,16), {SCORES_NAME} (1,896,1)");
            Debug.Log("[BlazeFace] Input preproc: rotate by webcam.videoRotationAngle + apply videoVerticallyMirrored, then center-crop to square.");
        }
    }

    void OnDestroy()
    {
        if (cpuTex != null) Destroy(cpuTex);
        worker?.Dispose();
    }

    void Update()
    {
        if (camTex == null || !camTex.isPlaying) return;
        if (camTex.width <= 16 || camTex.height <= 16) return;

        if (!printedWebcamReady)
        {
            printedWebcamReady = true;
            if (enableLogs)
                Debug.Log($"[BlazeFace] Webcam ready {camTex.width}x{camTex.height} rot={camTex.videoRotationAngle} mirrorVert={camTex.videoVerticallyMirrored}");
        }

        timer += Time.deltaTime;
        if (timer < inferInterval) return;
        timer = 0f;

        inferCount++;
        bool doLog = enableLogs && (inferCount % Mathf.Max(1, logEveryN) == 0);

        // ✅ 关键修改：把摄像头画面转正到“竖屏(portrait/upright)”方向，再裁成正方形，缩放到 inputSize
        CopyWebcamToCPUTexture_PortraitUprightSquare(camTex, cpuTex);

        using var input = BuildInputTensorNHWC(cpuTex);

        worker.SetInput(INPUT_NAME, input);
        worker.Schedule();

        var boxesT = worker.PeekOutput(BOXES_NAME) as Unity.InferenceEngine.Tensor<float>;
        var scoresT = worker.PeekOutput(SCORES_NAME) as Unity.InferenceEngine.Tensor<float>;
        if (boxesT == null || scoresT == null)
        {
            Debug.LogError("[BlazeFace] output missing (check names regressors/classificators)");
            return;
        }

        float[] rawBoxes = boxesT.DownloadToArray();   // 896*16
        float[] rawScores = scoresT.DownloadToArray(); // 896

        DecodeAndNmsWithKeypoints(rawBoxes, rawScores, results, keypoints, out int above, out float maxScore);

        if (doLog)
        {
            Debug.Log($"[BlazeFace] #{inferCount}: maxScore={maxScore:0.000}, aboveTh={above}, faces={results.Count}");
            if (results.Count > 0)
            {
                var r = results[0];
                Debug.Log($"[BlazeFace] Face0 rect = ({r.xMin:0.000},{r.yMin:0.000},{r.width:0.000},{r.height:0.000})");
                var kp = keypoints[0];
                Debug.Log($"[BlazeFace] Face0 k0={kp[0].x:0.000},{kp[0].y:0.000}  k1={kp[1].x:0.000},{kp[1].y:0.000}");
            }
        }
    }

    /// <summary>
    /// 把 WebCamTexture 画面先按 videoRotationAngle 旋转到 upright（人看起来是竖直的），
    /// 再按 upright 画面居中裁成正方形，最后缩放到 dst(inputSize x inputSize)。
    ///
    /// 这样模型输入就是 portrait/upright 的，输出框也会变成“竖直方向”的坐标系。
    ///
    /// 注意：这里使用 Unity 像素坐标（origin 在左下），与 GetPixels32/SetPixels32 一致。
    /// </summary>
    void CopyWebcamToCPUTexture_PortraitUprightSquare(WebCamTexture src, Texture2D dst)
    {
        int sw = src.width;
        int sh = src.height;
        if (sw <= 0 || sh <= 0) return;

        // 拿相机像素（避免每帧 new）
        if (camPixels == null || camPixels.Length != sw * sh)
            camPixels = src.GetPixels32();
        else
            src.GetPixels32(camPixels);

        int rot = ((src.videoRotationAngle % 360) + 360) % 360; // 0/90/180/270
        bool mirrorV = src.videoVerticallyMirrored;

        // upright 后的宽高
        int uw = (rot == 90 || rot == 270) ? sh : sw;
        int uh = (rot == 90 || rot == 270) ? sw : sh;

        // 居中裁成正方形
        int crop = Mathf.Min(uw, uh);
        float cropX0 = (uw - crop) * 0.5f;
        float cropY0 = (uh - crop) * 0.5f;

        int dw = dst.width;
        int dh = dst.height;

        // 最近邻采样（inputSize 很小，够快；想更平滑可以自己改双线性）
        for (int y = 0; y < dh; y++)
        {
            float v = (y + 0.5f) / dh;                // 0..1（左下原点）
            float uy = cropY0 + v * crop;             // upright 坐标
            for (int x = 0; x < dw; x++)
            {
                float u = (x + 0.5f) / dw;
                float ux = cropX0 + u * crop;

                // videoVerticallyMirrored：在“显示为 upright”意义上做垂直翻转
                float uxm = ux;
                float uym = mirrorV ? (uh - 1f - uy) : uy;

                // upright -> source 反变换（假设 upright = source 顺时针旋转 rot）
                float sx, sy;
                switch (rot)
                {
                    case 0:
                        sx = uxm;
                        sy = uym;
                        break;

                    case 90:
                        // src -> upright (CW90): newX = H-1 - y, newY = x
                        // inverse: x = newY, y = H-1 - newX
                        sx = uym;
                        sy = (sh - 1f) - uxm;
                        break;

                    case 180:
                        sx = (sw - 1f) - uxm;
                        sy = (sh - 1f) - uym;
                        break;

                    case 270:
                        // src -> upright (CW270): newX = y, newY = W-1 - x
                        // inverse: x = W-1 - newY, y = newX
                        sx = (sw - 1f) - uym;
                        sy = uxm;
                        break;

                    default:
                        sx = uxm;
                        sy = uym;
                        break;
                }

                int isx = Mathf.Clamp(Mathf.RoundToInt(sx), 0, sw - 1);
                int isy = Mathf.Clamp(Mathf.RoundToInt(sy), 0, sh - 1);

                dstPixels[y * dw + x] = camPixels[isy * sw + isx];
            }
        }

        dst.SetPixels32(dstPixels);
        dst.Apply(false, false);
    }

    Unity.InferenceEngine.Tensor<float> BuildInputTensorNHWC(Texture2D tex)
    {
        Color32[] pixels = tex.GetPixels32();
        int H = tex.height;
        int W = tex.width;

        var shape = new Unity.InferenceEngine.TensorShape(1, H, W, 3);
        var t = new Unity.InferenceEngine.Tensor<float>(shape);

        for (int y = 0; y < H; y++)
        {
            int row = y * W;
            for (int x = 0; x < W; x++)
            {
                Color32 p = pixels[row + x];

                if (inputIsMinus1To1)
                {
                    t[0, y, x, 0] = (p.r / 127.5f) - 1f;
                    t[0, y, x, 1] = (p.g / 127.5f) - 1f;
                    t[0, y, x, 2] = (p.b / 127.5f) - 1f;
                }
                else
                {
                    t[0, y, x, 0] = p.r / 255f;
                    t[0, y, x, 1] = p.g / 255f;
                    t[0, y, x, 2] = p.b / 255f;
                }
            }
        }
        return t;
    }

    void DecodeAndNmsWithKeypoints(
        float[] rawBoxes, float[] rawScores,
        List<Rect> outRects, List<Vector2[]> outKps,
        out int numAbove, out float maxScore)
    {
        outRects.Clear();
        outKps.Clear();
        numAbove = 0;
        maxScore = 0f;

        const int NUM = 896;
        const int STRIDE = 16;

        List<Cand> cands = new List<Cand>(NUM);

        for (int i = 0; i < NUM; i++)
        {
            float score = Sigmoid(rawScores[i]);
            if (score > maxScore) maxScore = score;
            if (score < scoreThreshold) continue;
            numAbove++;

            int bi = i * STRIDE;
            Anchor a = anchors[i];

            float dx = rawBoxes[bi + 0];
            float dy = rawBoxes[bi + 1];
            float dw = rawBoxes[bi + 2];
            float dh = rawBoxes[bi + 3];

            float xCenter = (dx / xScale) * (useAnchorSizeForXY ? a.w : 1f) + a.x;
            float yCenter = (dy / yScale) * (useAnchorSizeForXY ? a.h : 1f) + a.y;
            float w = (dw / wScale) * (useAnchorSizeForWH ? a.w : 1f);
            float h = (dh / hScale) * (useAnchorSizeForWH ? a.h : 1f);

            float xmin = xCenter - w * 0.5f;
            float ymin = yCenter - h * 0.5f;
            float xmax = xCenter + w * 0.5f;
            float ymax = yCenter + h * 0.5f;

            Vector2[] kp = new Vector2[6];
            for (int k = 0; k < 6; k++)
            {
                float kx = rawBoxes[bi + 4 + k * 2 + 0];
                float ky = rawBoxes[bi + 4 + k * 2 + 1];

                float x = (kx / xScale) * (useAnchorSizeForXY ? a.w : 1f) + a.x;
                float y = (ky / yScale) * (useAnchorSizeForXY ? a.h : 1f) + a.y;
                kp[k] = new Vector2(x, y);
            }

            xmin = Mathf.Clamp01(xmin);
            ymin = Mathf.Clamp01(ymin);
            xmax = Mathf.Clamp01(xmax);
            ymax = Mathf.Clamp01(ymax);

            for (int k = 0; k < 6; k++)
            {
                kp[k].x = Mathf.Clamp01(kp[k].x);
                kp[k].y = Mathf.Clamp01(kp[k].y);
            }

            cands.Add(new Cand
            {
                score = score,
                xmin = xmin,
                ymin = ymin,
                xmax = xmax,
                ymax = ymax,
                kps = kp
            });
        }

        cands.Sort((a, b) => b.score.CompareTo(a.score));

        List<Cand> kept = new List<Cand>(maxFaces);
        for (int i = 0; i < cands.Count; i++)
        {
            var c = cands[i];
            bool keep = true;
            for (int j = 0; j < kept.Count; j++)
            {
                if (IoU(c, kept[j]) > iouThreshold) { keep = false; break; }
            }
            if (!keep) continue;
            kept.Add(c);
            if (kept.Count >= maxFaces) break;
        }

        for (int i = 0; i < kept.Count; i++)
        {
            var c = kept[i];
            outRects.Add(new Rect(c.xmin, c.ymin, c.xmax - c.xmin, c.ymax - c.ymin));
            outKps.Add(c.kps);
        }
    }

    struct Cand
    {
        public float score;
        public float xmin, ymin, xmax, ymax;
        public Vector2[] kps;
    }

    static float Sigmoid(float x) => 1f / (1f + Mathf.Exp(-x));

    static float IoU(Cand a, Cand b)
    {
        float ixmin = Mathf.Max(a.xmin, b.xmin);
        float iymin = Mathf.Max(a.ymin, b.ymin);
        float ixmax = Mathf.Min(a.xmax, b.xmax);
        float iymax = Mathf.Min(a.ymax, b.ymax);

        float iw = Mathf.Max(0f, ixmax - ixmin);
        float ih = Mathf.Max(0f, iymax - iymin);
        float inter = iw * ih;

        float areaA = (a.xmax - a.xmin) * (a.ymax - a.ymin);
        float areaB = (b.xmax - b.xmin) * (b.ymax - b.ymin);
        float union = areaA + areaB - inter + 1e-6f;

        return inter / union;
    }

    static Anchor[] LoadAnchors(string csvText)
    {
        var lines = csvText.Split(new[] { '\n', '\r' }, StringSplitOptions.RemoveEmptyEntries);
        List<Anchor> list = new List<Anchor>(lines.Length);

        foreach (var line in lines)
        {
            var s = line.Trim();
            if (string.IsNullOrEmpty(s)) continue;

            var parts = s.Split(',');
            if (parts.Length < 4) continue;

            if (float.TryParse(parts[0], out float x) &&
                float.TryParse(parts[1], out float y) &&
                float.TryParse(parts[2], out float w) &&
                float.TryParse(parts[3], out float h))
            {
                list.Add(new Anchor { x = x, y = y, w = w, h = h });
            }
        }
        return list.ToArray();
    }

    void Log(string msg)
    {
        if (enableLogs) Debug.Log(msg);
    }
}
