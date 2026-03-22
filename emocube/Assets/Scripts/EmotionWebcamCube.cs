//using System.Diagnostics;
//using System.Diagnostics;
using UnityEngine;

public class EmotionWebcamCube : MonoBehaviour
{
    [Header("Model")]
    public Unity.InferenceEngine.ModelAsset onnxModel;
    public Unity.InferenceEngine.BackendType backend = Unity.InferenceEngine.BackendType.GPUCompute;

    [Header("Target")]
    public Renderer cubeRenderer;

    [Header("Webcam")]
    public int webcamIndex = 0;
    public int webcamWidth = 640;
    public int webcamHeight = 480;
    public int webcamFPS = 30;

    [Header("Inference")]
    public int inputSize = 224;
    public float inferInterval = 3f;
    [Range(0f, 1f)]
    public float unknownThreshold = 0.55f;

    [Header("Debug")]
    public bool enableLogs = true;
    public int logEveryN = 10; // log once every N inferences

    static readonly float[] MEAN = { 0.485f, 0.456f, 0.406f };
    static readonly float[] STD = { 0.229f, 0.224f, 0.225f };

    WebCamTexture camTex;
    Texture2D cpuTex;

    Unity.InferenceEngine.Model runtimeModel;
    Unity.InferenceEngine.Worker worker;

    float timer;
    int inferCount = 0;

    enum Emotion { Happy = 0, Sad = 1, Angry = 2, Fear = 3, Unknown = 999 }

    void Start()
    {
        if (enableLogs) Debug.Log("[Emotion] Start()");

        if (onnxModel == null)
        {
            //Debug.LogError("[Emotion] onnxModel is null. Assign the ONNX model in Inspector.");
            enabled = false;
            return;
        }
        if (cubeRenderer == null)
        {
            //Debug.LogError("[Emotion] cubeRenderer is null. Assign the Cube Renderer in Inspector.");
            enabled = false;
            return;
        }

        if (enableLogs)
        {
            Debug.Log($"[Emotion] Model asset: {onnxModel.name}");
            Debug.Log($"[Emotion] Backend: {backend}");
        }

        // ✅ 改动点 1：不再自己找设备/new WebCamTexture/Play
        camTex = (WebcamManager.Instance != null) ? WebcamManager.Instance.CamTex : null;
        if (camTex == null)
        {
            //Debug.LogError("[Emotion] WebcamManager.Instance.CamTex is null. 请确认场景里有 WebcamManager 且摄像头成功启动。");
            enabled = false;
            return;
        }

        if (enableLogs)
        {
            Debug.Log("[Emotion] Using shared webcam from WebcamManager.");
        }

        cpuTex = new Texture2D(inputSize, inputSize, TextureFormat.RGBA32, false);

        runtimeModel = Unity.InferenceEngine.ModelLoader.Load(onnxModel);

        worker = new Unity.InferenceEngine.Worker(runtimeModel, backend);

        if (enableLogs)
        {
            Debug.Log("[Emotion] Worker created successfully.");
            Debug.Log("[Emotion] Waiting for webcam to become ready...");
        }

        SetCubeColor(Emotion.Unknown, 0f);
    }

    void OnDestroy()
    {
        if (enableLogs) Debug.Log("[Emotion] OnDestroy()");

        // ✅ 改动点 2：camTex 是共享的，千万别 Stop/Destroy
        // if (camTex != null)
        // {
        //     camTex.Stop();
        //     Destroy(camTex);
        // }

        if (cpuTex != null) Destroy(cpuTex);

        worker?.Dispose();
    }

    void Update()
    {
        if (camTex == null || !camTex.isPlaying) return;

        // Wait until webcam is actually ready
        if (camTex.width <= 16 || camTex.height <= 16)
        {
            if (enableLogs && Time.frameCount % 60 == 0)
                Debug.Log($"[Emotion] Webcam not ready yet. width={camTex.width}, height={camTex.height}");
            return;
        }

        // One-time webcam ready log
        if (enableLogs && Time.frameCount == 1)
        {
            Debug.Log($"[Emotion] Webcam ready. Actual size: {camTex.width}x{camTex.height}, FPS={camTex.requestedFPS}");
        }

        timer += Time.deltaTime;
        if (timer < inferInterval) return;
        timer = 0f;

        inferCount++;
        bool doLog = enableLogs && (inferCount % Mathf.Max(1, logEveryN) == 0);

        if (doLog)
            Debug.Log($"[Emotion] Inference #{inferCount} (dt={inferInterval:0.00}s)");

        CopyWebcamToCPUTexture(camTex, cpuTex);

        using Unity.InferenceEngine.Tensor<float> input = BuildInputTensorNCHW(cpuTex);

        if (doLog)
        {
            var s = input.shape;
            Debug.Log($"[Emotion] Input tensor shape: {s} (expect 1x3x{inputSize}x{inputSize})");
        }

        // Run
        worker.SetInput("input", input);
        worker.Schedule();

        var output = worker.PeekOutput("logits") as Unity.InferenceEngine.Tensor<float>;
        if (output == null)
        {
            Debug.LogError("[Emotion] Output 'logits' not found or not Tensor<float>.");
            return;
        }

        float[] logits = output.DownloadToArray();

        if (doLog)
        {
            Debug.Log($"[Emotion] Logits length: {logits.Length}");
            Debug.Log($"[Emotion] Logits: {FormatArray(logits)}");
        }

        (Emotion emo, float conf, float[] probs) = ArgMaxSoftmaxWithProbs(logits);

        //if (doLog)
        //{
        //    //Debug.Log($"[Emotion] Probs: {FormatArray(probs)}");
        //    //Debug.Log($"[Emotion] Pred: {emo}  conf={conf:0.000}  threshold={unknownThreshold:0.000}");
        //}

        SetCubeColor(emo, conf);
    }

    static void CopyWebcamToCPUTexture(WebCamTexture src, Texture2D dst)
    {
        RenderTexture rt = RenderTexture.GetTemporary(dst.width, dst.height, 0, RenderTextureFormat.ARGB32);
        Graphics.Blit(src, rt);

        RenderTexture prev = RenderTexture.active;
        RenderTexture.active = rt;

        dst.ReadPixels(new Rect(0, 0, rt.width, rt.height), 0, 0, false);
        dst.Apply(false, false);

        RenderTexture.active = prev;
        RenderTexture.ReleaseTemporary(rt);
    }

    Unity.InferenceEngine.Tensor<float> BuildInputTensorNCHW(Texture2D tex)
    {
        Color32[] pixels = tex.GetPixels32();
        int H = tex.height;
        int W = tex.width;

        var shape = new Unity.InferenceEngine.TensorShape(1, 3, H, W);
        var t = new Unity.InferenceEngine.Tensor<float>(shape);

        for (int y = 0; y < H; y++)
        {
            int row = y * W;
            for (int x = 0; x < W; x++)
            {
                Color32 p = pixels[row + x];

                float r = (p.r / 255f - MEAN[0]) / STD[0];
                float g = (p.g / 255f - MEAN[1]) / STD[1];
                float b = (p.b / 255f - MEAN[2]) / STD[2];

                t[0, 0, y, x] = r;
                t[0, 1, y, x] = g;
                t[0, 2, y, x] = b;
            }
        }

        return t;
    }

    static (Emotion emo, float conf, float[] probs) ArgMaxSoftmaxWithProbs(float[] logits)
    {
        float max = logits[0];
        for (int i = 1; i < logits.Length; i++) max = Mathf.Max(max, logits[i]);

        float sum = 0f;
        float[] exp = new float[logits.Length];
        for (int i = 0; i < logits.Length; i++)
        {
            exp[i] = Mathf.Exp(logits[i] - max);
            sum += exp[i];
        }

        float inv = 1f / (sum + 1e-9f);
        float[] probs = new float[logits.Length];

        int arg = 0;
        float best = exp[0] * inv;
        probs[0] = best;

        for (int i = 1; i < exp.Length; i++)
        {
            float p = exp[i] * inv;
            probs[i] = p;
            if (p > best)
            {
                best = p;
                arg = i;
            }
        }

        Emotion emo = arg switch
        {
            0 => Emotion.Happy,
            1 => Emotion.Sad,
            2 => Emotion.Angry,
            3 => Emotion.Fear,
            _ => Emotion.Unknown
        };

        return (emo, best, probs);
    }
    void SetCubeColor(Emotion emo, float conf)
    {
        // 低置信度就判 Unknown（保留你原来的阈值逻辑）
        if (emo != Emotion.Unknown && conf < unknownThreshold)
            emo = Emotion.Unknown;

        Color c =
            (emo == Emotion.Happy || emo == Emotion.Unknown) ? new Color(1.0f, 0.4f, 0.7f, 1f) :
            (emo == Emotion.Sad) ? new Color(0.2f, 0.4f, 1.0f, 1f) :
            (emo == Emotion.Angry) ? new Color(1.0f, 0.15f, 0.15f, 1f) :
            (emo == Emotion.Fear) ? Color.black :
            new Color(1.0f, 0.4f, 0.7f, 1f); // 理论不会走到，但给个兜底

        cubeRenderer.material.color = c;
    }

    //void SetCubeColor(Emotion emo, float conf)
    //{
    //    Emotion before = emo;

    //    if (emo != Emotion.Unknown && conf < unknownThreshold)
    //        emo = Emotion.Unknown;
    //     Color happyColor = new Color(1.0f, 0.4f, 0.7f, 1f);
    //    Color c =
    //        (emo == Emotion.Happy) ? new happyColor :
    //        (emo == Emotion.Sad) ? new Color(0.2f, 0.4f, 1.0f, 1f) :
    //        (emo == Emotion.Angry) ? new Color(1.0f, 0.15f, 0.15f, 1f) :
    //        (emo == Emotion.Fear) ? Color.black :
    //                                 happyColor;

    //    cubeRenderer.material.color = c;

    //    if (enableLogs && (inferCount % Mathf.Max(1, logEveryN) == 0))
    //    {
    //        //Debug.Log($"[Emotion] SetCubeColor: pred={before}, conf={conf:0.000} -> used={emo}, color={c}");
    //    }
    //}

    static string FormatArray(float[] a)
    {
        if (a == null) return "null";
        int n = a.Length;
        if (n == 0) return "[]";

        // Print up to 8 values to avoid spamming console
        int m = Mathf.Min(n, 8);
        string s = "[";
        for (int i = 0; i < m; i++)
        {
            s += a[i].ToString("0.000");
            if (i < m - 1) s += ", ";
        }
        if (n > m) s += ", ...";
        s += "]";
        return s;
    }
}
