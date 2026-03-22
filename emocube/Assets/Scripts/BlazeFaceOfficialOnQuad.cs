using System;
using Unity.Mathematics;
using Unity.InferenceEngine;
using UnityEngine;
//using System.Diagnostics;

public class BlazeFaceOfficialOnQuad : MonoBehaviour
{
    [Header("Model")]
    public ModelAsset faceDetector;      // blaze_face_short_range.onnx 导入后得到的 ModelAsset
    public TextAsset anchorsCSV;         // anchors.csv
    public BackendType backend = BackendType.GPUCompute;

    [Header("Thresholds (same as official)")]
    public float iouThreshold = 0.30f;
    public float scoreThreshold = 0.50f;

    [Header("Run")]
    public float inferInterval = 0.05f;  // 推理间隔
    public bool enableLogs = false;

    const int k_NumAnchors = 896;
    const int k_NumKeypoints = 6;
    const int detectorInputSize = 128;

    float[,] m_Anchors;

    Worker m_Worker;
    Tensor<float> m_Input;   // (1,128,128,3)

    float m_Timer;

    // ---- Public outputs for drawer ----
    public bool HasFace { get; private set; }
    // 0..1, y=0 顶部 (xmin,ymin,w,h)
    public Rect FaceRect01 { get; private set; }
    // 6 points, 0..1, y=0 顶部
    public Vector2[] Keypoints01 { get; private set; } = new Vector2[k_NumKeypoints];

    // internal
    float2x3 m_M; // tensor->image affine matrix
    int m_LastNumFaces = 0;

    void Start()
    {
        if (faceDetector == null)
        {
            Debug.LogError("[BlazeFaceOfficial] faceDetector is null");
            enabled = false;
            return;
        }
        if (anchorsCSV == null)
        {
            Debug.LogError("[BlazeFaceOfficial] anchorsCSV is null");
            enabled = false;
            return;
        }

        // load anchors
        m_Anchors = BlazeUtils.LoadAnchors(anchorsCSV.text, k_NumAnchors);

        // load model
        var model = ModelLoader.Load(faceDetector);

        // compile: 2*input-1 + NMSFiltering (official)
        var graph = new FunctionalGraph();
        var input = graph.AddInput(model, 0);
        var outputs = Functional.Forward(model, 2 * input - 1); // IMPORTANT: official uses -1..1
        var boxes = outputs[0];  // (1,896,16)
        var scores = outputs[1]; // (1,896,1)

        // anchors constant tensor (896,4)
        var anchorsData = new float[k_NumAnchors * 4];
        Buffer.BlockCopy(m_Anchors, 0, anchorsData, 0, anchorsData.Length * sizeof(float));
        var anchors = Functional.Constant(new TensorShape(k_NumAnchors, 4), anchorsData);

        // NMS in graph
        var idx_scores_boxes = BlazeUtils.NMSFiltering(boxes, scores, anchors, detectorInputSize, iouThreshold, scoreThreshold);

        // output0: selectedIndices (N)
        // output1: selectedScores (1,N,1)
        // output2: selectedBoxes  (1,N,16)
        model = graph.Compile(idx_scores_boxes.Item1, idx_scores_boxes.Item2, idx_scores_boxes.Item3);

        m_Worker = new Worker(model, backend);
        m_Input = new Tensor<float>(new TensorShape(1, detectorInputSize, detectorInputSize, 3));

        if (enableLogs)
            Debug.Log("[BlazeFaceOfficial] started. backend=" + backend);
    }

    void OnDestroy()
    {
        m_Worker?.Dispose();
        m_Input?.Dispose();
    }

    void Update()
    {
        var cam = (WebcamManager.Instance != null) ? WebcamManager.Instance.CamTex : null;
        if (cam == null || !cam.isPlaying) { HasFace = false; return; }
        if (cam.width <= 16 || cam.height <= 16) { HasFace = false; return; }

        m_Timer += Time.deltaTime;
        if (m_Timer < inferInterval) return;
        m_Timer = 0f;

        RunOnce(cam);
    }

    void RunOnce(Texture texture)
    {
        // build tensor->image affine matrix M (official)
        float texW = texture.width;
        float texH = texture.height;

        float size = Mathf.Max(texW, texH);
        float scale = size / (float)detectorInputSize;

        // M = Translation(0.5*((w,h)+(-size,size))) * Scale(scale,-scale)
        float2 trans = 0.5f * (new float2(texW, texH) + new float2(-size, size));
        m_M = BlazeUtils.mul(
            BlazeUtils.TranslationMatrix(trans),
            BlazeUtils.ScaleMatrix(new float2(scale, -scale))
        );

        // sample into input tensor (GPU compute)
        BlazeUtils.SampleImageAffine(texture, m_Input, m_M);

        // run
        m_Worker.Schedule(m_Input);

        // read outputs (sync readback; simplest for beginners)
        // out0: indices (N)
        var indicesT = m_Worker.PeekOutput(0) as Tensor<int>;
        var scoresT = m_Worker.PeekOutput(1) as Tensor<float>;
        var boxesT = m_Worker.PeekOutput(2) as Tensor<float>;

        if (indicesT == null || scoresT == null || boxesT == null)
        {
            Debug.LogError("[BlazeFaceOfficial] outputs missing (0/1/2)");
            HasFace = false;
            return;
        }

        int[] indices = indicesT.DownloadToArray();   // length = N
        float[] boxes = boxesT.DownloadToArray();     // length = 1*N*16
        // scores not strictly needed for drawing
        // float[] scores = scoresT.DownloadToArray();

        int numFaces = indices.Length;
        m_LastNumFaces = numFaces;

        if (numFaces <= 0)
        {
            HasFace = false;
            return;
        }

        // always take face0 only (you want only one box)
        int idx = indices[0];

        float2 anchorPos = detectorInputSize * new float2(m_Anchors[idx, 0], m_Anchors[idx, 1]);

        // boxesT shape is (1,N,16). We use i=0.
        // Flatten index: (0 * N + i) * 16 + c  => i*16 + c
        int base16 = 0 * 16;

        float dx = boxes[base16 + 0];
        float dy = boxes[base16 + 1];
        float w = boxes[base16 + 2];
        float h = boxes[base16 + 3];

        // center in tensor coords (px)
        float2 centerTensor = anchorPos + new float2(dx, dy);

        // map to image space (px)
        float2 centerImg = BlazeUtils.mul(m_M, centerTensor);

        // topRight to compute size (official)
        float2 topRightTensor = anchorPos + new float2(dx + 0.5f * w, dy + 0.5f * h);
        float2 topRightImg = BlazeUtils.mul(m_M, topRightTensor);

        float2 boxSizeImg = 2f * (topRightImg - centerImg); // (wPx, hPx)

        // rect in 0..1, y=0 top
        float xmin = (centerImg.x - 0.5f * boxSizeImg.x) / texW;
        float ymin = (centerImg.y - 0.5f * boxSizeImg.y) / texH;
        float ww = boxSizeImg.x / texW;
        float hh = boxSizeImg.y / texH;

        // clamp
        xmin = Mathf.Clamp01(xmin);
        ymin = Mathf.Clamp01(ymin);
        ww = Mathf.Clamp01(ww);
        hh = Mathf.Clamp01(hh);

        FaceRect01 = new Rect(xmin, ymin, ww, hh);

        // keypoints (6)
        for (int j = 0; j < k_NumKeypoints; j++)
        {
            float kx = boxes[base16 + 4 + 2 * j + 0];
            float ky = boxes[base16 + 4 + 2 * j + 1];

            float2 kpImg = BlazeUtils.mul(m_M, anchorPos + new float2(kx, ky));

            float u = kpImg.x / texW;
            float vTop = kpImg.y / texH;

            Keypoints01[j] = new Vector2(Mathf.Clamp01(u), Mathf.Clamp01(vTop));
        }

        HasFace = true;

        if (enableLogs)
        {
            Debug.Log($"[BlazeFaceOfficial] faces(NMS)={numFaces}, take1 rect={FaceRect01}");
        }
    }
}
