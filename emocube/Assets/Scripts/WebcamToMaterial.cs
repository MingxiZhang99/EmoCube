//using System.Diagnostics;
using UnityEngine;

public class WebcamToMaterial : MonoBehaviour
{
    public Renderer targetRenderer;

    void Start()
    {
        if (targetRenderer == null) targetRenderer = GetComponent<Renderer>();

        // 从 WebcamManager 获取同一个摄像头纹理（不要再 new / Play）
        var camTex = WebcamManager.Instance != null ? WebcamManager.Instance.CamTex : null;
        if (camTex == null)
        {
            Debug.LogError("没有拿到 WebcamManager 的 CamTex。请确认场景里有 WebcamManager 并且它 Awake 时成功启动了摄像头。");
            return;
        }

        // 关键：URP 通常用 _BaseMap；内置/Standard 通常用 _MainTex
        var mat = targetRenderer.material;

        bool setAny = false;
        if (mat.HasProperty("_BaseMap"))
        {
            mat.SetTexture("_BaseMap", camTex);
            setAny = true;
            Debug.Log("已设置材质 _BaseMap（URP）");
        }
        if (mat.HasProperty("_MainTex"))
        {
            mat.SetTexture("_MainTex", camTex);
            setAny = true;
            Debug.Log("已设置材质 _MainTex（Built-in）");
        }

        if (!setAny)
            Debug.LogError("材质没有 _BaseMap 或 _MainTex。请把材质 Shader 改成 URP/Unlit 或 Unlit/Texture。");
    }

    void Update()
    {
        var camTex = WebcamManager.Instance != null ? WebcamManager.Instance.CamTex : null;
        if (camTex == null) return;

        // 仅用于验证：打印一次帧信息
        if (camTex.didUpdateThisFrame)
        {
            Debug.Log($"帧更新: {camTex.width}x{camTex.height}, rot={camTex.videoRotationAngle}, mirror={camTex.videoVerticallyMirrored}");
            enabled = false; // 防止刷屏
        }
    }
}
