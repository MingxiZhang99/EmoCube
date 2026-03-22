//using System.Diagnostics;
using UnityEngine;

public class WebcamManager : MonoBehaviour
{
    public static WebcamManager Instance { get; private set; }
    public WebCamTexture CamTex { get; private set; }

    public int width = 1280;
    public int height = 720;
    public int fps = 30;

    void Awake()
    {
        if (Instance != null) { Destroy(gameObject); return; }
        Instance = this;

        var devices = WebCamTexture.devices;
        if (devices == null || devices.Length == 0)
        {
            Debug.LogError("没找到摄像头（权限/设备问题）");
            return;
        }

        CamTex = new WebCamTexture(devices[0].name, width, height, fps);
        CamTex.Play();
        Debug.Log("摄像头启动: " + devices[0].name);
    }
}
