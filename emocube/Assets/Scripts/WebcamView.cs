//using System.Diagnostics;
using UnityEngine;
using UnityEngine.UI;

public class WebcamView : MonoBehaviour
{
    public RawImage rawImage;
    public AspectRatioFitter fitter;   // 可选：让画面不变形
    public int requestedWidth = 1280;
    public int requestedHeight = 720;

    WebCamTexture _webCamTex;

    void Start()
    {
        if (rawImage == null) rawImage = GetComponent<RawImage>();

        // 选第一台摄像头
        var devices = WebCamTexture.devices;
        if (devices == null || devices.Length == 0)
        {
            Debug.LogError("没找到摄像头设备");
            return;
        }

        _webCamTex = new WebCamTexture(devices[0].name, requestedWidth, requestedHeight, 30);
        rawImage.texture = _webCamTex;
        _webCamTex.Play();
    }

    void Update()
    {
        if (_webCamTex == null) return;

        // 有些设备需要旋转/镜像处理，这里先只做比例适配（不做旋转）
        if (fitter != null && _webCamTex.width > 16)
        {
            fitter.aspectRatio = (float)_webCamTex.width / _webCamTex.height;
        }
    }

    public WebCamTexture GetTexture() => _webCamTex;
}
