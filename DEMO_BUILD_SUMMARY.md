# LipFD Demo 搭建信息总结

## 1. 当前目标

当前 demo 用于展示 LipFD / 轻量化 LipFD 的视频检测系统雏形。由于真实模型仍在训练中，第一阶段先完成前后端系统壳、上传交互、mock 检测流程和后续真实模型接入点。

整体展示逻辑：

```text
上传视频
  -> 后端接收
  -> mock 检测 / 后续真实 LipFD 检测
  -> 前端展示 Real/Fake、fake probability、延迟、FPS、窗口时间轴
```

## 2. 远端部署位置

远端服务器项目目录：

```text
/root/lx/LipFD/web
```

当前已同步到远端的结构：

```text
/root/lx/LipFD/web/
  README.md
  backend/
    main.py
    requirements.txt
    services/
      __init__.py
      detector.py
      mock_detector.py
  frontend/
    package.json
    index.html
    tsconfig.json
    tsconfig.node.json
    vite.config.ts
    src/
      App.vue
      main.ts
      api/
        client.ts
        types.ts
      components/
        MetricTile.vue
        ScoreChart.vue
      styles/
        main.css
```

本地也保留了一份同样的 scaffold：

```text
G:\保研资料\论文复现\web
```

## 3. 前端信息

前端技术栈：

```text
Vue 3
Vite
TypeScript
lucide-vue-next
```

设计方向：

```text
浅灰白背景
冷灰信息面板
浅色科研控制台
简约、大气、克制
避免紫蓝渐变、发光球、机器人图标、过度 AI 感
```

已完成的前端功能：

```text
视频上传
模型选择
开始检测按钮
视频预览
检测结果面板
fake probability 进度条
Latency / FPS / Windows / File size 指标
时间轴分数曲线
后端异常提示
```

当前前端默认请求：

```text
POST /api/detect/mock
GET  /api/models
```

Vite 开发服务器配置：

```text
前端端口：5173
后端代理：/api -> http://127.0.0.1:8000
```

## 4. 后端信息

后端技术栈：

```text
FastAPI
Uvicorn
python-multipart
```

已完成接口：

```text
GET  /api/health
GET  /api/models
POST /api/detect/mock
POST /api/detect
```

当前 `/api/detect/mock` 会返回模拟检测结果，包括：

```text
label
fakeProbability
confidence
latencyMs
fps
windows
fileSizeMb
timeline
```

`/api/detect` 当前也暂时转到 mock detector，后续真实模型接入时主要替换：

```text
/root/lx/LipFD/web/backend/services/detector.py
```

## 5. 远端运行方式

启动后端：

```bash
cd /root/lx/LipFD/web/backend
pip install -r requirements.txt
uvicorn main:app --host 127.0.0.1 --port 8000 --reload
```

启动前端：

```bash
cd /root/lx/LipFD/web/frontend
npm install
npm run dev -- --host 127.0.0.1 --port 5173
```

本地浏览器查看前端时，使用 SSH 端口转发：

```powershell
ssh -p 12123 -L 5173:127.0.0.1:5173 -L 8000:127.0.0.1:8000 root@172.28.7.26
```

然后本地打开：

```text
http://127.0.0.1:5173
```

## 6. 当前尚未启动的部分

目前只完成了代码 scaffold 和远端同步，没有在远端执行依赖安装或启动服务。

原因：

```text
模型仍在训练中
避免影响训练环境
前端后端依赖安装可能需要额外网络和时间
```

## 7. 输入模块推荐方案

虽然模型还在训练，可以先把输入预处理模块做完。推荐先做“上传视频文件的准实时检测”，暂不做摄像头真流式。

推荐流程：

```text
上传视频
  -> 保存临时文件
  -> 读取视频 fps / frame_count / duration
  -> 提取音频 wav
  -> 生成整段音频 mel 频谱
  -> 每隔 interval 秒抽取一个检测窗口
  -> 每个窗口取连续 5 帧
  -> 按帧位置截取对应 mel 片段
  -> 拼成 LipFD 输入图
  -> 当前接 mock detector
  -> 后续接真实模型 detector
```

示例：

```text
interval = 0.5 秒
window_len = 5 帧
视频 fps = 25

第 0.0 秒：frame 0-4
第 0.5 秒：frame 12-16
第 1.0 秒：frame 25-29
第 1.5 秒：frame 37-41
```

重要原则：

```text
音频和视频必须按同一时间轴对齐
不要随意拆分后各自抽样
不要修改官方输入语义
```

尤其是音频读取，继续保持官方语义：

```python
librosa.load(audio_file)
```

不要改为：

```python
librosa.load(audio_file, sr=None)
```

否则可能破坏 mel 与视频帧之间的映射关系。

## 8. 输入模块建议新增接口

下一步可以在 FastAPI 中增加：

```text
POST /api/input/prepare
GET  /api/input/jobs/{job_id}
GET  /api/input/jobs/{job_id}/windows/{index}
```

`POST /api/input/prepare` 返回示例：

```json
{
  "jobId": "demo-job",
  "filename": "sample.mp4",
  "duration": 12.4,
  "fps": 25,
  "frameCount": 310,
  "windows": 24,
  "intervalSec": 0.5
}
```

窗口信息示例：

```json
[
  {
    "index": 0,
    "time": 0.0,
    "startFrame": 0,
    "endFrame": 4,
    "sampleImage": "/api/input/jobs/demo-job/windows/0"
  }
]
```

## 9. 后续真实模型接入点

真实模型接入时主要补齐：

```text
backend/services/input_pipeline.py
backend/services/detector.py
```

`input_pipeline.py` 负责：

```text
视频保存
音频提取
mel 生成
视频帧采样
LipFD 输入图生成
窗口 metadata 返回
```

`detector.py` 负责：

```text
加载原始 LipFD 或轻量化 LipFD checkpoint
读取预处理窗口
执行推理
返回 score / label / latency
```

## 10. 当前状态一句话总结

当前 demo 已完成“Vue 3 + Vite + TypeScript 前端壳”和“FastAPI 后端 mock 检测壳”，并已同步到远端 `/root/lx/LipFD/web`。下一步最适合先做视频输入预处理模块，使系统具备真实上传、拆分、采样、生成 LipFD 输入窗口的能力。

## 11. 2026-05-31 真实最佳模块接入与测试

已将 `/api/detect` 从 mock hook 切换为真实 LipFD-Light 最佳模块：

```text
model id: lipfd-light-best
CLIP: ViT-B/32
Region Awareness: ResNet18
checkpoint: /root/lx/LipFD/lightweight/results/checkpoints/region_resnet18_clip_b32_ra0p01/best.pth
preprocess_device: gpu
pipeline: lightweight/scripts/benchmark_video_pipeline.py
```

前端变化：

```text
默认请求 POST /api/detect
默认模型 lipfd-light-best
新增 paired WAV 上传入口
保留 /api/detect/mock 作为 UI-only 演示兜底接口
```

后端变化：

```text
web/backend/services/detector.py
  -> 保存上传的视频和 paired wav
  -> 调用 benchmark_video_pipeline.py --preprocess_device gpu
  -> 读取 result.json 和 scores.csv
  -> 转换为前端已有 DetectionResult 响应结构
```

音频处理更新：

```text
远端 apt/conda 安装 ffmpeg 曾因依赖冲突/下载中断失败。
已在 lips 环境中安装 imageio-ffmpeg，提供独立 ffmpeg 可执行文件：
/root/anaconda3/envs/lips/lib/python3.10/site-packages/imageio_ffmpeg/binaries/ffmpeg-linux-x86_64-v7.0.2

当前 /api/detect 支持只上传视频：
1. 如果没有上传 paired wav，后端自动用 ffmpeg 从视频中抽取临时 wav；
2. 如果上传了 paired wav，则优先使用上传的 wav；
3. 后续仍由 librosa.load(wav) 按官方默认语义读取音频。
```

已完成测试：

```text
后端 Python 语法检查：通过
前端 npm install + npm run build：通过，已生成 dist
GET /api/health：通过
GET /api/models：通过，lipfd-light-best / region-resnet18 为 ready
POST /api/detect 传入 AVLips/0_real/0.mp4 + AVLips/wav/0_real/0.wav：通过
POST /api/detect 只传入 AVLips/0_real/0.mp4：通过，后端自动抽取 wav
```

真实接口 smoke test 返回：

```text
status: 200
label: fake
fakeProbability: 0.6412
windows: 10
fileSizeMb: 0.86
timeline_len: 10
first window score: 0.6784
```

只上传视频、自动抽音频 smoke test 返回：

```text
status: 200
label: fake
fakeProbability: 0.6530
windows: 10
fileSizeMb: 0.26
preprocessDevice: gpu
clip: ViT-B/32
backbone: resnet18
```

注意：

```text
当前 /api/detect 每次请求都会启动一个新的 Python 子进程并加载模型。
因此接口 smoke test 的 latencyMs 约 234.76 ms/window，不等价于 20 视频 benchmark 中的 66.8230 ms/window。
下一步系统化优化应将模型改成后端常驻加载，并把视频窗口送入常驻 detector，而不是每次请求重新加载模型。
```
