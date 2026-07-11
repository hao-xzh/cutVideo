# 离线 Word 黄标音频剪辑器

一款完全离线的 Windows/macOS 桌面工具。它读取带有 `发言人 HH:MM` 时间锚点的
DOCX，把黄色高亮文字对齐到音频，经过人工复核后一次性导出剪辑完成的 WAV、MP3、
项目记录和切点清单。

## 工作方式

1. 选择 MP3/WAV/M4A/AAC/FLAC 音频和对应 DOCX，执行预检。
2. 使用随安装包交付的 `paraformer-zh + fsmn-vad` 先识别整段音频的真实讲话和绝对
   时间戳，再把完整 Word 正文单调映射到真实语音。为消除长音频累计时间漂移，每个含黄标
   的段落会在粗定位窗口内再次识别，生成新的绝对字符时间；DOCX 时间只在真实语音完全无
   法定位某段时作为低置信搜索提示，不能限制或覆盖切点。`fa-zh` 再用两种左右短上下文做
   局部字符边界对齐；两次结果稳定且与局部真实识别时间戳接近时才合并边界，否则必须试听
   复核。ASR 映射在精确文字锚点之间允许中文同音字匹配（无声调拼音），但不会模糊匹配
   非中文文字；同一黄色范围内的稀疏命中若跨越超过 1.5 秒的音频空洞，也不会被合并成切点。
3. 高置信度切点可自动通过；单字/双字、上下文重复、局部覆盖不足或双模型边界差异
   超过 120 ms 的项目必须人工处理。
4. 对每个不确定片段可分别“听原句”“只听待删”“听剪后”，在约 5 ms/点的局部波形上
   拖动切点；波形可用空白拖拽、视野条或 Shift+滚轮前后移动，用滚轮或“缩小/放大”按
   指针位置缩放，也可输入毫秒数或使用 `±20 ms` 微调。点击“确认删除”或“保留此处”后自动
   前往下一项。试听文件统一生成 48 kHz 双声道 PCM，并直接由 Qt 音频输出播放，不依赖
   媒体解码插件。所有待复核项处理完前禁止导出。
5. 导出前再次核对音频/DOCX SHA-256 和实际 PCM 时间轴。四个输出先写入临时目录，全部
   成功后统一提交；中途失败会回滚旧文件。

默认输出：

```text
<原名>_剪辑完成.wav
<原名>_剪辑完成.mp3
<原名>.cutvideo.json
<原名>_切点.csv
```

项目会自动保存。再次选择相同文件时按 SHA-256 复用分析结果；文件移动后可在打开项目时
重新关联。对齐算法版本也写入项目；旧版“Word 时间主导”项目不会被静默复用，必须重新
分析。应用不登录、不上传、不遥测，运行时不会访问网络。

## 样例回归

仓库测试会在样例文件存在时验证：60 个时间锚点、38 个黄色范围、208 个黄标字符、15 个
一至二字范围。音频时长只采用 FFmpeg 解码后的 PCM 样本数；当前机器对样例得到
`120,374,411 / 48,000 = 2507.800229 s`，不采用 Windows 文件属性显示的错误时长。
真实回归文件不进入 Git；需要复跑时分别设置 `CUTVIDEO_SAMPLE_AUDIO` 与
`CUTVIDEO_SAMPLE_DOCX` 环境变量。

当前“整段粗定位 → 每处短音频重识别 → 双上下文字符边界”真实模型回归中，38 项均成功
使用音频文字定位，没有一项回退到 Word 时间，也没有局部重识别失败；基于安全策略，38 项
仍全部进入人工复核。尚未建立 38 处人工边界金标，因此不能把自动切点精度声明为已经通过
`±100 ms` 验收。

## 开发运行

需要正式版 Python 3.11：

```powershell
py -3.11 -m venv .venv311
.\.venv311\Scripts\python -m pip install -c constraints-release.txt -e ".[dev,ml]"
.\.venv311\Scripts\python -m cutvideo
```

模型和二进制默认位于：

```text
resources/
  bin/windows-x86_64/{ffmpeg.exe,ffprobe.exe}
  bin/macos-arm64/{ffmpeg,ffprobe}
  models/fa-zh/
  models/paraformer-zh/
  models/fsmn-vad/
```

可用 `CUTVIDEO_RESOURCE_ROOT`、`CUTVIDEO_FFMPEG`、`CUTVIDEO_FFPROBE`、
`CUTVIDEO_MODEL_ROOT` 覆盖开发路径。应用只接受本地模型目录。

如模型资源缺失，可在接受对应模型许可证后执行：

```powershell
.\.venv311\Scripts\python scripts\fetch_models.py --accept-model-licenses
.\.venv311\Scripts\python scripts\verify_resources.py
```

## Windows x64 打包

仓库已带经哈希固定的 LGPL FFmpeg 和本地模型：

```powershell
powershell -ExecutionPolicy Bypass -File scripts\build_windows.ps1
```

产物位于 `dist/CutVideo/`。必须分发整个目录，不能只复制 `CutVideo.exe`；模型、
FFmpeg、许可证和 Qt 运行库都在同一目录树内。FFmpeg/FFprobe 子进程使用 Windows
无窗口模式运行，不会反复闪现 CMD。正式 MSI/签名可在此独立目录基础上完成。

## Apple Silicon macOS 打包

GitHub 仓库不保存约 1.06 GB 的模型权重或平台 FFmpeg 二进制。必须在 macOS 13+
Apple Silicon 机器上原生执行；下面一条命令会安装缺少的 Homebrew 依赖、建立 Python
3.11 环境、按固定提交下载模型、构建 LGPL FFmpeg，并生成应用与 DMG：

```bash
git clone https://github.com/hao-xzh/cutVideo.git
cd cutVideo
bash scripts/bootstrap_macos.sh --accept-model-licenses
```

产物为 `dist/CutVideo.app` 与 `dist/CutVideo.dmg`。首次准备需要下载约 1 GB 模型并编译
FFmpeg，请预留至少 12 GB 可用空间。重复构建可复用已下载模型和 FFmpeg 构建目录。
发布依赖通过 `constraints-release.txt` 固定；脚本同时检查 Python 与产物必须为原生 arm64，
并设置 `MACOSX_DEPLOYMENT_TARGET=13.0`。macOS 13 兼容性仍应在实际 13.x 机器上复验。

脚本会调用 `scripts/build_macos_ffmpeg.sh`，从
[FFmpeg 8.1.2 官方源码](https://ffmpeg.org/download.html)构建不启用 GPL/nonfree 组件的
arm64 版本；这一步需要 Xcode Command Line Tools 和 Homebrew，并会安装 `pkg-config`、
LGPL `lame`。构建完成后会写入二进制 SHA-256，再继续生成 `dist/CutVideo.dmg`。

设置 `APPLE_CODESIGN_IDENTITY` 可注入正式签名；未设置时使用 ad-hoc 签名。Developer ID
公证不在首版自动流程内。由于 Windows 无法完成 Apple 原生链接、签名和音频设备测试，
Mac 产物仍需在目标机器上断网执行完整验收。

## 验证

```powershell
.\.venv311\Scripts\python -m pytest -q
.\.venv311\Scripts\python -m ruff check src tests scripts
.\.venv311\Scripts\python scripts\verify_resources.py
.\.venv311\Scripts\python -m pip check
```

测试覆盖 DOCX OOXML、项目校验、模型结果适配、实际 PCM 时间轴、切点收缩、试听、
96 kHz/多声道输入、双格式导出、源文件替换拒绝和四文件提交回滚。

打包脚本还会启动冻结后的程序执行 `--self-test-models`：从发布目录发现 FFmpeg，分别运行
`fa-zh` 与 `paraformer-zh + fsmn-vad` 的随包示例推理，并验证 PCM 试听后端可导入。任一
模型、FFmpeg、Qt 音频后端缺失或冻结依赖不完整都会直接让构建失败。

## 输入约定与限制

- 时间锚点必须是独立段落且严格匹配 `发言人 HH:MM`；下一正文段落属于该锚点。
- 只处理 `w:highlight="yellow"`，其他颜色不会删除。
- 首版只处理单组音频和 DOCX，不处理视频、批量任务、Intel Mac 或自动更新。
- WAV 保持源采样率和声道，PCM 16-bit；MP3 默认 192 kbps，在源参数不被 MP3 支持时
  选择兼容采样率/声道。不会做响度归一化。

## 许可证

应用自身与捆绑组件许可证分开。发布目录包含 `THIRD_PARTY_NOTICES.md`、模型许可证、
FFmpeg 构建配置和源码地址。不要用带 GPL/nonfree 组件的 FFmpeg 构建替换发布资源，
除非重新评估整个分发方式。Qt/PySide6 部署参考
[Qt for Python 官方文档](https://doc.qt.io/qtforpython-6/deployment/index.html)。
