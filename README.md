# 离线音频剪辑器

一款模型准备完成后完全离线运行的 Windows/macOS 桌面工具，包含两个互不影响的工作区：

- **Word 黄标剪辑**：读取带有 `发言人 HH:MM` 时间锚点的 DOCX；识别到的真实文字会
  从音频开头持续显示，并用同一批时间戳把每一处黄色高亮文字对齐到连续口播区间。未标记正文
  只用于上下文定位；所有黄标无论算法置信度高低，都必须人工确认删除或保留后才能导出。
- **音频处理**：只导入一段音频，连续识别并流式显示带时间戳文字；用户选择文字或框选波形建立
  “删除”标注，试听与精调后导出处理完成的音频。

## 工作方式

1. 将 MP3/WAV/M4A/AAC/FLAC 音频和对应 DOCX 直接拖入输入框（也可使用文件选择器），
   执行预检。文件选择器会记住两个工作区最近使用的目录。
   时间锚点支持 `发言人 12:34`、`发言人 张三 12:34` 和
   `发言人 张三 01:12:34` 三类常用写法。
2. Apple 芯片 Mac 使用固定版本的 `Qwen3-ASR 0.6B 4-bit + Qwen3 Forced Aligner
   0.6B 4-bit`，通过 MLX/Metal GPU 从前到后连续识别；模型默认保存在
   `~/Library/Application Support/CutVideo/models`；已有模型会原样保留，旧版 App
   内嵌模型会在更新安装前迁移到只读的 `/Library/Application Support/CutVideo/models`，
   只有两个位置都确实缺失或校验失败时才联网下载。Windows 保留离线
   `paraformer-zh + fsmn-vad + fa-zh` 路径。真实文字会随识别进度持续显示；同一批绝对
   时间戳再通过“唯一锚点 + VAD/段落约束的单调 DP”把完整 Word 正文映射到真实语音；
   重复句会比较前两名路径差距，差距不足时强制复核。为消除长音频累计时间漂移，每个含黄标
   的段落会在粗定位窗口内再次识别，生成新的绝对字符时间；DOCX 时间只在真实语音完全无
   法定位某段时作为低置信搜索提示，不能限制或覆盖切点。`fa-zh` 再用两种左右短上下文做
   局部字符边界对齐；两次结果稳定且与局部真实识别时间戳接近时才合并边界，否则必须试听
   复核。Word 映射允许口播在文稿中间插入少量“呃、嗯、啊”等语气词，也允许文稿存在少量
   漏读、复读或同音差异；黄色范围取其前后可靠锚点之间的连续口播区间，因此中间实际说出的
   语气词也包含在试听/删除范围内。重复短句必须结合左右正文和全局单调顺序消歧；无法唯一
   确定时会明确标为风险。非中文文字不会做宽松模糊匹配；同一黄色范围内的稀疏命中若跨越
   超过 1.5 秒的音频空洞，也不会被合并成切点。
   模型只有词级或句级时间时，内部字符共享原始粗粒度区间，不会平均伪造逐字切点。
3. 时间戳单位、范围、单调性、字符覆盖和相邻边界会先经过程序校验；单字/双字、上下文重复、
   粗时间戳、置信度缺失、局部覆盖不足或双模型差异会额外显示风险原因。但程序不自动作出
   删除决定：每一处黄色标记都进入列表，必须人工“确认删除”或“保留此处”。
4. 对每个不确定片段可分别“听原句”“只听待删”“听剪后”，在约 5 ms/点的局部波形上
   拖动切点；波形可用空白拖拽、视野条或 Shift+滚轮前后移动，用滚轮或“缩小/放大”按
   指针位置缩放，也可输入毫秒数或使用 `±20 ms` 微调。点击“确认删除”或“保留此处”后自动
   前往下一项。试听文件统一生成 48 kHz 双声道 PCM，并直接由 Qt 音频输出播放，不依赖
   媒体解码插件。自动切口依次经过语音起止点、相邻字符保护、最多 20 ms 有限外扩和严格
   零交叉处理；发生外扩会保留风险提示。所有黄标完成确认前禁止导出。
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
分析。应用不登录、不上传音频或文稿、不遥测；只有 Apple 芯片 Mac 缺少本地模型时，才会
从版本化发布资源下载固定模型压缩包。下载完成并通过 SHA-256 校验后，识别和剪辑均可断网运行。

在 Apple 芯片 Mac 上，Qwen 识别和强制对齐固定使用 MLX/Metal GPU；正式包不会在 GPU
初始化失败后悄悄退回慢速 CPU 并继续给出看似正常的结果。模型采用 0.6B 4-bit，以控制
16 GB 机器上的峰值占用和温度，同时保留完整长音频的顺序处理能力。

## 音频处理工作区

1. 拖入单独音频或使用文件选择器并执行“连续识别”，真实文字会从前到后持续显示；首批结果
   出现后会持续追加，直到整段完成。完整转写保留 Qwen 原始标点作为不可见的语义边界，
   优先在完整句末换行，只在超长无句末口播中使用长停顿或最长 60 秒安全网；每行开头显示
   该段的音频开始时间。旧项目没有保存原始标点时会先用长停顿改善布局，重新识别后才能启用完整语义分段。
2. 流式文字出现后，Qwen Forced Aligner 会继续在后台完成全文逐字边界；这期间文字保持可见，
   但按文字试听和按文字删除暂时禁用，防止把识别草稿的均匀时间分配误当成真实切点。精确对齐
   成功后，可在完整转写中拖选文字，再通过按钮、右键、`Delete/Backspace`、`Ctrl+B` 或
   macOS `⌘B` 建立删除标注；重叠标注会安全合并。
3. 选择文字或点击已有标注时，文字范围、波形和毫秒输入框同步。波形支持直接框选、拖动
   两侧边界、滚轮缩放以及 `Alt+拖动` 平移。
4. 精确逐字时间戳仍只提供初始建议；右侧橙色时间范围才是最终实际删除范围。修复过的零时长、
   词/句级时间戳、
   置信度缺失或低置信结果会显示“需复核”，并阻止导出；试听后拖动边界、修改毫秒值或明确
   点击“确认边界”即可完成复核，结果自动保存。
5. 右侧统一提供“原音 ±3s”“删除段”“删除后”。前后对比均保留切点
   两侧各 3 秒上下文；建立删除标注后仍可反复试听和继续调整。所有预览都是本地 PCM，
   不依赖系统媒体解码插件。
6. 项目自动保存为 `<原名>.audioprocess.json`；默认导出：

```text
<原名>_处理完成.wav
<原名>_处理完成.mp3
<原名>.audioprocess.json
```

当前标注操作只有“删除”；项目结构已为后续扩展其他操作保留独立的 `operation` 字段。
播放、预览和分析共用固定高度的任务区，进度条出现或消失不会推动候选列表、波形或导出区；
两个工作区的选择、分析、试听、保存和导出状态都会在当前操作区就近反馈。

## 样例回归

仓库测试会在样例文件存在时验证：60 个时间锚点、38 个黄色范围、208 个黄标字符、15 个
一至二字范围。音频时长只采用 FFmpeg 解码后的 PCM 样本数；当前机器对样例得到
`120,374,411 / 48,000 = 2507.800229 s`，不采用 Windows 文件属性显示的错误时长。
真实回归文件不进入 Git；需要复跑时分别设置 `CUTVIDEO_SAMPLE_AUDIO` 与
`CUTVIDEO_SAMPLE_DOCX` 环境变量。

上一版“整段粗定位 → 每处短音频重识别 → 双上下文字符边界”真实模型基线中，38 项均成功
使用音频文字定位，没有一项回退到 Word 时间，也没有局部重识别失败；基于安全策略，38 项
仍全部进入人工复核。本版已更换为单调 DP、候选差距与新边界策略，旧项目会要求重新分析；
尚未建立 38 处人工边界金标，因此不能把自动切点精度声明为已经通过 `±100 ms` 验收。

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
  models/qwen3-asr-0.6b-4bit/
  models/qwen3-forced-aligner-0.6b-4bit/
  selftest/asr_example.wav
```

可用 `CUTVIDEO_RESOURCE_ROOT`、`CUTVIDEO_FFMPEG`、`CUTVIDEO_FFPROBE`、
`CUTVIDEO_MODEL_ROOT` 覆盖开发路径。应用只把校验完成的本地模型目录交给识别后端；自动下载
仅用于恢复缺失的固定模型，不会让模型库在推理时自行联网。

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

Git 仓库不保存约 1.06 GB 的模型权重或平台 FFmpeg 二进制。必须在 macOS 13+
Apple Silicon 机器上原生执行；下面一条命令会安装缺少的 Homebrew 依赖、建立 Python
3.11 环境、按固定提交下载模型、构建 LGPL FFmpeg，并生成应用与 DMG：

```bash
git clone https://github.com/hao-xzh/cutVideo.git
cd cutVideo
bash scripts/bootstrap_macos.sh --accept-model-licenses
```

产物为 `dist/CutVideo.pkg` 与 `dist/CutVideo.dmg`；DMG 内只有一个代码安装包，不包含
模型权重。构建阶段仍会用本机固定模型对冻结后的 App 做真实推理自检，因此首次构建需要
下载约 1 GB 模型并编译 FFmpeg，请预留至少 12 GB 可用空间。重复构建可复用已下载模型和
FFmpeg 构建目录。安装包会在覆盖旧版 App 前，把旧包内两套模型原样迁移到用户的
系统级 Application Support；新用户或本地模型损坏时，应用启动后才下载两个固定压缩包到
用户 Application Support，支持断点续传，并依次校验压缩包、权重文件和完整模型目录
SHA-256。
发布依赖通过 `constraints-release.txt` 固定；脚本同时检查 Python 与产物必须为原生 arm64，
并设置 `MACOSX_DEPLOYMENT_TARGET=15.0`。构建会遍历应用内全部 Mach-O，任何二进制的
最低系统版本高于 15.0 或不是纯 arm64 都会让构建失败；最终仍应在实际 macOS 15 机器上复验。

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

测试覆盖 DOCX OOXML、两类项目校验、完整转写 token、文字删除标注、模型结果适配、实际 PCM 时间轴、切点收缩、试听、
96 kHz/多声道输入、双格式导出、源文件替换拒绝和四文件提交回滚。

打包脚本还会启动冻结后的程序执行 `--self-test-models`：从代码包发现 FFmpeg，并通过外置
模型目录真实运行 Qwen ASR 与 Forced Aligner 的随包示例推理，同时验证 PCM 试听后端可导入。
脚本还会拒绝任何包含 `.safetensors`/`.onnx` 权重的 macOS App；任一模型、FFmpeg、Qt
音频后端缺失或冻结依赖不完整都会直接让构建失败。

## 输入约定与限制

- 时间锚点必须是独立段落，支持 `发言人 MM:SS`、`发言人 姓名 MM:SS` 和
  `发言人 姓名 HH:MM:SS`；下一正文段落属于该锚点。
- 只处理 `w:highlight="yellow"`，其他颜色不会删除。
- 两个工作区均只处理单组任务，不处理视频、批量任务、Intel Mac 或自动更新。
- WAV 保持源采样率和声道，PCM 16-bit；MP3 默认 192 kbps，在源参数不被 MP3 支持时
  选择兼容采样率/声道。不会做响度归一化。

## 许可证

应用自身与捆绑组件许可证分开。发布目录包含 `THIRD_PARTY_NOTICES.md`、模型许可证、
FFmpeg 构建配置和源码地址。不要用带 GPL/nonfree 组件的 FFmpeg 构建替换发布资源，
除非重新评估整个分发方式。Qt/PySide6 部署参考
[Qt for Python 官方文档](https://doc.qt.io/qtforpython-6/deployment/index.html)。
