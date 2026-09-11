# XYQQuiz

XYQQuiz 是一个在 Windows 本机运行的《梦幻西游》答题辅助显示工具，支持科举和教师节“看图说话”。它通过 Windows Graphics Capture 读取游戏窗口，在独立桌面窗口中显示实时预览，并根据题库和当前选项框出候选答案。

> 本项目不会自动点击、不会向游戏发送输入、不会读取游戏进程内存，也不会注入游戏或在游戏窗口内绘制。请自行确认并遵守游戏规则。

## 功能

- 自动识别科举和教师节活动，无需手动切换。
- 教师节内置 335 条可用官方图标和 32 条独立补充记录，共 367 条（已排除一张错误的佛法无边原图）；所有来源均兼容薄边框和装饰边框，官方更新不会覆盖补充库。
- 图标证据充分、选项文字仅有一处识别差异且明显最接近时，显示红色虚线候选框；相近选项无法拉开差距时继续重试。
- 新题最多允许一个选项不可读，仅在图标能与可读答案精确、唯一且高可信对应时框选；已识别同题可容忍一个选项中的小范围彩色鼠标遮挡。
- 同图换选项会重新映射答案；切题、关闭答题框和题库更新会清除旧提示。
- 教师节优先适配 1024×768 及以上游戏画面。四张真实截图和缩放/移动回放已验证，实际活动连续答题仍需实机验收。
- 自动寻找 `mhtab.exe` / `MHXYMainFrame` 游戏窗口并显示本地实时预览。
- 先独立识别题目，再定位 3～4 个选项，使用内置离线科举题库匹配答案。
- 布局分析按分辨率自适应缩放，OCR 使用原始画面，软件预览按性能模式缩放。
- 题目 OCR 按精确区域、扩展区域和面板题目带逐级容错；题面不可信时不会盲目识别选项。
- 通过时间连续性保留同题状态、短暂容忍布局抖动，并对候选结果定时重试。
- 首次正常运行按硬件自动分档，优先低配；标准配置预览最高 30 FPS，低配最高 10 FPS。识别使用独立节拍，低配扫描最高 5 FPS，无题时扫描最高 5 Hz。
- 桌面硬件预览通过原生辅助程序绘制 GPU 纹理；软件预览通过 WebSocket 传送 I420 平面帧。暂停预览、最小化或隐藏页面会减少预览开销，识别继续运行。
- OCR 与预览独立选择后端，默认自动显卡优先；DirectML 设备完成真实 OCR 自检后才可手动选择，失败回退 CPU 并保留设置。
- CPU/GPU 均使用统一红色边框：高可信结果为实线，候选为虚线，框旁不显示等级和评分；侧栏保留诊断数据，评分不是正确概率。
- 支持本地补题和答案修正，本地数据与官方题库分开保存在 `user-data\questions.json`。
- 支持单实例启动、端口冲突提示、题库原子更新和一键退出。
- 可按需保存识别诊断或不含游戏画面的环境诊断。

当前版本为 `0.5.4`，包含新增补充图标与旧库增量合并、性能记录、标准模式动态取帧及首次初始化修复，保留硬件自动分档、低配模式和统一框选样式，详见 [0.5.4 更新说明](docs/releases/v0.5.4.md)。Windows 11 x64 已验证；Windows 10 1903 及以上 x64 是目标兼容范围，但尚未完成实机验证。教师节基础验证范围见 [v0.4 验收记录](docs/v0.4-validation.md)，名称差异与未收录干扰项修复见 [v0.4.1 修复记录](docs/v0.4.1-feedback-fix.md)。

## 直接使用 Windows 便携版

1. 获取 `XYQQuiz-v0.5.4-win10-win11-x64.zip` 和同名 `.sha256`。
2. 完整解压到一个新目录，不要直接在压缩包里运行。
3. 双击 `XYQQuiz.exe`，首次捕获时允许 UAC 管理员权限请求。
4. 等待默认 `1440×900` 的可缩放桌面窗口打开；游戏题面出现后，答案框会显示在窗口预览中。
5. 关闭桌面窗口或使用“退出程序”结束后台；外部浏览器模式请使用“退出程序”。

第二次双击 EXE 会还原并聚焦已经运行的桌面窗口。若页面提示会话失效，也请重新双击 EXE，不要手工拼接本地 URL。

便携包自带程序、OCR 模型、布局和离线题库，正常启动和识别不需要联网。“更新题库”是唯一会主动访问题库来源的日常功能。

“更新题库”分别更新科举和教师节，并显示各自结果。教师节更新会先完整下载、校验所有图标再切换版本，失败时保留旧题库。旧配置或自定义数据目录首次运行 v0.4 时会自动补齐缺失的教师节资源；本地补题继续用于科举。

桌面窗口使用系统中的 Microsoft Edge WebView2 Evergreen Runtime，并在 `127.0.0.1` 上选择随机空闲端口。发布包不会捆绑体积较大的 Fixed Version Runtime；如果系统缺少 WebView2，程序会自动退回外部浏览器模式，识别功能仍可使用。也可通过 `XYQQuiz.exe --external-browser` 主动使用外部浏览器进行调试；该模式使用 `config.web.port`（默认 `8765`），端口被占用时会给出明确提示。

## 自检与诊断

当前版本支持“开始性能记录 / 停止性能记录”和“导出耗时报告”：正常连续答题后导出逐题及各阶段的中位数、P95。记录不包含截图、题目或答案；退出前需导出。操作与测量边界见 [性能记录说明](docs/performance-recording.md)。

双击便携包中的 `一键自检.cmd`，报告会写入 `diagnostics\self-test-latest`。

- “保存识别诊断”包含当前完整游戏画面、题目/选项裁剪、识别状态和日志尾部。点击前会显示隐私确认；分享前仍应自行检查角色名、聊天和其他个人信息。
- “导出环境诊断”不包含游戏画面或题库正文，主要用于排查系统、配置和依赖问题。
- 诊断文件只写在本地 `diagnostics\`，程序不会自动上传。
- 本地补题只写入 `user-data\questions.json`。发布 ZIP 不预置该文件，诊断中也不会导出本地题目正文。

命令行自检：

```powershell
XYQQuiz.exe --version --report-dir .\diagnostics\version
XYQQuiz.exe --self-test --headless --report-dir .\diagnostics\self-test
```

## 从源码运行

需要 Windows x64 和 Python 3.11 或更高版本。

```powershell
py -3.11 -m venv .venv
.venv\Scripts\python.exe -m pip install --upgrade pip
.venv\Scripts\python.exe -m pip install -e ".[dev,release]"
Copy-Item config.example.json config.json
.venv\Scripts\xyq-quiz.exe --config config.json
```

默认桌面模式只监听 `127.0.0.1` 上的随机空闲端口；`--external-browser` 模式使用 `config.web.port`（默认 `8765`）。配置文件只允许本机回环地址，HTTP 与 WebSocket 接口还使用进程随机凭据、受控会话和严格 Host/Origin 校验。

`recognition.ocr_workers` 现在只允许 `1`，以保证只常驻一份 OCR 模型。旧版 `config.json` 若配置为其他值，启动时会提示迁移错误；请改为 `1` 或删除该项。

新配置默认使用 `capture.preview_fps: 30` 和 `recognition.scan_fps: 15`。若升级时复制了明确写有 `preview_fps: 15` 的旧 `config.json`，程序会尊重旧值；需要 30 FPS 预览时请将该项改为 `30`。

首次正常启动仅在至少 12 个逻辑线程、16 GiB 已安装内存且检测到硬件图形适配器时选标准模式，其余或读取失败时选低配。结果保存后不重复判断；旧版明确保存的高低配选择继续沿用。可以在性能设置中手动调整，重启生效。上述 FPS 配置在低配模式下还会受运行上限约束，硬件分档不改变显卡优先策略。

另一台电脑接手开发、资源目录及发布流程见 [开发交接说明](docs/development-handoff.md)。

性能设置中的“低配模式”将预览限制为最高 10 FPS、软件预览宽度最高 640 像素，识别独立以最高 5 FPS 检查新题，OCR 保留原图清晰度。保存后重启生效，关闭该模式可恢复原配置。“暂停预览”按钮立即停止显示画面，识别与答案文字继续更新；最小化或隐藏页面也会暂停预览处理。细节见 [低配模式](docs/low-resource-mode.md)。

## 测试

```powershell
.venv\Scripts\python.exe -m pytest -q
```

真实游戏截图包含第三方内容和可能的个人信息，因此不进入公开仓库。维护者可把本地 fixture manifest 和图片放在忽略目录中，再显式运行回归：

```powershell
.venv\Scripts\python.exe -m pytest tests\integration\test_recognition_fixtures.py -q `
  --recognition-manifest tests\fixtures\recognition\manifest.json `
  --recognition-layout data\layouts\keju-default.json `
  --recognition-layout data\layouts\keju-picture.json
```

## 构建便携包

开发验证构建：

```powershell
.venv\Scripts\python.exe -m pip install --require-hashes -r requirements-release.txt
.\scripts\build-release.ps1 -Version 0.5.4 -Commit working-tree -AllowDevelopmentCommit
```

正式发布构建必须从干净提交运行，并传入完整 40 位 Git SHA：

```powershell
.\scripts\build-release.ps1 -Version 0.5.4 -Commit (git rev-parse HEAD)
```

产物位于 `release\`，包括 ZIP 和 SHA-256 文件。构建脚本会审计公开树和最终 ZIP，拒绝打入 `user-data\`、`diagnostics\` 或本地 `questions.json`。GitHub 的 `v*` 标签工作流会先做公开内容审计和完整测试，再使用标签对应的真实提交 SHA 构建并创建 Release。

## 更新与升级

程序启动不会自动联网。界面中的“更新题库”分别获取网易科举数据和教师节官方图标，完整校验后原子切换；失败时保留当前可用题库。教师节独立补充库不受官方更新覆盖，更新题库不会升级程序。

升级程序时请解压到全新目录，再按需复制旧目录中的 `config.json`、`data`、`user-data`、`logs` 和 `diagnostics`。不要把新版 EXE 或 `_internal` 覆盖到旧目录。

## 已知限制

- Windows 10 1909（18363.657）已收到 `windows_capture.pyd / 0xC0000409` 原生捕获崩溃反馈，0.5.4 尚未修复；目标系统范围不代表所有系统和驱动组合已验收。
- 硬件分档已测试逻辑边界及本机读取，尚未完成多台真实低端机性能验证；线程数、内存和适配器检测不是跑分保证。
- 游戏 UI、字体、DPI、动画遮挡或极端分辨率变化可能导致识别失败。
- 题目不在题库、OCR 置信度不足，或正确答案无法和选项唯一匹配时，可能已显示 OCR 题目但不会画框；这是预期的安全降级。
- 乡试和殿试尚未完成覆盖全部界面的活动现场验证；当前发布承诺以已经实际验证的科举主流程为准。
- 首版没有代码签名，Windows SmartScreen 可能显示“未知发布者”。请只从项目 Release 下载并核对 SHA-256，不要全局关闭安全软件。

## 许可证与第三方内容

XYQQuiz 的原创项目代码依据 [PolyForm Noncommercial License 1.0.0](LICENSE) 提供。这是一份非商业源码许可：允许查看和非商业使用，也允许在非商业目的下修改、Fork 和重新发布；重新发布时必须遵守许可证中的通知要求。任何商业使用均不在本许可证授权范围内，必须事先取得版权方另行书面授权。

这不是 OSI 定义的开源许可证，请勿将本项目描述为“开源软件”。上述项目代码许可证不覆盖内置题库、游戏截图衍生的布局锚点、游戏素材、商标或其他第三方内容；这些内容的来源、各自权利和分发边界见 [THIRD_PARTY_NOTICES.txt](THIRD_PARTY_NOTICES.txt)。

XYQQuiz 是非官方项目，与网易没有关联，也未获得网易认可或背书。
