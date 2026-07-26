# XYQQuiz

XYQQuiz 是一个在 Windows 本机运行的《梦幻西游》科举答题辅助显示工具。它通过 Windows Graphics Capture 读取游戏窗口，在独立桌面窗口中显示实时预览，并根据题目、题库答案和选项的综合评分框出候选答案。

> 本项目不会自动点击、不会向游戏发送输入、不会读取游戏进程内存，也不会注入游戏或在游戏窗口内绘制。请自行确认并遵守游戏规则。

## 功能

- 自动寻找 `mhtab.exe` / `MHXYMainFrame` 游戏窗口并显示本地实时预览。
- 先独立识别题目，再定位 3～4 个选项，使用内置离线科举题库匹配答案。
- 布局分析可按分辨率自适应缩放，预览和 OCR 仍使用原始画面。
- 题目 OCR 按精确区域、扩展区域和面板题目带逐级容错；题面不可信时不会盲目识别选项。
- 通过时间连续性保留同题状态、短暂容忍布局抖动，并对候选结果定时重试。
- 画面默认以 30 FPS 采集和预览；识别使用独立节拍，答题框存在时最高 15 FPS，无题界面完成状态清理后降为每 0.2 秒扫描一次。
- 预览使用 WebSocket 传送 I420 平面帧并由 WebView2 WebCodecs 直接绘制，不再逐帧压缩和解码 JPEG；识别节拍不受预览帧率影响。
- 设置中可分别选择 OCR 与预览执行后端；DirectML 选项只在对应显卡完成真实 OCR 自检后显示，失败时自动回退 CPU 且保留用户选择。
- 候选框按综合评分由绿色渐变到红色：低评分为半透明虚线，高评分为较粗实线；评分不是正确概率。
- 支持本地补题和答案修正，本地数据与官方题库分开保存在 `user-data\questions.json`。
- 支持单实例启动、端口冲突提示、题库原子更新和一键退出。
- 可按需保存识别诊断或不含游戏画面的环境诊断。

当前版本为 `0.2.0`。Windows 11 x64 已验证；Windows 10 1903 及以上 x64 是目标兼容范围，但尚未完成实机验证。目前以科举主流程的实际活动验证为准；乡试和殿试只有兼容性实现与有限样本验证，不承诺已经覆盖全部现场界面，仍需后续活动回归。

## 直接使用 Windows 便携版

1. 从 GitHub Releases 下载 `XYQQuiz-v0.2.0-win10-win11-x64.zip` 和同名 `.sha256`。
2. 完整解压到一个新目录，不要直接在压缩包里运行。
3. 双击 `XYQQuiz.exe`，首次捕获时允许 UAC 管理员权限请求。
4. 等待默认 `1440×900` 的可缩放桌面窗口打开；游戏题面出现后，答案框会显示在窗口预览中。
5. 使用界面右侧的“退出程序”安全关闭后台。

第二次双击 EXE 会还原并聚焦已经运行的桌面窗口。若页面提示会话失效，也请重新双击 EXE，不要手工拼接本地 URL。

便携包自带程序、OCR 模型、布局和离线题库，正常启动和识别不需要联网。“更新题库”是唯一会主动访问题库来源的日常功能。

桌面窗口使用系统中的 Microsoft Edge WebView2 Evergreen Runtime，并在 `127.0.0.1` 上选择随机空闲端口。发布包不会捆绑体积较大的 Fixed Version Runtime；如果系统缺少 WebView2，程序会自动退回外部浏览器模式，识别功能仍可使用。也可通过 `XYQQuiz.exe --external-browser` 主动使用外部浏览器进行调试；该模式使用 `config.web.port`（默认 `8765`），端口被占用时会给出明确提示。

## 自检与诊断

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
.\scripts\build-release.ps1 -Version 0.2.0 -Commit working-tree -AllowDevelopmentCommit
```

正式发布构建必须从干净提交运行，并传入完整 40 位 Git SHA：

```powershell
.\scripts\build-release.ps1 -Version 0.2.0 -Commit (git rev-parse HEAD)
```

产物位于 `release\`，包括 ZIP 和 SHA-256 文件。构建脚本会审计公开树和最终 ZIP，拒绝打入 `user-data\`、`diagnostics\` 或本地 `questions.json`。GitHub 的 `v*` 标签工作流会先做公开内容审计和完整测试，再使用标签对应的真实提交 SHA 构建并创建 Release。

## 更新与升级

程序启动不会自动联网。界面中的“更新题库”会从公开网易科举页面获取新数据，完整校验后原子切换 generation；失败时保留当前可用题库。

升级程序时请解压到全新目录，再按需复制旧目录中的 `config.json`、`data`、`user-data`、`logs` 和 `diagnostics`。不要把新版 EXE 或 `_internal` 覆盖到旧目录。

## 已知限制

- 游戏 UI、字体、DPI、动画遮挡或极端分辨率变化可能导致识别失败。
- 题目不在题库、OCR 置信度不足，或正确答案无法和选项唯一匹配时，可能已显示 OCR 题目但不会画框；这是预期的安全降级。
- 乡试和殿试尚未完成覆盖全部界面的活动现场验证；当前发布承诺以已经实际验证的科举主流程为准。
- 首版没有代码签名，Windows SmartScreen 可能显示“未知发布者”。请只从项目 Release 下载并核对 SHA-256，不要全局关闭安全软件。

## 许可证与第三方内容

XYQQuiz 的原创项目代码依据 [PolyForm Noncommercial License 1.0.0](LICENSE) 提供。这是一份非商业源码许可：允许查看和非商业使用，也允许在非商业目的下修改、Fork 和重新发布；重新发布时必须遵守许可证中的通知要求。任何商业使用均不在本许可证授权范围内，必须事先取得版权方另行书面授权。

这不是 OSI 定义的开源许可证，请勿将本项目描述为“开源软件”。上述项目代码许可证不覆盖内置题库、游戏截图衍生的布局锚点、游戏素材、商标或其他第三方内容；这些内容的来源、各自权利和分发边界见 [THIRD_PARTY_NOTICES.txt](THIRD_PARTY_NOTICES.txt)。

XYQQuiz 是非官方项目，与网易没有关联，也未获得网易认可或背书。
