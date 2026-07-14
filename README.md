# XYQQuiz

XYQQuiz 是一个在 Windows 本机运行的《梦幻西游》科举答题辅助显示工具。它通过 Windows Graphics Capture 读取游戏窗口，在浏览器里显示实时预览，并在题目、题库答案和选项都达到高置信度时框出答案。

> 本项目不会自动点击、不会向游戏发送输入、不会读取游戏进程内存，也不会注入游戏或在游戏窗口内绘制。请自行确认并遵守游戏规则。

## 功能

- 自动寻找 `mhtab.exe` / `MHXYMainFrame` 游戏窗口并显示本地实时预览。
- OCR 识别题目和 3～4 个选项，使用内置离线科举题库匹配答案。
- 兼容普通文字题、图片题、不同题面高度和多种窗口分辨率。
- 只在高置信度匹配时显示答案框；信息不足时保持无框状态。
- 支持单实例启动、端口冲突提示、题库原子更新和一键退出。
- 可按需保存识别诊断或不含游戏画面的环境诊断。

当前版本为 `0.1.0`。Windows 11 x64 已验证；Windows 10 1903 及以上 x64 是目标兼容范围，但尚未完成实机验证。会试和殿试已有实际界面适配，乡试仅做了兼容性实现，仍缺少活动现场回归。

## 直接使用 Windows 便携版

1. 从 GitHub Releases 下载 `XYQQuiz-v0.1.0-win10-win11-x64.zip` 和同名 `.sha256`。
2. 完整解压到一个新目录，不要直接在压缩包里运行。
3. 双击 `XYQQuiz.exe`，首次捕获时允许 UAC 管理员权限请求。
4. 等待浏览器自动打开；游戏题面出现后，答案框会显示在网页预览中。
5. 使用网页右侧的“退出程序”安全关闭后台。

第二次双击 EXE 会重新打开已经运行的实例。若浏览器提示会话失效，也请重新双击 EXE，不要手工拼接本地 URL。若 `8765` 端口被其他程序占用，程序会明确提示冲突；可退出占用程序，或修改解压目录中的 `config.json`。

便携包自带程序、OCR 模型、布局和离线题库，正常启动和识别不需要联网。“更新题库”是唯一会主动访问题库来源的日常功能。

## 自检与诊断

双击便携包中的 `一键自检.cmd`，报告会写入 `diagnostics\self-test-latest`。

- “保存识别诊断”包含当前完整游戏画面、题目/选项裁剪、识别状态和日志尾部。点击前会显示隐私确认；分享前仍应自行检查角色名、聊天和其他个人信息。
- “导出环境诊断”不包含游戏画面或题库正文，主要用于排查系统、配置和依赖问题。
- 诊断文件只写在本地 `diagnostics\`，程序不会自动上传。

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

默认服务只监听 `127.0.0.1:8765`。配置文件只允许本机回环地址，HTTP 与 WebSocket 接口还使用进程随机令牌、一次性浏览器引导令牌和严格 Host/Origin 校验。

`recognition.ocr_workers` 现在只允许 `1`，以保证只常驻一份 OCR 模型。旧版 `config.json` 若配置为其他值，启动时会提示迁移错误；请改为 `1` 或删除该项。

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
.\scripts\build-release.ps1 -Version 0.1.0 -Commit working-tree -AllowDevelopmentCommit
```

正式发布构建必须从干净提交运行，并传入完整 40 位 Git SHA：

```powershell
.\scripts\build-release.ps1 -Version 0.1.0 -Commit (git rev-parse HEAD)
```

产物位于 `release\`，包括 ZIP 和 SHA-256 文件。GitHub 的 `v*` 标签工作流会先做公开内容审计和完整测试，再使用标签对应的真实提交 SHA 构建并创建 Release。

## 更新与升级

程序启动不会自动联网。网页中的“更新题库”会从公开网易科举页面获取新数据，完整校验后原子切换 generation；失败时保留当前可用题库。

升级程序时请解压到全新目录，再按需复制旧目录中的 `config.json`、`data`、`logs` 和 `diagnostics`。不要把新版 EXE 或 `_internal` 覆盖到旧目录。

## 已知限制

- 游戏 UI、字体、DPI、动画遮挡或极端分辨率变化可能导致识别失败。
- 题目不在题库、OCR 置信度不足，或正确答案无法和选项唯一匹配时，可能已显示 OCR 题目但不会画框；这是预期的安全降级。
- 乡试尚未完成活动现场验证。
- 首版没有代码签名，Windows SmartScreen 可能显示“未知发布者”。请只从项目 Release 下载并核对 SHA-256，不要全局关闭安全软件。

## 许可证与第三方内容

XYQQuiz 的原创项目代码依据 [PolyForm Noncommercial License 1.0.0](LICENSE) 提供。这是一份非商业源码许可：允许查看和非商业使用，也允许在非商业目的下修改、Fork 和重新发布；重新发布时必须遵守许可证中的通知要求。任何商业使用均不在本许可证授权范围内，必须事先取得版权方另行书面授权。

这不是 OSI 定义的开源许可证，请勿将本项目描述为“开源软件”。上述项目代码许可证不覆盖内置题库、游戏截图衍生的布局锚点、游戏素材、商标或其他第三方内容；这些内容的来源、各自权利和分发边界见 [THIRD_PARTY_NOTICES.txt](THIRD_PARTY_NOTICES.txt)。

XYQQuiz 是非官方项目，与网易没有关联，也未获得网易认可或背书。
