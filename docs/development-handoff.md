# 0.5.0 开发交接

当前维护入口为 `main`；复现正式包请检出 `v0.5.0`。仓库包含源代码、模型、布局、官方题库与独立补充图标，另一台电脑无需原开发机临时目录即可运行公开测试和构建。完整私人游戏截图不在仓库中，相应回归会跳过。

## 环境与启动

使用 Windows x64、Python 3.11；构建原生预览还需要 Visual Studio“使用 C++ 的桌面开发”工作负载及 Windows 10/11 SDK。脚本通过 vswhere 自动发现工具集。桌面运行依赖系统 WebView2 Evergreen Runtime。

```powershell
git clone https://github.com/PilyTang/XYQQuiz.git
Set-Location XYQQuiz
git switch main
py -3.11 -m venv .venv
.venv\Scripts\python.exe -m pip install --require-hashes -r requirements-release.txt
.venv\Scripts\python.exe -m pip install --no-build-isolation --no-deps -e .
.venv\Scripts\python.exe -m pip install "pytest==8.4.2" "pytest-asyncio==0.26.0"
Copy-Item config.example.json config.json
.venv\Scripts\xyq-quiz.exe --config config.json
```

仅首次建立配置时复制示例，保留已有配置中的用户设置。首次自动分档只在正常启动执行；自检和版本查询不消耗它。显卡检测仅用于分档，实际 OCR 和预览初始化仍各自验证、回退。

## 修改入口

| 范围 | 入口与说明 |
|---|---|
| 首次硬件分档 | `src/xyq_quiz/performance/hardware_profile.py`、`config.py`；[分档与低配模式](low-resource-mode.md) |
| 后端保存与运行上限 | `performance/settings.py`、`performance/controller.py`；[显卡优先](2026-09-08-gpu-default.md) |
| 教师节识别 | `recognition/teachers_day.py`、`knowledge/teacher_matcher.py`、`runtime/coordinator.py`；[遮挡容错](teacher-cursor-occlusion.md) |
| 图标资源与更新 | [独立补充库与全来源边框匹配](teachers-day-supplement-icons.md)；官方更新不可覆盖补充记录 |
| 预览和框选 | [CPU/GPU 统一样式](overlay-style.md)；更改原生辅助程序后必须重新编译 |
| 发布 | `scripts/build-release.ps1`、`.github/workflows/release.yml`、`packaging/` |

以上源码简称均位于 `src/xyq_quiz/`。配置、日志、诊断和本地科举补题是用户数据，不进入发布包；不得将完整游戏截图加入 Git。

## 验证与发布

```powershell
.venv\Scripts\python.exe scripts/check_public_tree.py
git diff --check
.venv\Scripts\python.exe -m pytest -q
```

0.5.0 功能提交 `35f6f8e9d87b994a072edfc72016b94e4bc387a0` 的干净源码验证为 765 passed、3 skipped；本地冻结包 headless 自检为 9 PASS、1 WGC 跳过，并验证中文及空格解压路径。此数字对应功能提交，发布标签对应提交及最终结果以该标签的 GitHub Actions 为准。

合并到 main 后等待 CI 通过，再从对应干净提交创建版本标签。正式构建命令：

```powershell
.\scripts\build-release.ps1 -Version 0.5.0 -Commit (git rev-parse HEAD)
```

标签、应用版本、PE 版本、发布说明必须一致。推送 `v0.5.0` 会触发完整测试、公开树审计、冻结包构建与自检，发布 ZIP 和 SHA-256。完整解压到新目录验证，保留 `longPathAware`，不要以源码运行代替冻结 EXE 验证。构建产物不提交 Git。

## 尚未解决的边界

- Windows 10 1909（18363.657）报告 `windows_capture.pyd / 0xC0000409`；原生捕获崩溃仍待定位，不能标为已修复。
- 硬件分档覆盖边界、保存、迁移、异常及本机读取；尚未在多台真实低端机器验证门槛和速度收益。
- 私人完整截图、全部系统/驱动/DPI 组合及长期活动现场验收不由公开测试替代。

旧 `v0.4.1-handoff.md`、日期命名的实验文档和历史验收数字只用于追溯；用户当前行为以 README、[0.5.0 发布说明](releases/v0.5.0.md) 及上述专题文档为准。
