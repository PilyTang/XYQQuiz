# 贡献指南

感谢参与 XYQQuiz。提交代码前请遵守以下边界：

- 不加入自动点击、游戏输入、进程内存读取或注入功能。
- 不提交真实游戏全屏截图、角色名、服务器信息、聊天内容或个人绝对路径。
- 新识别逻辑必须在信息不足时安全降级为无答案框。
- 修改行为时同时补充或更新测试。

开发环境与测试命令见 [README.md](README.md)。提交前运行：

```powershell
.venv\Scripts\python.exe .\scripts\check_public_tree.py
.venv\Scripts\python.exe -m pytest -q
```

真实截图回归只在本地私有 fixture 目录运行。Issue 或 Pull Request 中请使用脱敏截图、合成样本或最小文本复现。
