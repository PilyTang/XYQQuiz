---
status: accepted
---

# 将 DirectML 作为可选 OCR 执行后端

XYQQuiz 的 OCR 执行后端支持 CPU 与 DirectML，不增加 CUDA 分支。DirectML 按具体 Windows 图形适配器分别初始化和自检，只有能够完成真实模型推理的设备组合才出现在设置中；启动或运行中失败时仅 OCR 回退 CPU，保留用户选择并报告原因。

自动后端策略只在 DirectML 通过完整固定识别样本、且最终题目、答案、置信等级和框选决定不退化时考虑它。允许不改变业务结果的微小浮点差异；热态 OCR P95 延迟不得比 CPU 慢超过 10%，单次 OCR 的 CPU 时间必须至少降低 20%。未达到自动门槛但通过可用性自检的 DirectML 设备仍允许用户手动选择。
