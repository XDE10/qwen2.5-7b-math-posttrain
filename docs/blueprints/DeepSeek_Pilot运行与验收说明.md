# DeepSeek Pilot 运行与验收说明

本轮执行日期：2026-10-23。用途：验证训练侧难度估计、同题独立精炼解答与缓存/预算/判分链路。范围固定为难度开发50题、冻结验收50题、OpenR1解答50题；不自动推广1500题。

## 文件入口

- 冻结划分、盲请求与本地验证参考：`data/deepseek_pilot_20261023/`。`blind_requests.json` 无标准答案或原解答；`verifier_references.json` 仅在本地验证端读取。
- 已执行结果与失败记录：[执行报告](<../reports/deepseek_pilot_20261023/执行报告.md>)。
- 请求/响应缓存及预算预留：`reports/deepseek_pilot_20261023/api_cache/`。缓存键包含模型、消息、采样参数和题目/尝试身份，不包含密钥。
- [冻结 policy](<../data/deepseek_pilot_20261023/policy.json>)；[划分 manifest](<../data/deepseek_pilot_20261023/manifest.json>)。

## 环境

使用既有 CPU Python：`python`。Math-Verify 安装在独立 `.pilot_deps/` 中，不覆盖旧虚拟环境的 SymPy；验证在带20秒总超时的子进程运行。

重建验证依赖：

```powershell
python -m pip install --target .pilot_deps "math-verify[antlr4_13_2]==0.9.0" "sympy==1.14.0" "latex2sympy2_extended==1.11.0"
```

已有目录不要无故重复安装；实际版本记录见 `reports/deepseek_pilot_20261023/verifier_dependencies.json`。

## 默认不付费，显式 live 才请求

```powershell
python -m scripts.prepare_deepseek_pilot
python -m scripts.run_deepseek_pilot
python -m pytest tests/test_deepseek_pilot.py -q
```

如需恢复本轮，先在当前进程中设置 `DEEPSEEK_API_KEY`，再运行：

```powershell
try {
    python -u -m scripts.run_deepseek_pilot --live
} finally {
    Remove-Item Env:DEEPSEEK_API_KEY -ErrorAction SilentlyContinue
}
python -m scripts.report_deepseek_pilot
```

不要把真实密钥写入脚本、配置、命令说明或版本库。已完成请求优先读取缓存；不存在持久化的 API 密钥文件。新会话恢复时需要重新配置密钥。

## 预算与恢复规则

冻结上限：累计预留3 USD、最多320个请求、每题至多3次盲生成。本轮没有内容反馈修复；长度不够也不把正确答案反馈给生成器。账户余额差额与按 usage 估算的费用分别报告。

派发前持久化最坏情况费用；失败请求不自动退还预留。由于不能证明网络中断时服务器未计费，`reserved` 的不确定请求默认不重复派发。响应已落盘但状态日志写失败时，从原响应恢复，不再次调用 API。Windows 短暂文件替换锁执行有界重试。

修改提示、模型、准入规则或重新抽样必须新建版本，不能覆盖本轮冻结 policy 后用同一结果宣称新方案独立验收。冻结验收集一旦用于诊断本次失败，下一次独立验收需另留未用于改进的训练侧样本。

## 结果如何使用

难度 gate 不通过时停止全量标签生产。`estimated_level` 永远不是官方 Level；MATH 训练题校准也不能直接保证 OpenR1 跨来源难度准确。

解答自动门槛包括停止状态、结构、学生 tokenizer 长度、噪声和 Math-Verify；另用不同模型进行过程审核。通过只能进入待审核银标候选，不能直接写成所有推导已正确或人类金标。过程审核器本身的错误也需保留。

`sft_pilot_raw.jsonl` 与 `sft_pilot_compact.jsonl` 使用同一接受交集，均标记 `training_approved=false`。没有通过的题保留在失败清单，不用其他题补齐50。未进行 GPU 更新，不据此确定2048 RL预算或报告性能增益。
