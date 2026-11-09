# 推理与 RL 环境的解析结果（2026-11-09，参考用）

用 uv 在云端沙箱里把 [../infer.txt](../infer.txt) 和 [../rl.txt](../rl.txt) 分别按 Python 3.12、3.13 解析出的完整依赖（含 `# via` 来源），2026-11-09 从沙箱迁入仓库，避免随沙箱关闭丢失。代码不读这些文件：VM 上仍按 `envs/{infer,rl}.txt` 安装。

| 文件 | 关键版本 |
|---|---|
| `infer-py3.12.txt`、`infer-py3.13.txt` | vllm 0.33.0、torch 2.14.0、transformers 5.20.2、flashinfer-python 0.7.2.post1 |
| `rl-py3.12.txt`、`rl-py3.13.txt` | vllm 0.32.0、trl 1.17.1、peft 0.21.2、torch 2.13.0、transformers 5.19.0、flashinfer-python 0.7.0.post1 |

这些关键版本与 G4 冒烟 VM（Python 3.13）上实际装出的一致（`colab-reports` 分支 `v93/latest_smoke_summary.txt` 的 env 行）。VM 实际安装的完整清单是第一次建环境时 `uv pip freeze` 写到 Drive 的 `projects/project1/v93/envcache/locks/{infer,rl}-<tag>.txt`，以那份为准；两者不一致时说明 PyPI 上有新版本进入了约束范围。
