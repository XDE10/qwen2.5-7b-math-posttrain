# src/ —— 现行库代码

**16 个现行模块**（含 4 个 `__init__.py`），从入口出发的导入闭包内。其余 59 个历史模块已移入 `archive/src/`。

## data/（7 个文件）

| 模块 | 职责 |
|---|---|
| `__init__.py` | 「版本化数据准备，独立于模型训练」 |
| `approved_training_inputs.py` | 文件级工程批准校验，白名单限定实际模型输入（哈希 + 逐行授权） |
| `geometry_model_serialization.py` | 区分审计身份与模型可见的几何文本；源 radius ID 不进入序列化文本 |
| `perception_text_guards.py` | 保守的有限 prose 蕴含判断；不匹配 ≠ 图像排他 |
| `perception_v4.py` | fact 级 grounding 与稀疏 scoped reward（不是整图精度） |
| `project3_perception_prototype.py` | 独立有限 InterGPS AST/fact 原型，永不认证视觉金标 |
| `project3_perception_v2.py` | 仅审查用的 InterGPS v2 安全符号 AST；源记录不变成金标 |

## evaluation/（2 个文件）

| 模块 | 职责 |
|---|---|
| `__init__.py` | 「确定性评测参考，带显式拒绝状态」 |
| `math_verifier.py` | 有界答案抽取与安全表达式等价，不用 `eval`/`sympify`；只接受最终标记或独立表达式 |

## losses/（2 个文件）

| 模块 | 职责 |
|---|---|
| `__init__.py` | 导出 `sft_loss`、`group_advantages`、`policy_loss`、`aligned_kl`、`mixture_fkl` |
| `reference.py` | 显式 CPU 参考损失（正确性判据，不是 GPU 训练器）；mask 描述目标 token 位置，SFT/OPD 中位置 t-1 预测 t |

## training/（4 个文件）

| 模块 | 职责 |
|---|---|
| `approved_hf.py` | manifest 门禁的 HF SFT/GRPO 入口（被 `scripts/training/*.py` 调用）；GPU 验收另算 |
| `frozen_verifier_v4.py` | 面向未改动 v2 grader core 的持久有界子进程传输 |
| `math_reward_v2.py` | 可审计状态：基础设施失败直接中止，绝不记成答案错误 |
| `perception_reward_v4.py` | 公开训练回调；稀疏标签只是私有 reward 元数据 |

## 边界

- 现行 = 从活跃入口出发的导入闭包 + 文档明确引用；不在闭包内的模块不要擅自提升为主线。
- `src/synthesis/`、`src/teachers/` 是历史模块归档后残留的空目录（只剩未跟踪的 `__pycache__`，不入库），不是现行代码。
- 历史 59 个模块在 `archive/src/`（各代 project1/2/3 数据处理、奖励、训练、验证器），只作追溯；其中的 `from src...` 导入可能已失效，不保证可直接运行。
- 新增/修改模块要同步对应测试（`tests/data|evaluation|losses|training/`），并遵守「不删历史文件」的约定：被取代的旧模块进 `archive/src/`。
