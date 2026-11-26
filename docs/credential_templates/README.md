# 凭据配置模板说明

本目录仅存放**脱敏示例模板**。真实密钥、OAuth Token、PAT 只保存在用户本机私密目录（如 `~/.config/math_posttrain/` 或 `~/.config/p1/`）或 Google Drive 私密目录中，**严禁提交到仓库**。

| 模板文件 | 用途 | 推荐存放位置或环境变量 |
|---|---|---|
| `deepseek.json.example` | DeepSeek API 教师造数 | 环境变量 `DEEPSEEK_API_KEY`，或 `~/.config/math_posttrain/deepseek.json` |
| `swanlab.json.example` | SwanLab 在线训练曲线记录 | 环境变量 `SWANLAB_API_KEY`，或 `<WORK_DIR>/.project1_credentials/swanlab.json` |
| `rclone.conf.example` | 可选 Colab VM 挂载 Google Drive | `P1_RCLONE_CONF`（默认 `~/.config/rclone/rclone.conf`） |
| `gh_credentials.example` | 可选 Colab 文本进度回传 | `~/.config/p1/gh_credentials`（权限 `chmod 600`） |
