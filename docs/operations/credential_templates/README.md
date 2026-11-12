# 凭据文件模板

真实凭据只放在下表的位置，不进 git，不贴进对话或日志。这里的 `*.example` 只给格式，尖括号里的内容换成自己的值。`bash experiments/qfft_v93/check_local.sh` 会检查这些文件是否存在、权限是否为 600，不显示内容。

| 模板 | 真实位置 | 权限 | 谁读 | 怎么生成 |
|---|---|---|---|---|
| [gh_credentials.example](gh_credentials.example) | 本机 `~/.config/p1/gh_credentials`（`P1_GH_CREDENTIALS` 可改） | 600，目录 700 | launch.sh 上传到 VM 的 `/content/gh_credentials`，VM 用它把报告推到 `colab-reports` 分支，任务结束删除 | `bash experiments/qfft_v93/colab/setup_wsl.sh` 会提示输入 token（不回显）并写好。建议 fine-grained token：只选本仓库，Contents 设为 Read and write，有效期覆盖整个项目 |
| [rclone.conf.example](rclone.conf.example) | 本机 `~/.config/rclone/rclone.conf`（`P1_RCLONE_CONF` 可改） | 600 | launch.sh 上传到每台 VM 并挂载 `mydrive:` 到 `/content/mydrive` | `rclone config` 新建名为 `mydrive` 的 Google Drive 远端（scope 选 drive，root folder 留空即 My Drive 根目录），或直接用在 `colab console` 里用过的那份 |
| [swanlab.json.example](swanlab.json.example) | Google Drive `.project1_credentials/swanlab.json`（VM 上是 `/content/mydrive/.project1_credentials/swanlab.json`） | Drive 私有 | 训练与 GRPO 阶段读取后注入子进程环境；读不到时训练直接报错，不转离线 | SwanLab 设置页复制 API Key |
| [deepseek.json.example](deepseek.json.example) | 本机 `~/.config/p1/deepseek.json`（或 Windows 上的旧位置、环境变量 `DEEPSEEK_API_KEY`） | 600 | `teacher fill` 调官方 DeepSeek API | DeepSeek 开放平台创建 API Key |

另外两样也只在本机：OpenCode Go 的 Key 在 opencode 的 `opencode.db` 里（`teacher.py` 只读查询），代理设置在 `~/.proxy_env.sh`（新终端先 `source ~/.proxy_env.sh`）。

`rclone.conf` 里的 refresh token 在多台 VM 上共用没有问题。若某天挂载或 `rclone lsf` 报 token 失效，在本机运行 `rclone config reconnect mydrive:` 重新授权即可，launch.sh 下次会上传新的文件。
