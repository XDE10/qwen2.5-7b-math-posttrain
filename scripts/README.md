# scripts 索引

共 138 个入口/模块（含历史批次 v1…v92 与 .pyw 窗口入口）。文件名中的版本号表示迭代批次，最新批次见 [AGENTS.md](../AGENTS.md)；每条命令以 `python -m scripts.<模块名>` 或直接 `python scripts/<文件>` 在**项目根目录**执行。

本目录混放主线与历史脚本：历史脚本保留用于复现当日报告，不代表当前推荐路径。

| 文件 | 说明（模块 docstring 首行） |
|---|---|
| [`analyze_openr1_candidates.py`](analyze_openr1_candidates.py) | Full pinned default/train read-only analysis, resumable per-shard cache. |
| [`apply_code_v2_input_budget.py`](apply_code_v2_input_budget.py) | Final OPD input-budget gate and deterministic training-prompt refill. |
| [`approve_rl_pool_v4.py`](approve_rl_pool_v4.py) | User-preauthorized conditional approval after complete, independently checked E0 sampling. |
| [`approve_rl_pool_v8.py`](approve_rl_pool_v8.py) | User-preauthorized conditional approval after complete, independently checked E0 sampling. |
| [`audit_inequality_exclusions_v7.py`](audit_inequality_exclusions_v7.py) | Read-only reason audit. No API, no quota or acceptance-rule changes. |
| [`audit_official_code_benchmarks.py`](audit_official_code_benchmarks.py) | Read-only supplementary decontamination against downloaded official releases. |
| [`audit_project2_prompt_contract.py`](audit_project2_prompt_contract.py) | Offline, pinned-tokenizer contract audit. Never loads model weights. |
| [`audit_project3_overlap.py`](audit_project3_overlap.py) | Run only local CPU audit, after prepare_project3_sources.py. |
| [`audit_v3_source_metadata.py`](audit_v3_source_metadata.py) | Additional source-metadata review queue; no changes to the frozen v3 pool. |
| [`build_code_v3.py`](build_code_v3.py) | Versioned exact-wrapper removal; replacements ONLY from audited AceCode pool. |
| [`build_compact_a10080_v18.py`](build_compact_a10080_v18.py) | Derive a large-microbatch variant without changing the effective batch. |
| [`build_compact_cosine_v17.py`](build_compact_cosine_v17.py) | Small versioned derivative; preserve every previously executed runner. |
| [`build_data_human_review_pack.py`](build_data_human_review_pack.py) | Build a read-only, deterministic human review packet. No API or training. |
| [`build_geometry_funnel_v8.py`](build_geometry_funnel_v8.py) | Read-only, mutually exclusive geometry exits and full-pool image overlap. |
| [`build_project1_drafts_v2.py`](build_project1_drafts_v2.py) | Generate unfrozen, execution-blocked experiment drafts. Never changes release. |
| [`build_project3_all_split_candidates.py`](build_project3_all_split_candidates.py) | All unique matched splits, separated for train/diagnostic-only use. |
| [`build_project3_perception_preview.py`](build_project3_perception_preview.py) | Incremental supplement; does not modify the frozen 200-case prototype. |
| [`build_project3_perception_prototype.py`](build_project3_perception_prototype.py) | Build a frozen, train-only 200-case perception prototype and pending reviewpack. |
| [`build_project3_perception_v2.py`](build_project3_perception_v2.py) | Re-run exactly the frozen 200 training topics, retaining every original record. |
| [`build_screening_v8.py`](build_screening_v8.py) | Create new sampler/approval versions, preserving the frozen v4 sources. |
| [`build_sft_epoch_v10.py`](build_sft_epoch_v10.py) | Derive isolated epoch-based SFT runner/configs without modifying v6/drafts. |
| [`build_swanlab_online_v11.py`](build_swanlab_online_v11.py) | Derive online SFT version while retaining v10 and its training budget. |
| [`check_sft_tokenizer_v6.py`](check_sft_tokenizer_v6.py) | Offline full-pair encoding check; no model loading, training, or installs. |
| [`check_topup_source_integrity.py`](check_topup_source_integrity.py) |  |
| [`code_sandbox_worker.py`](code_sandbox_worker.py) | Container-only worker. Never import or execute this file on the Windows host. |
| [`compact_distillation_v2.py`](compact_distillation_v2.py) | Versioned cache audit + independent 40-question pilot, no automatic scaling. |
| [`compact_distillation_v3.py`](compact_distillation_v3.py) | Data-only v3 pool, frozen 30-question pilot; bulk generation is disabled. |
| [`compact_distillation_v4.py`](compact_distillation_v4.py) | Local v3 cache reaudit, explicit scope pool, frozen independent v4 pilot. |
| [`configure_topup_training.py`](configure_topup_training.py) | Versioned smoke configs for approved V6 rows, not a GPU acceptance claim. |
| [`decontaminate_project1_v2.py`](decontaminate_project1_v2.py) | No download, no deletion, no approval changes. Output hit lists only. |
| [`deliver_topup_v6.py`](deliver_topup_v6.py) | Wait for bounded API work, then publish versioned local reports and hashes. |
| [`diagnose_project3_vision.py`](diagnose_project3_vision.py) | Paired original/black/no-image/occluded diagnostics on identical fixed IDs. |
| [`download_project3_train_ranges.py`](download_project3_train_ranges.py) | Optional official-source ranged fallback; validate Git blob identity before use. |
| [`download_screening_sources.py`](download_screening_sources.py) | Download the two authorized public datasets only; logs remain concise. |
| [`embedding_screen_project1_v4.py`](embedding_screen_project1_v4.py) | CPU-only chunked MiniLM cosine screening; preserve every sealed source. |
| [`evaluate_approved_math.py`](evaluate_approved_math.py) | Fixed greedy math evaluation; sealed JSONL and model identity in every run. |
| [`evaluate_compact_dev_hard_v20.py`](evaluate_compact_dev_hard_v20.py) | v20: completed compact v18 on unchanged dev/hard protocols, no SwanLab. |
| [`evaluate_compact_math500_v20.py`](evaluate_compact_math500_v20.py) | v20: user-authorized MATH-500 snapshot for compact v18, frozen v15 protocol. |
| [`evaluate_e1_v13.py`](evaluate_e1_v13.py) | Evaluate one completed formal SFT arm on dev and the fixed 637 candidates. |
| [`evaluate_math500_v15.py`](evaluate_math500_v15.py) | Fixed MATH-500 greedy evaluation for E0 or one completed E1 arm. |
| [`evaluate_project1_dev_v9.py`](evaluate_project1_dev_v9.py) | Fixed dev-only E0 protocol and preserved training-side hard-candidate probe. |
| [`export_code_function_cache.py`](export_code_function_cache.py) | Reproducible small function-row cache from the pinned full parquet source. |
| [`export_delegated_ai_releases.py`](export_delegated_ai_releases.py) | Export only explicitly delegated AI-pass subsets; never certify whole pools. |
| [`export_project1_sft_v2.py`](export_project1_sft_v2.py) | Export sealed proposal indices as review-only native SFT candidates, CPU/local only. |
| [`export_strict_geometry_v8.py`](export_strict_geometry_v8.py) | Reparse every original-pixel response without further API calls. |
| [`export_topup_v6.py`](export_topup_v6.py) | Exact quota exports, only pass evidence; preserve all rejected/surplus rows. |
| [`finalfill_common.py`](finalfill_common.py) |  |
| [`finalize_delegated_release.py`](finalize_delegated_release.py) | Protect old artifacts, refresh current entry notices and hash new delivery. |
| [`finalize_overnight_report.py`](finalize_overnight_report.py) | Verify required evidence and assemble the human report without network access. |
| [`finalize_project1_sft.py`](finalize_project1_sft.py) | Revalidate the sealed OpenR1 selection against source trajectories and export. |
| [`finalize_topup_v6.py`](finalize_topup_v6.py) | Seal exact observed counts and input checks after both API owners finish. |
| [`freeze_project1_screening_v3.py`](freeze_project1_screening_v3.py) | Conservative candidate exclusion and SFT parameter preregistration only. |
| [`grade_rl_candidates_v8.py`](grade_rl_candidates_v8.py) | CPU-only grading after v8 GPU generation exits; no model imports/downloads. |
| [`materialize_colab_config_v2.py`](materialize_colab_config_v2.py) | Resolve Drive variables and pin a completed SFT export into a new config. |
| [`math_verify_pilot_worker.py`](math_verify_pilot_worker.py) | Isolated bounded Math-Verify worker; stdout is JSON, never an API credential. |
| [`math_verify_v3_strict_worker.py`](math_verify_v3_strict_worker.py) | Conservative typed admission supplement, separate from frozen pilot verifier. |
| [`pin_e0_snapshot_v4.py`](pin_e0_snapshot_v4.py) | Verify already-local 7B files against pinned Hub metadata; no weight download. |
| [`plan_project1_sft_v2.py`](plan_project1_sft_v2.py) | Reproduce proposal-only selections from the sealed full audit; no SFT export. |
| [`preflight_delegated_release.py`](preflight_delegated_release.py) | Real approved-row/CPU-tokenizer preflight; not GPU training or model smoke. |
| [`prepare_code_v2_stdio_extension.py`](prepare_code_v2_stdio_extension.py) | Extend dev candidates from never-trained v1 prompts; no split is overwritten. |
| [`prepare_deepseek_pilot.py`](prepare_deepseek_pilot.py) | Freeze train-only calibration splits and a 50-question blind-solution pilot. |
| [`prepare_finalfill_v7.py`](prepare_finalfill_v7.py) | Inventory and independently freeze unused sources without weakening filters. |
| [`prepare_internship_migration.py`](prepare_internship_migration.py) | Create three self-contained migration copies without changing project sources. |
| [`prepare_project3_sources.py`](prepare_project3_sources.py) | Download pinned public annotation/data files, preserving licenses and SHA256. |
| [`prepare_rl_support_v4.py`](prepare_rl_support_v4.py) | CPU support-domain selfcheck, resumable; never modifies sealed source. |
| [`prepare_strict_inequalities_v8.py`](prepare_strict_inequalities_v8.py) |  |
| [`prepare_topup_code_train_reserve.py`](prepare_topup_code_train_reserve.py) | Fresh stdin train reserve; no dev candidate recycled across splits. |
| [`prepare_topup_geometry_reserve.py`](prepare_topup_geometry_reserve.py) | Remaining train-only reserve for a second fact audit if 1000 are insufficient. |
| [`prepare_topup_p1.py`](prepare_topup_p1.py) | Fresh 1500 questions with shortest complete correct source length 1200..3200. |
| [`prepare_topup_p1_supplement.py`](prepare_topup_p1_supplement.py) | Separate five-question supplement, not a rewrite of the frozen 1495 cohort. |
| [`prepare_topup_p2.py`](prepare_topup_p2.py) | Seal new dev BEFORE train; no rejected or historical train IDs recycled. |
| [`prepare_topup_p2_dev_reserve.py`](prepare_topup_p2_dev_reserve.py) | Additional never-assigned stdin dev reserves; exact same test gates. |
| [`prepare_topup_p3.py`](prepare_topup_p3.py) |  |
| [`prepare_type_reserves_v8.py`](prepare_type_reserves_v8.py) | Unused same-standard reserves; preparation itself grants no approval. |
| [`project1_batch500.py`](project1_batch500.py) | Versioned <=500 blind generations, shared $5 cap, separate local/AI audits. |
| [`publish_strict_v8_report.py`](publish_strict_v8_report.py) | Human-readable delivery, sourced exclusively from the sealed artifacts. |
| [`quarantine_code_literal_conflicts.py`](quarantine_code_literal_conflicts.py) | Additional bounded static diagnostics; never execute reference programs. |
| [`reaudit_typed_inequalities_v8.py`](reaudit_typed_inequalities_v8.py) | Bounded offline typed/role probe on27 known development cases, not a pilot. |
| [`recover_bounded_source_v8.py`](recover_bounded_source_v8.py) | Deterministically retry local resource timeouts, never paid semantic decisions. |
| [`recover_compact_server_v19.py`](recover_compact_server_v19.py) | Fresh-server recovery from existing Drive artifacts; no model downloads. |
| [`refill_project1_native_v5.py`](refill_project1_native_v5.py) | Refill fixed six-stratum native proposal after v4 posthoc exclusions. No API. |
| [`repair_trial_journal_v4.py`](repair_trial_journal_v4.py) | Explicitly back up and remove only a torn final trial-journal fragment. |
| [`reparse_topup_geometry.py`](reparse_topup_geometry.py) | Zero-call sidecar parsing: retain raw response, actual finish and fact guards. |
| [`report_code_v2.py`](report_code_v2.py) | Human-readable execution record from sealed pilot and bulk artifacts. |
| [`report_compact_v2.py`](report_compact_v2.py) | Rebuild a human-readable audit without paid calls or changes to frozen runs. |
| [`report_compact_v3.py`](report_compact_v3.py) | Supplement v3 with typed-answer isolation and reproducible readable data report. |
| [`report_data_revision_v4.py`](report_data_revision_v4.py) | Build review-only delivery from actual versioned artifacts; no API or training. |
| [`report_data_screening.py`](report_data_screening.py) | Summarize actual screening outputs; no model/API calls. |
| [`report_deepseek_pilot.py`](report_deepseek_pilot.py) | Summarize actual paid pilot evidence without changing acceptance decisions. |
| [`report_delegated_release.py`](report_delegated_release.py) | Report measured delegated audits and precisely scoped training releases. |
| [`report_project1_pretraining_audit.py`](report_project1_pretraining_audit.py) | Report measured SFT selection bias and fixed-tokenizer readiness limits. |
| [`resolve_geometry_uncertain_v7.py`](resolve_geometry_uncertain_v7.py) | One recorded independent pixel+exact-proof adjudication; no new API. |
| [`resume_code_v3_dev.py`](resume_code_v3_dev.py) | Complete pending v3 dev only after fresh restricted-container tests succeed. |
| [`revalidate_tokenizer_alignment.py`](revalidate_tokenizer_alignment.py) | Offline replay of the fixed four-model tokenizer audit; no weights/network. |
| [`review_compact_format.py`](review_compact_format.py) | Post-hoc format robustness review; never modifies either frozen pilot. |
| [`review_geometry_grounding.py`](review_geometry_grounding.py) | Delegated visual audit of the fixed 200 prototypes, within the shared $5 cap. |
| [`review_math_pairs_release.py`](review_math_pairs_release.py) | Delegated review of new batch pairs plus the prior ten AI-pass v4 pairs. |
| [`review_project2_deepseek.py`](review_project2_deepseek.py) | User-delegated itemwise AI audit of fixed P2 math/code train/dev. No training. |
| [`run_compact_a10080_v18.py`](run_compact_a10080_v18.py) | Compact-only 80GB GPU launch; automatically authenticated online training. |
| [`run_compact_cosine_v17.py`](run_compact_cosine_v17.py) | Launch only compact from original E0; automatic SwanLab credentials. |
| [`run_compact_eval_v20.py`](run_compact_eval_v20.py) | Run user-requested dev, fixed hard probe and MATH-500 at 4096; no training/login. |
| [`run_cpu_demo.py`](run_cpu_demo.py) | Repeatable local-only synthesis/geometry smoke; no HTTP provider exists. |
| [`run_deepseek_pilot.py`](run_deepseek_pilot.py) | Run only the frozen pilot; never automatically expand to 1500 questions. |
| [`run_e1_autoauth_v12.py`](run_e1_autoauth_v12.py) | Load the user-authorized private Drive credential, then launch online E1. |
| [`run_e1_review_v13.py`](run_e1_review_v13.py) | Sync completed offline SFT, then evaluate both arms online without retraining. |
| [`run_finalfill_v7.py`](run_finalfill_v7.py) | Bounded rolling top-up to1500, full train geometry coverage, same guards. |
| [`run_math500_v15.py`](run_math500_v15.py) | Sequential fixed E0/raw/compact test; automatic auth; never trains. |
| [`run_rl_screening_pipeline_v4.py`](run_rl_screening_pipeline_v4.py) | One command: sampler subprocess exits/releases GPU, then CPU approval checks. |
| [`run_strict_v8.py`](run_strict_v8.py) |  |
| [`run_strict_v8_resume.py`](run_strict_v8_resume.py) |  |
| [`run_topup_reserves.py`](run_topup_reserves.py) | Resume remaining work after the first owner's lock releases, same $8 ledger. |
| [`run_topup_v6.py`](run_topup_v6.py) | One shared-budget owner, resumable independent pipelines, max 8 in flight. |
| [`sample_rl_candidates_v4.py`](sample_rl_candidates_v4.py) | Single-GPU vLLM screening, durable sharded raw/graded journals, no Drive gate. |
| [`sample_rl_candidates_v8.py`](sample_rl_candidates_v8.py) | Single-GPU vLLM screening, durable sharded raw/graded journals, no Drive gate. |
| [`seal_code_v2_pilot.py`](seal_code_v2_pilot.py) | Seal the pilot after explicit agent review and regression checks. |
| [`seal_local_preparation_v4.py`](seal_local_preparation_v4.py) | Seal CPU evidence and pre-render problem-only prompts for the sampler. |
| [`seal_strict_v8.py`](seal_strict_v8.py) | Versioned release of observed approvals, with exact counts and hashes. |
| [`strict_v8_common.py`](strict_v8_common.py) |  |
| [`strict_v8_verify_worker.py`](strict_v8_verify_worker.py) |  |
| [`summarize_project1_v15.py`](summarize_project1_v15.py) | Read-only metric aggregation; no inference, scoring changes or approval. |
| [`topup_common.py`](topup_common.py) |  |
| [`train_project1_rl.py`](train_project1_rl.py) |  |
| [`train_project1_rl_v2.py`](train_project1_rl_v2.py) |  |
| [`train_project1_rl_v4.py`](train_project1_rl_v4.py) |  |
| [`train_project1_sft.py`](train_project1_sft.py) |  |
| [`train_project1_sft_v10.py`](train_project1_sft_v10.py) |  |
| [`train_project1_sft_v11.py`](train_project1_sft_v11.py) |  |
| [`train_project1_sft_v17.py`](train_project1_sft_v17.py) |  |
| [`train_project1_sft_v18.py`](train_project1_sft_v18.py) |  |
| [`train_project1_sft_v2.py`](train_project1_sft_v2.py) |  |
| [`train_project1_sft_v4.py`](train_project1_sft_v4.py) |  |
| [`train_project1_sft_v6.py`](train_project1_sft_v6.py) | Versioned SFT entry; no model is loaded merely by importing it. |
| [`train_project2_opd.py`](train_project2_opd.py) |  |
| [`train_project3.py`](train_project3.py) |  |
| [`verifier_stream_worker_v4.py`](verifier_stream_worker_v4.py) | Stream JSON requests; execute the original strict worker function unchanged. |
| [`verify_reference_v4.py`](verify_reference_v4.py) | v4 run-identity aware reference validation; arbitrary local output-root. |
| [`verify_sft_reference_v2.py`](verify_sft_reference_v2.py) | CPU tiny verification or explicitly requested local merged-model verification. |
| [`write_approved_training_configs.py`](write_approved_training_configs.py) | Write explicit initial A100 smoke configs; no weight downloads or execution. |
