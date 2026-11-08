# tools 索引

共 60 个离线工具（校验、监控、造数入口）。除注明外均不下载模型、不调用付费 API。

| 文件 | 说明（模块 docstring 首行） |
|---|---|
| [`archive_qfft_followups_v70.py`](archive_qfft_followups_v70.py) | Archive existing QFFT responses for later handling. No API or GPU imports/calls. |
| [`build_external_review_zip_v92.py`](build_external_review_zip_v92.py) |  |
| [`build_qfft_ds_dataset_v63.py`](build_qfft_ds_dataset_v63.py) | Prepare by default. --run sends paid/subscription requests only during off-peak. |
| [`build_qfft_ds_dataset_v66.py`](build_qfft_ds_dataset_v66.py) | Prepare by default. --run sends paid/subscription requests only during off-peak. |
| [`classify_truncated_dev_v38.py`](classify_truncated_dev_v38.py) | DeepSeek reviews all 122 saved length-truncated responses; no GPU work. |
| [`colab_release_after_save_v57.py`](colab_release_after_save_v57.py) | Release this Colab VM only after a complete final-export receipt verifies. |
| [`continue_final_eval_v33.py`](continue_final_eval_v33.py) | Continue AFTER final dev256: CPU/DS review, MATH-500, then hard637 x4. |
| [`continue_qfft_all_v71.py`](continue_qfft_all_v71.py) | Supervise authorized completion of the frozen 1000 questions, preserving v66 outputs. |
| [`diagnose_dev_truncation_v36.py`](diagnose_dev_truncation_v36.py) | CPU-only examination of saved dev outputs; no generation or API calls. |
| [`evaluate_e0_romescape_avg8_v52.py`](evaluate_e0_romescape_avg8_v52.py) | E0 then RomEscape MATH-500, eight samples each, 7936 initial + optional 256 final-answer tokens. |
| [`evaluate_final_dev256_v32.py`](evaluate_final_dev256_v32.py) | Final SFT dev256 only, 32768 output tokens; same prompt function as AIMO SFT. |
| [`evaluate_last_e0_protocol_v30.py`](evaluate_last_e0_protocol_v30.py) | Only final SFT MATH-500, matching the historical E0 v15 protocol. |
| [`evaluate_last_with_hard637_v31.py`](evaluate_last_with_hard637_v31.py) | Run final-model MATH-500 v30, then the existing 637-question E0 0/4 probe. |
| [`evaluate_math500_final_answer_v41.py`](evaluate_math500_final_answer_v41.py) | Final step429 MATH-500, 7936 reasoning + optional 256 answer tokens. |
| [`evaluate_public_palign_math500_v47.py`](evaluate_public_palign_math500_v47.py) | Published P-ALIGN 7B LoRA MATH-500, 7936 initial + optional 256 final-answer tokens. |
| [`evaluate_public_romescape_math500_v51.py`](evaluate_public_romescape_math500_v51.py) | RomEscape public s1.1 SFT MATH-500, 7936 initial + optional 256 final-answer tokens. |
| [`evaluate_public_romescape_qfft_math500_v58.py`](evaluate_public_romescape_qfft_math500_v58.py) | RomEscape public s1.1 QFFT MATH-500, 7936 initial + optional 256 final-answer tokens. |
| [`evaluate_public_s11_math500_v43.py`](evaluate_public_s11_math500_v43.py) | Published s1.1-7B MATH-500, 7936 initial + optional 256 final-answer tokens. |
| [`evaluate_sft_ds_v29.py`](evaluate_sft_ds_v29.py) | MATH-500 main evaluation: native Math-Verify, then DeepSeek failure review. |
| [`package_aimo_sft_v25.py`](package_aimo_sft_v25.py) | Package frozen 1000 original traces and independent train/eval entrypoints. |
| [`package_aimo_sft_v26.py`](package_aimo_sft_v26.py) | Package all eligible original traces and independent train/eval entrypoints. |
| [`package_colab_compat_v27.py`](package_colab_compat_v27.py) | Build a small, explicit source overlay for the already-delivered v26 bundle. |
| [`package_parallel_bench_v22.py`](package_parallel_bench_v22.py) | Add the parallel benchmark to an unchanged v21 ZIP, verifying every entry. |
| [`package_projection_pilot_v21.py`](package_projection_pilot_v21.py) | Package only the independent inference pilot and its existing grader deps. |
| [`package_projection_pilot_v23.py`](package_projection_pilot_v23.py) | Build a self-contained v23 ZIP without changing previous releases. |
| [`package_projection_pilot_v24.py`](package_projection_pilot_v24.py) | Build a self-contained v24 ZIP without changing previous releases. |
| [`pilot_qfft_teachers_v60.py`](pilot_qfft_teachers_v60.py) | Bounded blind Flash/Pro teacher pilot: 12 train-pool questions, one call each. |
| [`prepare_aimo_eval_v25.py`](prepare_aimo_eval_v25.py) | Freeze public AIME 2024/2025 inputs and reuse the existing MATH500 file. |
| [`prepare_aimo_sft_v25.py`](prepare_aimo_sft_v25.py) | Freeze the first 1000 seeded, complete <=8192-token original AIMO2 traces. |
| [`prepare_aimo_sft_v26.py`](prepare_aimo_sft_v26.py) | Freeze all complete original AIMO2 traces fitting the Qwen 8192-token window. |
| [`prepare_projection_pilot_v21.py`](prepare_projection_pilot_v21.py) | Freeze 32 length-stratified, unapproved AIMO2 sampling inputs (CPU only). |
| [`prepare_romescape_topup6_v57.py`](prepare_romescape_topup6_v57.py) | Replace the six already excluded s1 rows from the existing approved raw pool. |
| [`prepare_s1_full_probe_v46.py`](prepare_s1_full_probe_v46.py) | Freeze the authorized capacity probe from the existing local s1K-1.1 snapshot. |
| [`probe_s1_full_sft_v46.py`](probe_s1_full_sft_v46.py) | Eight-update, single-A100 full-SFT capacity probe. Never starts the six-epoch run. |
| [`qfft_ds_common_v63.py`](qfft_ds_common_v63.py) | Frozen questions and blind teacher contract; importing this module never calls an API. |
| [`qfft_ds_common_v66.py`](qfft_ds_common_v66.py) | Frozen questions and blind teacher contract; importing this module never calls an API. |
| [`qfft_stream_transport_v79.py`](qfft_stream_transport_v79.py) | Durable OpenAI-style SSE transport. No retries, no credentials saved. |
| [`render_projection_prompt_v23.py`](render_projection_prompt_v23.py) | Print full ChatML for empty prefix and a synthetic 256-token continuation. |
| [`render_projection_prompt_v24.py`](render_projection_prompt_v24.py) | Print full ChatML for empty prefix and a synthetic 256-token continuation. |
| [`retry_dev_judge_evidence_v35.py`](retry_dev_judge_evidence_v35.py) | Retry only unfinished dev reviews after a judge's non-verbatim evidence. |
| [`retry_length_max_v86.pyw`](retry_length_max_v86.pyw) | User-authorized bounded max -> high repair; original 1000 stays read-only. |
| [`retry_truncation_diagnosis_v40.py`](retry_truncation_diagnosis_v40.py) | Retry only unfinished classifier cases with bounded-length evidence quotes. |
| [`review_e0_best_round_v54.py`](review_e0_best_round_v54.py) | Review only native failures in the highest-scoring E0 v52 round. Local CPU/API only. |
| [`review_final_local_v34.py`](review_final_local_v34.py) | CPU + DeepSeek review of synced final-model outputs on this Windows PC. |
| [`review_math500_answer_v42.py`](review_math500_answer_v42.py) | Local DS review of v41 native failures; preserve full original and final output. |
| [`review_public_palign_math500_v48.py`](review_public_palign_math500_v48.py) | Local DS review of public P-ALIGN 7B v47 native failures; preserve full original and final output. |
| [`review_public_s11_math500_v44.py`](review_public_s11_math500_v44.py) | Local DS review of public s1.1-7B v43 native failures; preserve full original and final output. |
| [`review_romescape_first_round_v55.py`](review_romescape_first_round_v55.py) | Review only native failures in RomEscape v52 round zero. Local CPU/API only. |
| [`review_romescape_qfft_math500_v59.py`](review_romescape_qfft_math500_v59.py) | Review only native failures in RomEscape QFFT v58 single round. Local CPU/API only. |
| [`run_pilot16_rope_decode_v39.py`](run_pilot16_rope_decode_v39.py) | 16-question diagnostic: native/YaRN4 x greedy/sampling; GPU generation only. |
| [`start_aimo_sft_v26.py`](start_aimo_sft_v26.py) | Colab terminal: reconstruct the v26 bundle and start training automatically. |
| [`start_aimo_sft_v26_notify.py`](start_aimo_sft_v26_notify.py) | Local VS Code notification extension receives completion signals from Colab. |
| [`start_aimo_sft_v27.py`](start_aimo_sft_v27.py) | Apply the small v27 Colab compatibility overlay, then train the v26 recipe. |
| [`start_aimo_sft_v28.py`](start_aimo_sft_v28.py) | Read the private SwanLab key automatically, then launch the fixed v26 recipe. |
| [`start_qfft_supervisor_v73.pyw`](start_qfft_supervisor_v73.pyw) | Task Scheduler entry: independent process host, durable stdout/stderr, no retries. |
| [`supplement_incomplete_high_v91.pyw`](supplement_incomplete_high_v91.pyw) | Explicitly authorized one-pass high repair of the effective 62 incomplete questions. |
| [`verify_migration.py`](verify_migration.py) | Offline migration verification; run from any working directory. |
| [`watch_incomplete_high_v91.pyw`](watch_incomplete_high_v91.pyw) | Read-only status window for the 62-question high supplement. |
| [`watch_qfft_length_retry_v86.pyw`](watch_qfft_length_retry_v86.pyw) | Read-only monitor. Closing the window never stops the two-worker task. |
| [`watch_qfft_progress_v72.pyw`](watch_qfft_progress_v72.pyw) | Read-only native progress window; closing this window does not stop generation. |
