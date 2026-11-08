"""Build a read-only, deterministic human review packet. No API or training.

Run from repository root with the existing CPU Python. Full source solutions and
reference programs are review-only and must not be sent to synthesis providers.
"""
import argparse
import ast
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
SEED = "human-review-pack-v1-20261023"


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def jread(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def records(path):
    with Path(path).open(encoding="utf-8") as f:
        for no, line in enumerate(f, 1):
            if line.strip():
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ValueError(f"{path}:{no}: expected object")
                yield row


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def link(path):
    p = Path(path)
    return f"[{p.name}](<{p.resolve().as_posix()}>)"


def fenced(text, language="text"):
    # Never allow an untrusted source to close the Markdown fence.
    fence = "`" * max(4, max((len(t) for t in str(text).split() if set(t) == {"`"}), default=0) + 1)
    return f"{fence}{language}\n{text}\n{fence}\n"


def validate_manifest(path, inventory):
    m = jread(path)
    for filename, expected in m.get("input_identity", {}).items():
        target = ROOT / filename
        actual = sha(target)
        if actual != expected:
            raise ValueError(f"Manifest input identity mismatch: {target}")
        inventory[target.relative_to(ROOT).as_posix()] = {
            "sha256": actual, "hash_matches": True, "kind": "manifest input identity"}
    for key, value in m.get("files", {}).items():
        if isinstance(value, str):
            target, expected, count = path.parent / key, value, None
        else:
            target = ROOT / value.get("path", str(path.parent.relative_to(ROOT) / key))
            expected, count = value["sha256"], value.get("rows")
        actual = sha(target)
        if actual != expected:
            raise ValueError(f"Manifest SHA mismatch: {target}")
        item = {"path": target.relative_to(ROOT).as_posix(), "sha256": actual,
                "manifest": path.relative_to(ROOT).as_posix(), "hash_matches": True}
        if target.suffix == ".jsonl":
            rows = list(records(target))
            item.update(rows=len(rows), schema_key_variants=dict(Counter(
                ",".join(sorted(r)) for r in rows)))
            if count is not None and len(rows) != count:
                raise ValueError(f"Manifest row mismatch: {target}")
        inventory[item["path"]] = item
    for source in m.get("identity", {}).get("source_files", []):
        target = ROOT / source["path"]
        if sha(target) != source["sha256"]:
            raise ValueError(f"Source parquet mismatch: {target}")
        inventory[target.relative_to(ROOT).as_posix()] = {
            "sha256": source["sha256"], "hash_matches": True, "kind": "full source parquet"}
    for cap, scenario in m.get("scenarios", {}).items():
        target = path.parent / f"sft_train_{cap}.jsonl"
        if target.exists() and sum(1 for _ in records(target)) != scenario["count"]:
            raise ValueError(f"Scenario row mismatch: {target}")
    return m


def stratified(rows, count, category):
    """Cover each subject/interface, rotating length tertiles, then fixed hashes."""
    groups = defaultdict(list)
    for row in rows:
        label = row.get("interface") or row.get("problem_type") or row.get("source") or "unknown"
        groups[label].append(row)
    queues = {}
    for label_index, (label, values) in enumerate(sorted(groups.items())):
        ordered = sorted(values, key=lambda r: (length(r), r["sample_id"]))
        buckets = defaultdict(list)
        for i, row in enumerate(ordered):
            buckets[min(2, i * 3 // len(ordered))].append(row)
        for bucket in buckets.values():
            bucket.sort(key=lambda r: hashlib.sha256(
                (SEED + category + r["sample_id"]).encode()).hexdigest())
        queue = []
        while any(buckets.values()):
            for step in range(3):
                b = (step + label_index) % 3
                if buckets[b]:
                    row = dict(buckets[b].pop(0))
                    row["_sampling_stratum"] = {"subject_or_interface": label, "length_tertile": b}
                    queue.append(row)
        queues[label] = queue
    selected = []
    while len(selected) < min(count, len(rows)):
        for label in sorted(queues):
            if queues[label] and len(selected) < count:
                selected.append(queues[label].pop(0))
    return selected


def length(row):
    return row.get("response_token_count", row.get("answer_token_count",
           row.get("answer_tokens", row.get("prompt_token_count", row.get("token_count", len(row.get("problem", "")))))))


def load_category(path, expected, required, inventory):
    rows = list(records(path))
    if expected is not None and len(rows) != expected:
        raise ValueError(f"Expected {expected} rows, got {len(rows)}: {path}")
    ids = [r["sample_id"] for r in rows]
    if len(set(ids)) != len(ids):
        raise ValueError(f"Repeated sample_id in {path}")
    for i, r in enumerate(rows):
        if not required <= r.keys():
            raise ValueError(f"Schema missing keys at {path}:{i+1}: {required-r.keys()}")
    rel = path.relative_to(ROOT).as_posix()
    inventory.setdefault(rel, {}).update(path=rel, sha256=sha(path), rows=len(rows),
        required_schema=sorted(required), required_schema_pass=True,
        unique_sample_ids=True, schema_key_variants=dict(Counter(
            ",".join(sorted(r)) for r in rows)))
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native-v3", type=Path)
    parser.add_argument("--native-v3-manifest", type=Path)
    parser.add_argument("--geometry-prototypes", type=Path)
    parser.add_argument("--api-candidates", type=Path,
        help="Post-hoc format-review pairs JSON; only pending review, never accepted training")
    parser.add_argument("--per-category", type=int, default=10)
    args = parser.parse_args()
    if not 5 <= args.per_category <= 10:
        parser.error("per-category must be between 5 and 10")
    out = ROOT / "reports/data_human_review_pack"
    out.mkdir(parents=True, exist_ok=True)
    inventory = {}
    manifests = {}
    for name in ["data/math_manifest.json", "data/project1_sft_v2/manifest.json",
                 "data/project2_code/manifest.json", "data/project2_code_v2/manifest.json"]:
        manifests[name] = validate_manifest(ROOT / name, inventory)
    if args.native_v3_manifest:
        manifests[str(args.native_v3_manifest)] = validate_manifest(
            args.native_v3_manifest.resolve(), inventory)
    code_sources = jread(ROOT / "data/screening_sources/code_source.json")
    for source in code_sources["files"]:
        target = ROOT / source["path"]
        actual = sha(target)
        if actual != source["sha256"]:
            raise ValueError(f"Pinned code source parquet mismatch: {target}")
        inventory[source["path"]] = {"sha256": actual, "hash_matches": True,
            "kind": "full source parquet", "manifest": "data/screening_sources/code_source.json"}
    req = {"sample_id", "problem"}
    math_train = load_category(ROOT / "data/math/math_train.jsonl", 7184,
        req | {"solution", "reference_answer", "level", "problem_type"}, inventory)
    math_dev = load_category(ROOT / "data/math/math_dev.jsonl", 256,
        req | {"solution", "reference_answer", "level", "problem_type"}, inventory)
    math_lookup = {r["sample_id"]: r for r in math_train + math_dev + list(records(ROOT / "data/math/gsm8k_train.jsonl"))}
    categories = []

    def add(key, title, path, rows, note):
        selected = stratified(rows, args.per_category, key)
        categories.append({"key": key, "title": title, "path": str(path),
            "population": len(rows), "note": note, "samples": selected})

    native = ROOT / "data/project1_sft_v2/sft_train_3000.jsonl"
    native_rows = load_category(native, 1500, req | {"solution", "reference_answer", "training_approved"}, inventory)
    if not args.native_v3:
        add("p1_native3000", "项目一原生3000候选（历史对照，不代表v3已审核）", native,
            native_rows, "保留源长轨迹；语义/过程审核尚未封板，不能写成训练已批准。")
    if args.native_v3:
        v3 = args.native_v3.resolve()
        v3rows = load_category(v3, 1500, req | {"solution", "reference_answer", "training_approved"}, inventory)
        add("p1_native_v3", "项目一原生净化v3候选", v3, v3rows,
            "净化只改变已登记字段；题意、推导和答案仍须用户确认。")
    if args.api_candidates:
        ap = args.api_candidates.resolve()
        arows = jread(ap)
        if not isinstance(arows, list):
            raise ValueError("API candidate review JSON must be a list")
        is_v3 = "compact_distillation_v3" in str(ap)
        if is_v3:
            validate_manifest(ap.parent / "frozen_run_manifest.json", inventory)
        ai_path = ap.parent / "ai_process_review.json" if is_v3 else ROOT / "reports/compact_distillation_v2_20261023/ai_process_spot_review.json"
        ai_lookup = {r["sample_id"]: r for r in jread(ai_path)["candidate_reviews"]
            if r["cohort"] == "new_pilot"} if ai_path.exists() else {}
        reviewed_path = ap.parent / "reviewed_common_candidates.json"
        reviewed_lookup = {r["sample_id"]: r for r in jread(reviewed_path)} if is_v3 and reviewed_path.exists() else {}
        for row in arows:
            if not {"sample_id", "problem", "compact_solution", "reference_answer"} <= row.keys():
                raise ValueError("Missing API review candidate schema")
            row["solution"] = row["compact_solution"]
            row["_ai_process_observation"] = ai_lookup.get(row["sample_id"])
            row["_typed_candidate_status"] = reviewed_lookup.get(row["sample_id"], {}).get("candidate_status")
        inventory[ap.relative_to(ROOT).as_posix()] = {"sha256": sha(ap), "rows": len(arows),
            "kind": "v3 frozen gate candidates pending human process review" if is_v3 else "post-hoc pending candidates; not independent gate"}
        add("p1_api_candidates", "项目一API精炼v3冻结准入候选（待人工审核）" if is_v3 else "项目一API精炼待审候选（事后格式稳健性重审）", ap, arows,
            "冻结自动准入只验证答案/结构等登记条件；中间过程和源题质量仍待用户签字。" if is_v3 else "这些不是独立pilot的正式通过集；格式规范化没有认证中间过程，AI抽查也不能代替用户签字。")
    add("p1_rl_train", "项目一数学RL训练池", ROOT / "data/math/math_train.jsonl", math_train,
        "7184是题池，不是已按7B真实难度筛出的最终RL题；参考解答仅供审核。")
    add("p1_math_dev", "项目一数学dev", ROOT / "data/math/math_dev.jsonl", math_dev,
        "开发集不能送入SFT/API生成训练解答，不能据此修改最终测试。")
    mp = ROOT / "data/project2_code/math_train_prompts.jsonl"
    pool = load_category(mp, 2500, req | {"source", "domain"}, inventory)
    enriched = []
    for row in pool:
        ref = math_lookup.get(row["sample_id"])
        if ref is None or ref["problem"] != row["problem"]:
            raise ValueError(f"Math source join mismatch {row['sample_id']}")
        enriched.append({**ref, **row, "_review_reference_source": "data/math/math_train.jsonl or gsm8k_train.jsonl"})
    add("p2_math_train", "项目二数学train（1500 MATH＋1000 GSM8K）", mp, enriched,
        "训练文件本身只有题目；本报告本地关联参考解答，不回写prompt也不发API。")
    add("p2_math_dev", "项目二共享数学dev", ROOT / "data/math/math_dev.jsonl", math_dev,
        "与项目一共享同一256题及SHA，不是第二份独立留出集。")
    cp = ROOT / "data/project2_code_v2/train_prompts.jsonl"
    code_train = load_category(cp, 2500, req | {"interface", "source", "prompt"}, inventory)
    cd = ROOT / "data/project2_code_v2/dev_verification.jsonl"
    code_dev = load_category(cd, 200, req | {"interface", "reference_code"}, inventory)
    manifest = manifests["data/project2_code_v2/manifest.json"]
    for rows, field in [(code_train, "train_interfaces"), (code_dev, "dev_interfaces")]:
        if dict(Counter(r["interface"] for r in rows)) != manifest[field]:
            raise ValueError(f"Interface quota mismatch: {field}")
    add("p2_code_train", "项目二代码v2 train", cp, code_train,
        "1250函数＋1250标准输入输出；多数训练题仅静态筛选，不能称已执行通过。")
    add("p2_code_dev", "项目二代码v2 dev", cd, code_dev,
        "100函数＋100标准输入输出；参考程序通过源用例不证明规格与覆盖完整。")
    wanted = {r["sample_id"] for c in categories if c["key"] == "p2_code_train" for r in c["samples"]}
    refs = {}
    for filename in ["data/project2_code_v2/function_candidates.jsonl",
                     "data/project2_code_v2/dev_verification.jsonl"]:
        for row in records(ROOT / filename):
            if row["sample_id"] in wanted:
                refs[row["sample_id"]] = (row, filename)
    # Standard-input training references were deliberately not exported to prompts.
    # Recover by exact source problem_id from pinned local parquet, never by row index.
    missing = wanted - refs.keys()
    if missing:
        import pyarrow.parquet as pq
        for source in sorted((ROOT / "data/screening_sources/code/data").glob("*.parquet")):
            for batch in pq.ParquetFile(source).iter_batches(batch_size=128,
                columns=["problem_id", "problem_statement", "gold_standard_solution", "verification_info"]):
                for row in batch.to_pylist():
                    sid = "code/" + str(row["problem_id"])
                    if sid in missing:
                        refs[sid] = ({"problem": row["problem_statement"],
                            "gold_standard_solution": row["gold_standard_solution"],
                            "verification_info": row["verification_info"]}, source.relative_to(ROOT).as_posix())
            if wanted <= refs.keys():
                break
    for category in categories:
        if category["key"] == "p2_code_train":
            for row in category["samples"]:
                if row["sample_id"] not in refs:
                    raise ValueError(f"Missing review-only code reference: {row['sample_id']}")
                ref, source = refs[row["sample_id"]]
                if row["problem"] != ref["problem"]:
                    raise ValueError(f"Code source problem mismatch: {row['sample_id']}")
                row.update({k: ref[k] for k in ["reference_code", "gold_standard_solution", "verification_info", "test_cases"] if k in ref})
                row["_review_reference_source"] = source
                if row["interface"] == "stdin_stdout":
                    from src.data.screen_code_interfaces import python_code
                    code, tree = python_code(row["gold_standard_solution"])
                    if hashlib.sha256(ast.dump(tree, include_attributes=False).encode()).hexdigest() != row["reference_ast_sha256"]:
                        raise ValueError(f"Recovered code AST mismatch: {row['sample_id']}")
                    row["reference_code"] = code
    if args.geometry_prototypes:
        geo = args.geometry_prototypes.resolve()
        gmanifest = ROOT / "reports/project3_perception_prototype/manifest.json"
        gm = jread(gmanifest)
        for filename, expected in {**gm["inputs"], **gm["artifacts"]}.items():
            target = Path(filename)
            if sha(target) != expected:
                raise ValueError(f"Geometry artifact SHA mismatch: {target}")
            inventory[target.relative_to(ROOT).as_posix()] = {
                "sha256": expected, "hash_matches": True,
                "manifest": gmanifest.relative_to(ROOT).as_posix()}
        preview_manifest = ROOT / "reports/project3_perception_review/manifest.json"
        if preview_manifest.exists():
            preview = jread(preview_manifest)
            for filename, expected in {**preview["inputs"], **preview["artifacts"]}.items():
                target = Path(filename)
                if sha(target) != expected:
                    raise ValueError(f"Geometry review preview SHA mismatch: {target}")
                inventory[target.relative_to(ROOT).as_posix()] = {
                    "sha256": expected, "hash_matches": True,
                    "manifest": preview_manifest.relative_to(ROOT).as_posix()}
        rows = list(records(geo))
        if len(rows) != 200:
            raise ValueError("Expected 200 geometry prototypes")
        inventory[geo.relative_to(ROOT).as_posix()] = {"sha256": sha(geo), "rows": len(rows)}
        for row in rows:
            row.setdefault("sample_id", str(row.get("original_id", row.get("id"))))
            row.setdefault("problem", row.get("question", ""))
            row.setdefault("problem_type", str(row.get("split", "geometry")))
            if row.get("training_approved") is not False:
                raise ValueError("Geometry prototype must remain unapproved")
        add("p3_geometry", "项目三视觉独占原型待审", geo, rows,
            "这是原型审核，不是视觉独占金标；图像、关联、文字泄露和关系语义均待用户确认。")
    meta = {"version": SEED, "script_sha256": sha(__file__), "api_calls": 0,
        "gpu_operations": 0, "training_runs": 0, "human_review_status": "pending",
        "sampling": "subject/interface round robin; length tertiles within each subject with starting tertile rotated by sorted subject index; fixed SHA256 order within each tertile; same seed; no quality cherry picking",
        "length_definition": "Existing token field when present; otherwise problem character count. Tertiles within each subject/interface; not tokenizer-equivalent across datasets.",
        "inventory": inventory, "categories": [{k: v for k, v in c.items() if k != "samples"} for c in categories],
        "selected": [], "pending_insertions": [x for x, yes in [("native_v3", bool(args.native_v3)), ("api_candidates", bool(args.api_candidates)), ("geometry_prototypes", bool(args.geometry_prototypes))] if not yes]}
    if args.api_candidates and "compact_distillation_v3" in str(args.api_candidates):
        for name in ["reviewed_common_candidates.json", "final_data_status.json", "source_review_flags.json"]:
            target = args.api_candidates.resolve().parent / name
            if target.exists():
                inventory[target.relative_to(ROOT).as_posix()] = {"sha256": sha(target),
                    "kind": "post-freeze supplementary review; original gate preserved"}
    text = ["# 三项目数据人工审核包", "", "**请用户逐题确认后再封板。全部人工状态为 pending；固定抽样不保证全量质量，也不是统计置信度或98%质量承诺。**", "",
        "完整题面、原生解答、参考代码只用于本地审核；没有发送API，没有执行数据集代码、训练或GPU操作。", "",
        "重建命令：`tiny_posttrain/.venv/Scripts/python.exe scripts/build_data_human_review_pack.py`；额外来源通过 `--native-v3 PATH --native-v3-manifest PATH --api-candidates PATH --geometry-prototypes PATH` 插入。已填写的人工模板会保留。", "",
        "## 全量身份检查", "", "构建前对原manifest中全部files条目做SHA核对、JSONL解析和schema键集合计数；有rows字段时严格核对。指定训练/dev再逐行核对必需字段、唯一ID和固定数量，代码接口配额核对manifest；任何不符抛错并停止。全量检查记录见 " + link(out / "metadata.json") + "。", "",
        "## 仍需完成的质量门槛", "", "- 原生/API数学：源题净化不改变数学条件；缺图/缺题/歧义隔离；本地答案独立验证；中间推导与分类讨论核查；重复草稿与冗长轨迹质量处置。", "- RL/OPD数学：原Level并非模型实测难度；参考解答并非已验证金标；正式RL题需实际正确率筛选，dev严格不进入训练。", "- 代码：逐题输入/返回协议、浮点容差、排序/多合法答案、边界与异常测试；参考实现过源测试只是一致性证据；训练题静态筛选与dev执行证据分开。", "- 几何：核对图、ID关联、原始人工关系、视觉独占性、文字可推出条件、跨划分图像复用与MathVista重叠；原型不等于金标。", "- 本包仅为固定分层样例；用户接受样例后仍需处理全量待审/隔离队列，不能由20或更多样例推断全量可靠。", ""]
    text[2:2] = ["## 九类样例快速导航", ""] + [f"- [{c['title']}](#{c['key']})：全量 {c['population']}，展示 {len(c['samples'])} 条。" for c in categories] + [""]
    text += ["## 专项审核工作单", "", "- 项目三30条双ID关联审核：" + link(ROOT / "reports/project3_perception_review/manual_link_review_30.md"), "- 项目三200题规范化预览：" + link(ROOT / "reports/project3_perception_review/canonical_preview_说明与200题预览.md"), "- 项目三200题原型详情：" + link(ROOT / "reports/project3_perception_prototype/prototype_200.md"), "- 项目三图像/事实审核待填：" + link(ROOT / "reports/project3_perception_prototype/manual_image_review_30_pending.md"), ""]
    source_flags = ROOT / "reports/compact_distillation_v3_20261023/source_review_flags.json"
    if source_flags.exists():
        overlay = jread(source_flags)
        text += ["## 原生净化池仍有源格式待审", "", f"1500题冻结池的补充全量扫描标记 **{overlay['flagged_questions']}题**：空单位markup、空标题、翻译元数据残留、solution/answer标题及参考开头关系表达式，类别可能重叠。这不是79题都数学错误，不能自动删除数学内容；请逐项人工修订/隔离。补充队列见 " + link(source_flags) + "。该overlay不修改冻结池或本包固定样本。", ""]
    if args.api_candidates and "compact_distillation_v3" in str(args.api_candidates):
        text += ["## 本轮API v3的已知失败不能隐藏", "", "冻结记录为30题、14个自动候选、12个共同候选，gate=false；所有候选均待用户审核。12对AI双轨迹复核为8 pass、2 isolate、2 uncertain，见 " + link(ROOT / "reports/compact_distillation_v3_20261023/ai_process_review.md") + "。本包从12中固定分层取10，不根据AI判断挑样。", "", "- `0e10fb1...` 精炼推导错误排除合法a=b分支；结论正确也不可训练。", "- `8937d875...` 源答案指数/不等号与题面冲突，Math-Verify仍错误返回True；应隔离源标签与关系比较风险。", "- 五圆相切缺外切条件、价格题平均价/固定价假设仍待澄清。", "- 两条原生多boxed答案mismatch属于集合聚合协议问题，不是已确认原生数学错误；保留冻结失败记录。", ""]
        text += ["补充保守类型重审保留 **9 对**，关系式/不等式与多个不同boxed进入任务相关人工待审，不直接判数学错误。结合上述AI过程复核，只有 **6 对**同时满足类型门槛与AI pass；这6对仍待用户审核，训练批准为0。它们不替代或覆盖冻结14/12与AI8的历史口径。逐题状态见 " + link(ROOT / "reports/compact_distillation_v3_20261023/reviewed_common_candidates.json") + "；完整状态见 " + link(ROOT / "reports/compact_distillation_v3_20261023/final_data_status.json") + "。", ""]
    template = []
    for category in categories:
        text += [f"<a id=\"{category['key']}\"></a>", "", "## " + category["title"], "", f"全量 {category['population']} 条；本包 {len(category['samples'])} 条。{category['note']}", "", "源文件：" + link(category["path"]), ""]
        for i, row in enumerate(category["samples"], 1):
            sid = row["sample_id"]
            clean = {k: v for k, v in row.items() if not k.startswith("_")}
            item = {"category": category["key"], "sample_id": sid, "record_sha256": digest(clean),
                "source": category["path"], "stratum": row["_sampling_stratum"],
                "human_status": "pending", "reviewer": "", "decision": "", "notes": ""}
            meta["selected"].append(item)
            template.append(item)
            text += [f"### {i}. {sid}", "", f"状态：**人工待审**。领域/接口：{row.get('problem_type', row.get('interface', row.get('domain', 'unknown')))}；等级：{row.get('level', row.get('linked_train_level', 'unknown'))}；长度字段：{length(row)}。", "", f"SHA256（审核关联记录）：`{item['record_sha256']}`。抽样层：`{row['_sampling_stratum']}`。", "", "题面全文：", "", fenced(row["problem"])]
            if row.get("interface"):
                text += ["接口契约：", "", fenced(row.get("function_signature") or "stdin/stdout 完整程序"), "参考代码全文（仅审核）：", "", fenced(row.get("reference_code") or row.get("gold_standard_solution") or "未包含参考代码，应隔离核查", "python")]
                cases = row.get("test_cases") or (row.get("verification_info") or {}).get("test_cases") or []
                preview = json.dumps(cases[:2], ensure_ascii=False, indent=2)
                text += [f"测试摘要：源用例 {len(cases)} 个，完整用例SHA `{digest(cases)}`；执行状态 `{row.get('execution_status', '仅静态；本包未执行')}`。", "", "前两用例预览（最多2400字符，全文见来源文件）：", "", fenced(preview[:2400] + ("\n[展示截断]" if len(preview) > 2400 else ""), "json"), "参考来源：" + link(ROOT / row.get("_review_reference_source", category["path"])), ""]
            else:
                text += ["参考答案：", "", fenced(row.get("reference_answer", row.get("answer", "未提供"))), "原生/参考解答全文：", "", fenced(row.get("solution", row.get("compact_solution", "未提供文字解答；须结合图像及原始标注审核")))]
                if category["key"] == "p1_api_candidates":
                    text += ["同题原生长轨迹全文（对照审核）：", "", fenced(row["raw_solution"])]
                    ai = row.get("_ai_process_observation")
                    if ai:
                        text += [f"已有AI辅助观察（不是人工签字）：**{ai['decision']}**。{ai['reason']}；{ai['quality_observation']}。", ""]
                    if row.get("_typed_candidate_status"):
                        text += [f"补充类型＋过程联合状态：`{row['_typed_candidate_status']}`；所有状态均未获得人工训练批准。", ""]
                for key in ["image_path", "image", "diagram_path", "diagram_image_path"]:
                    image = row.get(key)
                    if isinstance(image, str) and (ROOT / image).is_file():
                        text += [f"![待审几何图](<{(ROOT / image).resolve().as_posix()}>)", ""]
                if category["key"] == "p3_geometry":
                    text += ["原型标注全文：", "", fenced(json.dumps(row, ensure_ascii=False, indent=2), "json")]
            text += ["人工填写：题意完整 [ ]；解答/代码符合题意 [ ]；过程/用例充分 [ ]；accept/isolate/uncertain：____；理由：____。", ""]
    if meta["pending_insertions"]:
        text += ["## 待插入来源", "", ", ".join(meta["pending_insertions"]) + " 尚未纳入本次生成；不能把占位视为审核完成。", ""]
    # Keep user-written decisions intact across rebuilds.
    tpath = out / "human_review_template.jsonl"
    existing = {(r["category"], r["sample_id"]): r for r in records(tpath)} if tpath.exists() else {}
    for row in template:
        previous = existing.get((row["category"], row["sample_id"]))
        if previous:
            if previous["record_sha256"] != row["record_sha256"]:
                signed = previous.get("reviewer") or previous.get("decision") or previous.get("notes") or previous.get("human_status", "pending") != "pending"
                if signed:
                    raise ValueError("Existing signed human review record changed; refuse overwrite")
            else:
                row.update(previous)
    current_keys = {(r["category"], r["sample_id"]) for r in template}
    for key, previous in existing.items():
        if key not in current_keys and (previous.get("reviewer") or previous.get("decision") or previous.get("notes") or previous.get("human_status", "pending") != "pending"):
            template.append({**previous, "not_selected_in_current_build": True})
    tpath.write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in template), encoding="utf-8")
    (out / "metadata.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (ROOT / "reports/data_human_review_pack.md").write_text("\n".join(text), encoding="utf-8")
    print(json.dumps({"categories": len(categories), "samples": len(template), "inventory_files": len(inventory), "pending_insertions": meta["pending_insertions"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
