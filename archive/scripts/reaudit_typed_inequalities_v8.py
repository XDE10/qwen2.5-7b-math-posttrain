"""Bounded offline typed/role probe on27 known development cases, not a pilot."""
import json,sys,subprocess
from scripts.finalfill_common import *
PROFILES={**{k:{'kind':'one_variable_solution_set','domain':'real','variable':'x'} for k in ['8a7b6316','ce7ab0a7','1f83bde7','3eac3eb0','8e7886d9','19e100c9','17523372','a599c034','a5f6c2bc','3961d733','997a9bcc','2e53c7ca','72bf1968','5cac2ab0']},
 '228a23c1':{'kind':'one_variable_solution_set','domain':'real','variable':'a'},'1ffc2ec3':{'kind':'one_variable_solution_set','domain':'real','variable':'k'},'736d95ea':{'kind':'one_variable_solution_set','domain':'real','variable':'k'},'ce835970':{'kind':'one_variable_solution_set','domain':'real','variable':'b'},
 '6f0b409f':{'kind':'one_variable_solution_set','domain':'integer','variable':'x','enumerated_roots':True},'96523155':{'kind':'one_variable_solution_set','domain':'integer','variable':'x'},'a56c98d5':{'kind':'one_variable_solution_set','domain':'integer','variable':'x'}}
EXCLUSIONS={'19faae1d':'ordered_two_variable_equality_case_not_univariate_set','27a6ea59':'function_family_not_univariate_set','894b4674':'fractional_part_of_x_not_x_solution_set','5b4acf7d':'source_exponent_V_bar_x_ambiguous_do_not_guess_sqrt','22e8f48e':'proof_plus_equality_case_reference_incomplete','3e64f527':'multiple_questions_and_parameter_sequences_not_one_answer'}
def worker(r):
    from src.synthesis.typed_answers_v8 import check_roles,solution_set
    from scripts.refill_project1_native_v5 import source_filter
    p=PROFILES.get(r['sample_id'][:8]);reason=EXCLUSIONS.get(r['sample_id'][:8])
    if not p:return {'sample_id':r['sample_id'],'technical_candidate':False,'reason':reason or 'no_reviewed_profile'}
    try:gold=str(solution_set(r['reference_answer'],p))
    except Exception as e:return {'sample_id':r['sample_id'],'technical_candidate':False,'profile':p,'reason':str(e)}
    v=check_roles(r['solution'],r['reference_answer'],p);scope=source_filter(r)['v4_triage'];other=list(scope['quality_isolation_reasons'])+list(scope['experiment_scope_exclusions'])
    other += [x for x in scope['manual_review_flags'] if x not in ['malformed_boxes','distinct_boxes_components_collection_or_intermediate_requires_review']]
    return {'sample_id':r['sample_id'],'profile':p,'gold_set':gold,'role_check':v,'other_source_flags':other,'technical_candidate':v['correct'] and not other,'production_approved':False,'requires':'blind generation + original150..1500 gate + both full-trajectory reviews; diagnostic source cases are development not independent acceptance set'}
def run():
    data=rows(REPORT/'inequality_diagnosis/remaining_candidate_details.jsonl');data=[r for r in data if r['category']=='requires_typed_inequality_verifier' or r['source_filter']['v4_triage']['manual_review_flags']]
    results=[]
    for r in data:
        try:
            p=subprocess.run([sys.executable,'-m','scripts.reaudit_typed_inequalities_v8','worker'],input=json.dumps(r),text=True,encoding='utf-8',capture_output=True,timeout=25,cwd=ROOT)
            results.append(json.loads(p.stdout.strip().splitlines()[-1]))
        except Exception as e:results.append({'sample_id':r['sample_id'],'technical_candidate':False,'reason':'bounded_probe_error','error_type':type(e).__name__})
    write(REPORT/'inequality_diagnosis/typed_v8_reaudit.jsonl',results);candidates=[r for r in data if next(x for x in results if x['sample_id']==r['sample_id'])['technical_candidate']];write(DATA/'project1/typed_v8_development_candidates.jsonl',candidates)
    dump(REPORT/'inequality_diagnosis/typed_v8_summary.json',{'examined':len(data),'typed_role_supported':sum(x.get('role_check',{}).get('correct',False) for x in results),'technical_source_candidates':len(candidates),'new_training_approvals':0,'api_calls':0,'not_an_independent_pilot':True,'verifier_sha':sha(ROOT/'src/synthesis/typed_answers_v8.py')});print(read(REPORT/'inequality_diagnosis/typed_v8_summary.json'),flush=True)
if __name__=='__main__':
    if len(sys.argv)>1:print(json.dumps(worker(json.load(sys.stdin))))
    else:run()
