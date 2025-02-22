"""
Originally, for the paper, we used prompts done for the Llama 3.1 evaluations. However, Llama has since changed their
huggingface repo for newer evals and models.

So now, instead of using their prompts, we use the prompts we used for the original paper.

All the code to use Llamas prompts are there, it's just not aligned with the papers results anymore as they change them
often.
"""
import json
from pathlib import Path
import datasets
from typing import Optional
from collections import Counter
from tqdm import tqdm
from functools import partial
from src.utils.paths import ROOT_FOLDER

from src import cache
from src.model import Model
from experiments.utils import rollout, build_prefix_prompt, answer_percentage_from_metrics, rollout_logprobs

from key_handler import KeyHandler
from eval_datasets import AGIEvalDataset, ARCDataset, BigBenchDataset, \
    CSQADataset, GPQADataset, MMLUDataset, MMLUProDataset, MuSRDataset, PIQADataset, SocialIQADataset, StrategyQADataset, \
    WinograndeDataset, ReasoningDataset, ContextHubDataset, MATHDataset, GSM8KDataset, BigBenchHardDataset, \
    BigGenBenchDataset, MusiqueDataset, FOLIODataset, MusiqueDatasetAll
import concurrent
from experiments.helper_prompts.router import switch
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy

KeyHandler.set_env_key()

def is_math(datasetname):
    if 'math' in datasetname or 'gsm8k' in datasetname:
        return True
    return False


def handle_rollout(prefix, model, prompt, ex, temperature, top_p, rollout_size, batch_size, dataset_token_len, info,
                   prepend_to_out=None, comp_prompt=None, use_bos_token=True, is_4k:bool=False, eval_model=None):
    vllm_kwargs = {'seed': 42}
    acc, unp, met = rollout(model, prompt, ex, num_rollouts=rollout_size, batch_size=batch_size,
                            completion_length=dataset_token_len, temperature=temperature, top_p=top_p,
                            prepend_to_out=prepend_to_out, use_bos_token=use_bos_token, comp_prompt=comp_prompt, vllm_sampling_kwargs=vllm_kwargs, evaluation_model=eval_model)
    return {
        'prefix': prefix,
        'prompt': prompt,
        'accuracy': acc,
        'unparseable': unp,
        'metrics': met,
        'answer_distribution': answer_percentage_from_metrics(met) if not is_math(ex['dataset_type']) else {},
        'info': info
    }

def run_experiment(
        datasetname: str,
        model,
        dataset,
        output_folder: Optional[Path] = None,
        offset: int = 0,
        sample_size: int = 100,
        push_to_hub: bool = False,
        is_4k_model: bool = False,
        is_closed_source: bool = False,
        skip_fs_cot: bool = False,
        skip_fs_direct: bool = False,
        skip_zs_cot: bool = False,
        skip_zs_direct: bool = False,
        eval_model: Model = None,
        taur_rows=None
):
    if eval_model is not None and dataset.requires_eval_model is False:
        eval_model = None

    dataset_token_lens = {
        'mmlu': [10, 1024] if not is_4k_model else [10, 800],
        'mmlu_pro': [10, 1024],
        'math': [100, 2048] if not is_4k_model else [100, 1500],
        "gsm8k": [100, 1024],
        "csqa": [10, 512],
        "mm_musr": [10, 2048] if not is_4k_model else [10, 756],
        "ta_musr": [10, 2048] if not is_4k_model else [10, 756],
        "op_musr": [10, 2048] if not is_4k_model else [10, 756],
        "gpqa": [10, 2048] if not is_4k_model else [10, 2048, 400],
        "stratqa": [10, 512],
        'contexthub_deductive_level1': [10, 1024],
        'contexthub_deductive_level2': [10, 1024],
        'contexthub_deductive_level3': [10, 1024],
        'contexthub_deductive_level4': [10, 1024],
        'contexthub_abductive_level1': [10, 1024],
        'contexthub_abductive_level2': [10, 1024],
        'contexthub_abductive_level3': [10, 1024],
        'contexthub_abductive_level4': [10, 1024],
        'arc_challenge': [10, 1024],
        'arc_easy': [10, 1024],
        'agieval_lsat_lr': [10, 1024] if not is_4k_model else [10, 756],
        'agieval_lsat_ar': [10, 1024],
        'agieval_lsat_rc': [10, 1024] if not is_4k_model else [10, 756],
        'siqa': [10, 512],
        'piqa': [10, 512],
        'winogrande': [10, 512],
        'bbh': [30, 1024],
        'biggen_bench': [1024, 2048] if not is_4k_model else [512, 1024],
        'musique': [128, 2048],
        'musique_all': [128, 2048],
        'gsm8k_hard': [100, 1024],
        'folio': [100, 1024],
        'folio_cot_solver': [2048, 2048]
    }

    use_llama_prompts = {
        'mmlu': True,
        'mmlu_pro': True,
        'math': True,
        'gsm8k': True,
        "gpqa": True,
        "stratqa": False,
        "mm_musr": False,
        "ta_musr": False,
        "op_musr": False,
        'contexthub_deductive_level1': True,
        'contexthub_deductive_level2': True,
        'contexthub_deductive_level3': True,
        'contexthub_deductive_level4': True,
        'contexthub_abductive_level1': True,
        'contexthub_abductive_level2': True,
        'contexthub_abductive_level3': True,
        'contexthub_abductive_level4': True,
        'csqa': False,
        'arc_challenge': False,
        'arc_easy': False,
        'agieval_lsat_lr': False,
        'agieval_lsat_ar': False,
        'agieval_lsat_rc': False,
        'piqa': False,
        'siqa': False,
        'winogrande': False,
        'bbh': False,
        'biggen_bench': False,
        'musique': False,
        'musique_all': False,
        'gsm8k_hard': False,
        'folio': False,
        'folio_cot_solver': False
    }

    does_not_have_fewshot_setting = {
        'arc_challenge': True,
        'arc_easy': True,
        'piqa': True,
        'siqa': True,
        'winogrande': True,
        'bbh': True,
        'musique': True,
        'musique_all': True,
        'folio': True,
        'folio_cot_solver': True,
        'gsm8k_hard': True,
    }

    examples = dataset[offset:offset + sample_size]
    try:
        original_data = json.load((output_folder / (datasetname + ".json")).open('r')) if (output_folder / (datasetname + ".json")).is_file() else {}
    except Exception as e:
        original_data = {}
        print(f"Error loading the file: {e}")

    scores = {
        'zs_direct': 0,
        'zs_cot': 0,
        'fs_direct': 0,
        'fs_cot': 0
    }
    totals = {
        'zs_direct': 0,
        'zs_cot': 0,
        'fs_direct': 0,
        'fs_cot': 0
    }
    unps = {
        'zs_direct': 0,
        'zs_cot': 0,
        'fs_direct': 0,
        'fs_cot': 0
    }

    fewshot_for_4k = {
        'mmlu': 3,
        'mmlu_pro': 3,
        "mm_musr": 1,
        "ta_musr": 1,
        "op_musr": 1,
        'contexthub_deductive_level3': 1,
        'contexthub_deductive_level4': 1,
        'contexthub_abductive_level3': 1,
        'contexthub_abductive_level4': 1,
    }

    append_to_messages = {
        'csqa': 'Answer: ',
        'stratqa': 'The answer to the last question is, Answer: ',
        'mm_musr': 'Answer: ',
        'ta_musr': 'The best answer is ',
        'bbh': 'The best answer is ',
        'op_musr': 'Answer: ',
        'contexthub_deductive_level1': 'Answer: ',
        'contexthub_deductive_level2': 'Answer: ',
        'contexthub_deductive_level3': 'Answer: ',
        'contexthub_deductive_level4': 'Answer: ',
        'contexthub_abductive_level1': 'Answer: ',
        'contexthub_abductive_level2': 'Answer: ',
        'contexthub_abductive_level3': 'Answer: ',
        'contexthub_abductive_level4': 'Answer: ',
        'agieval_lsat_lr': 'The answer is therefore ',
        'agieval_lsat_ar': 'The answer is therefore ',
        'agieval_lsat_rc': 'The answer is therefore ',
        'arc_challenge': 'Answer: ',
        'arc_easy': 'Answer: ',
        'piqa': 'Answer: ',
        'siqa': 'Answer: ',
        'winogrande': 'Answer: ',
        'biggen_bench': '',
        # 'musique': 'Answer: ',
        'gsm8k': '$\\boxed{',
        # 'math': '$\\boxed{',
        'gsm8k_hard': '$\\boxed{',
        'folio': 'Answer: ',
        'folio_cot_solver': '\t# Answer:'
    }
    use_da_system_prompt = ['ta_musr', 'agieval_lsat_lr', 'agieval_lsat_ar', 'agieval_lsat_rc']
    use_cot_system_prompt = ['ta_musr', 'agieval_lsat_lr', 'agieval_lsat_ar', 'agieval_lsat_rc']

    def prep_messages(msgs, append_assistant_message: str = None, fs_cot=False, include_da_sys_prompt=False, include_cot_sys_prompt=False):
        def setup_prompt_messages(msgs):
            if isinstance(msgs, str):
                return msgs

            custom_system_msg = {'role': 'system',
                             'content': 'You answer questions and only give the answer. You will always return "' + (
                                 'The best answer is [answer_choice].' if not is_math(
                                     datasetname) else "The final answer is $\\boxed{[answer_choice]}$. I hope it is correct.") + '" as your response where you fill in the answer choice with the correct answer.'}

            # TODO - gsm8k needs to be told for fewshot direct to not do CoT.
            if msgs[-1]['role'] == 'user':
                return msgs
            elif is_closed_source:
                # TODO - Special case for closed source models to behave correctly.
                msgs = [custom_system_msg] + msgs
                if is_math(datasetname):
                    msgs[-2]['content'] += '\n\nYou may only give the answer. Begin your response with "The final answer is $\\boxed{".'
                return msgs[:-1]
            return msgs

        msgs = setup_prompt_messages(msgs)
        if is_4k_model and datasetname in fewshot_for_4k and fs_cot:
            new_msgs = []
            ct = 0
            for m in list(reversed(msgs)):
                if m['role'] == 'user':
                    ct += 1
                new_msgs.append(m)
                if ct == fewshot_for_4k[datasetname] + 1:
                    break
            return list(reversed(new_msgs))

        # TODO - not great, we have to implement task specific prefixes to get weaker models to perform well.
        if append_assistant_message and msgs[-1]['role'] != 'assistant' and not is_closed_source:
            msgs.append({'role': 'assistant', 'content': append_assistant_message})
        elif datasetname == 'stratqa' and fs_cot and not is_closed_source:
            msgs.append({'role': 'assistant', 'content': 'To answer the last question let\'s think step by step. '})
        elif datasetname == 'ta_musr' and is_closed_source and include_da_sys_prompt:
            msgs[-1]['content'] += '\n\nYou must follow the instructions. Do not elaborate,  do not explain, only give your answer. If you elaborate no credit will be given.'


        if include_da_sys_prompt:

            msgs = [{'role': 'system', 'content': 'You answer questions. At the end of the question you always give an answer and nothing else. You must pick an answer. You always give only one answer and that one answer is the one you think is best. You always give the answer in the form of the answer choice letter.'}] + [x for x in msgs if x['role'] != 'system']
        if include_cot_sys_prompt:
            msgs = [{'role': 'system', 'content': 'You answer questions by reasoning about them before answering. You always give your reasoning and then your answer. You always give the answer in the form of the answer choice letter. You must pick an answer. You only ever give one answer. If you think multiple answers are correct, you always choose the best one only.'}] + [x for x in msgs if x['role'] != 'system']

        return msgs

    # Create a lookup dictionary outside the thread executor
    taur_dict = {
        (row['question'], str(row['answerKey'])): row
        for row in taur_rows['latest']
    }

    with cache.static_context:

        with ThreadPoolExecutor(max_workers=250) as executor:
            futures = []
            for exidx, ex in enumerate(examples):
                use_the_llama_prompts = use_llama_prompts[datasetname]

                # Quickly fetch the taur_row using the precomputed dictionary
                taur_row = taur_dict.get((ex['question'], str(ex['answerKey'])))

                if taur_row is None:
                    # Handle the case where there is no matching row
                    # (e.g., skip or raise an error)
                    print("SKIPPING")
                    continue
                    # raise Exception("NO ROW")

                def loads(s):
                    try:
                        return json.loads(s)
                    except:
                        return None

                # Now directly load the prompts
                zsd_prompt = loads(taur_row['zero_shot_direct_messages'])
                zsc_prompt = loads(taur_row['zero_shot_cot_messages'])
                fsd_prompt = loads(taur_row['few_shot_direct_messages'])
                fsc_prompt = loads(taur_row['few_shot_cot_messages'])


                if not skip_zs_direct and zsd_prompt is not None:
                    msgs = switch(
                        model.model,
                        datasetname,
                        'zs_direct',
                        prep_messages(deepcopy(zsd_prompt), append_assistant_message = append_to_messages.get(datasetname, None), include_da_sys_prompt=datasetname in use_da_system_prompt)
                    )

                    futures.append(
                        executor.submit(handle_rollout, 'zs_direct', model,
                                        msgs,
                                        ex, 0.0, 1.0, 1, 1,
                                        dataset_token_lens[datasetname][0], {'exidx': exidx, 'setting': 'zs_direct'}, use_bos_token=False,
                                        prepend_to_out=msgs[-1]['content'] if msgs[-1]['role'] == 'assistant' else None,
                                        is_4k=is_4k_model,
                                        eval_model=eval_model
                                        )

                    )
                if not skip_zs_cot and zsc_prompt is not None:
                    futures.append(
                        executor.submit(handle_rollout, 'zs_cot', model,
                                        switch(model.model, datasetname, 'zs_cot', prep_messages(deepcopy(zsc_prompt), include_cot_sys_prompt=datasetname in use_cot_system_prompt)),
                                        ex, 0.0, 1.0, 1, 1,
                                        dataset_token_lens[datasetname][1], {'exidx': exidx, 'setting': 'zs_cot'},
                                        use_bos_token=False, is_4k=is_4k_model,
                                        eval_model=eval_model)

                    )

                if does_not_have_fewshot_setting.get(datasetname, False):
                    continue

                if not skip_fs_direct and not does_not_have_fewshot_setting.get(datasetname, False) and fsd_prompt is not None:
                    msgs = switch(model.model, datasetname, 'fs_direct', prep_messages(deepcopy(fsd_prompt), append_assistant_message = append_to_messages.get(datasetname, None),  include_da_sys_prompt=datasetname in use_da_system_prompt))
                    futures.append(
                        executor.submit(handle_rollout, 'fs_direct', model,
                                        msgs,
                                        ex, 0.0, 1.0, 1, 1,
                                        dataset_token_lens[datasetname][0], {'exidx': exidx, 'setting': 'fs_direct'},
                                        use_bos_token=False,
                                        prepend_to_out=msgs[-1]['content'] if msgs[-1]['role'] == 'assistant' else None,
                                        is_4k = is_4k_model,
                                        eval_model=eval_model
                                        )

                    )

                if not skip_fs_cot and not does_not_have_fewshot_setting.get(datasetname, False) and fsc_prompt is not None:
                    futures.append(
                        executor.submit(handle_rollout, 'fs_cot', model,
                                        switch(model.model, datasetname, 'fs_cot', prep_messages(deepcopy(fsc_prompt), fs_cot=True, include_cot_sys_prompt=datasetname in use_cot_system_prompt)),
                                        ex, 0.0, 1.0, 1, 1,
                                        dataset_token_lens[datasetname][1] if not is_4k_model or len(dataset_token_lens[datasetname]) == 2 else dataset_token_lens[datasetname][2], {'exidx': exidx, 'setting': 'fs_cot'},
                                        use_bos_token=False,
                                        eval_model=eval_model),

                    )

            pbar = tqdm(total=len(futures), desc=f"Running experiment on {dataset.dataset_type}", position=0,
                        leave=True)

            i = 0
            for future in concurrent.futures.as_completed(futures):
                i += 1
                # print(i)
                res = future.result()
                info = res['info']
                setting = info['setting']
                exidx = info['exidx']


                pbar.update(1)

                acc, unp, met = res['accuracy'], res['unparseable'], res['metrics']
                examples[exidx].setdefault('llama_3_1_eval__outs', {'zs_direct': {}, 'zs_cot': {}, 'fs_direct': {}, 'fs_cot': {}})


                score = 1 if acc == 1.0 else 0
                if unp == 1.0:
                    score = 1 / len(examples[exidx]['choices']['text']) if examples[exidx].get('choices') else 0
                examples[exidx]['llama_3_1_eval__outs'][setting] = {
                    'prompt': res['prompt'],
                    'metrics': [{k:v for k, v in x.items() if 'metrics' not in k} for x in met],
                    'score': score,
                    'accuracy': acc,
                    'unparseable': unp
                }

                if 'mmlu' == datasetname:
                    if examples[exidx]['llama_3_1_eval__outs'].get('additional_information') is None:
                        examples[exidx]['llama_3_1_eval__outs']['additional_information'] = {'subject': examples[exidx].get('subject')}
                elif 'math' == datasetname.lower():
                    if examples[exidx]['llama_3_1_eval__outs'].get('additional_information') is None:
                        examples[exidx]['llama_3_1_eval__outs']['additional_information'] = {'level': examples[exidx].get('level'), 'type': examples[exidx].get('type')}
                elif 'mmlu_pro' == datasetname:
                    if examples[exidx]['llama_3_1_eval__outs'].get('additional_information') is None:
                        examples[exidx]['llama_3_1_eval__outs']['additional_information'] = {'category': examples[exidx].get('category'), 'src': examples[exidx].get('src')}
                elif 'gpqa' == datasetname:
                    if examples[exidx]['llama_3_1_eval__outs'].get('additional_information') is None:
                        examples[exidx]['llama_3_1_eval__outs']['additional_information'] = {}
                elif 'bbh' == datasetname:
                    if examples[exidx]['llama_3_1_eval__outs'].get('additional_information') is None:
                        examples[exidx]['llama_3_1_eval__outs']['additional_information'] = {'subset': examples[exidx].get('subset')}
                elif 'biggen_bench' == datasetname:
                    if examples[exidx]['llama_3_1_eval__outs'].get('additional_information') is None:
                        examples[exidx]['llama_3_1_eval__outs']['additional_information'] = {'capability': examples[exidx].get('capability'), 'task': examples[exidx].get('task')}



                scores[setting] += score
                totals[setting] += 1
                unps[setting] += 1 if unp == 1.0 else 0

                pbar_descrip = f"ZSC: ACC {scores['zs_cot'] / max(1, totals['zs_cot']):.3f} T: {totals['zs_cot']} [Unp: {unps['zs_cot'] / max(1, totals['zs_cot']):.2f}], ZSD: ACC {scores['zs_direct'] / max(1, totals['zs_direct']):.3f} T: {totals['zs_direct']} [Unp: {unps['zs_direct']/max(1, totals['zs_direct']):.2f}] | FSC: ACC {scores['fs_cot'] / max(1, totals['fs_cot']):.3f} T: {totals['fs_cot']} [Unp: {unps['fs_cot'] / max(1, totals['fs_cot']):.2f}], FSD: ACC {scores['fs_direct'] / max(1, totals['fs_direct']):.3f} T: {totals['fs_direct']} [Unp: {unps['fs_direct']/max(1, totals['fs_direct']):.2f}] | ${model.total_cost:.2f}"
                pbar.set_description(pbar_descrip)

    stats = {
        'zs_direct': {
            'accuracy': scores['zs_direct'] / max(1, totals['zs_direct']),
            'unparseable': unps['zs_direct'] / max(1,totals['zs_direct']),
            'total': totals['zs_direct']
        },
        'zs_cot': {
            'accuracy': scores['zs_cot'] / max(1, totals['zs_cot']),
            'unparseable': unps['zs_cot'] / max(1,totals['zs_cot']),
            'total': totals['zs_cot']
        },
        'fs_direct': {
            'accuracy': scores['fs_direct'] / max(1, totals['fs_direct']),
            'unparseable': unps['fs_direct'] / max(1,totals['fs_direct']),
            'total': totals['fs_direct']
        },
        'fs_cot': {
            'accuracy': scores['fs_cot'] / max(1, totals['fs_cot']),
            'unparseable': unps['fs_cot'] / max(1,totals['fs_cot']),
            'total': totals['fs_cot']
        }
    }
    data = {
        'raw_data': examples,
        "llama_3_1_eval_outs": stats,
        **{k: v for k, v in original_data.items() if
           k != 'raw_data' and k != 'llama_3_1_eval_outs'}

    }

    if output_folder:
        with open(output_folder / f"{datasetname}.json", "w") as f:
            json.dump(data, f)

    pbar_descrip = f"ZSC: ACC {scores['zs_cot'] / max(1, totals['zs_cot']):.3f} [Unp: {unps['zs_cot'] / max(1, totals['zs_cot']):.2f}], ZSD: ACC {scores['zs_direct'] / max(1, totals['zs_direct']):.3f} [Unp: {unps['zs_direct'] / max(1, totals['zs_direct']):.2f}] | FSC: ACC {scores['fs_cot'] / max(1, totals['fs_cot']):.3f} [Unp: {unps['fs_cot'] / max(1, totals['fs_cot']):.2f}], FSD: ACC {scores['fs_direct'] / max(1, totals['fs_direct']):.3f} [Unp: {unps['fs_direct'] / max(1, totals['fs_direct']):.2f}]"
    print(pbar_descrip)

    if push_to_hub:
        from experiments.hf_utils import upload_eval
        upload_eval(examples, model.model, datasetname)



    return data if not output_folder else ()




if __name__ == "__main__":
    name_to_dataset = {
        'gpqa': GPQADataset,
        'csqa': CSQADataset,
        'mmlu_pro': MMLUProDataset,
        'mmlu': MMLUDataset,
        'mm_musr': partial(MuSRDataset, path_or_url=ROOT_FOLDER / "eval_datasets/thirdparty/musr/murder_mystery.json"),
        'ta_musr': partial(MuSRDataset, path_or_url=ROOT_FOLDER / "eval_datasets/thirdparty/musr/team_allocation.json"),
        'op_musr': partial(MuSRDataset, path_or_url=ROOT_FOLDER / "eval_datasets/thirdparty/musr/object_placements.json"),
        'stratqa': partial(StrategyQADataset, path_or_url=ROOT_FOLDER / "eval_datasets/thirdparty/strategyqa_dataset/strategyqa_train_filtered.json"),
        "math": MATHDataset,
        "gsm8k": partial(GSM8KDataset, variant="original"),
        "contexthub_deductive_level1": ContextHubDataset,
        "contexthub_deductive_level2": partial(ContextHubDataset, level="data_level2"),
        "contexthub_deductive_level3": partial(ContextHubDataset, level="data_level3"),
        "contexthub_abductive_level1": partial(ContextHubDataset, level="data_level1", logic_type="abductive"),
        "contexthub_abductive_level2": partial(ContextHubDataset, level="data_level2", logic_type="abductive"),
        "contexthub_abductive_level3": partial(ContextHubDataset, level="data_level3", logic_type="abductive"),
        'arc_challenge': partial(ARCDataset, subset='ARC-Challenge'),
        'arc_easy': partial(ARCDataset, subset='ARC-Easy'),
        'agieval_lsat_lr': partial(AGIEvalDataset, prompt_setting='zero-shot-CoT', slice='lsat-lr'),
        'agieval_lsat_ar': partial(AGIEvalDataset, prompt_setting='zero-shot-CoT', slice='lsat-ar'),
        'agieval_lsat_rc': partial(AGIEvalDataset, prompt_setting='zero-shot-CoT', slice='lsat-rc'),
        'piqa': PIQADataset,
        'siqa': SocialIQADataset,
        'winogrande': WinograndeDataset,
        'bbh': BigBenchHardDataset,
        'biggen_bench': BigGenBenchDataset,
        "musique_all": MusiqueDatasetAll,
        "gsm8k_hard": partial(GSM8KDataset, variant="hard"),
        "folio": FOLIODataset,
        "folio_cot_solver": partial(FOLIODataset, used_cot_solver_no_cot_prompt=True, use_llama_3_1_prompts=False),

    }

    [print(x, " ") for x in name_to_dataset.keys()]

    from argparse import ArgumentParser

    parser = ArgumentParser()

    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--eval_model', type=str, required=False)

    parser.add_argument('--output_folder', type=str, required=True)
    parser.add_argument("--datasets", type=str, choices=list(name_to_dataset.keys()), nargs='+', required=True)

    parser.add_argument("--num_samples", type=int, default=100)
    parser.add_argument('--offset', type=int, default=0)
    parser.add_argument('--push_to_hub', type=bool)
    parser.add_argument('--is_4k_model', type=bool)
    parser.add_argument('--is_closed_source', type=bool)
    parser.add_argument('--use_raw_llama3_prompts', type=bool) # TODO: Only for llama 3.1 8B on my machine because of my cache, shouldn't use otherwise :P

    parser.add_argument('--skip_fs_cot', action='store_true')
    parser.add_argument('--skip_fs_direct', action='store_true')
    parser.add_argument('--skip_zs_cot', action='store_true')
    parser.add_argument('--skip_zs_direct', action='store_true')

    parser.add_argument('--bust_cache', action='store_true')

    args = parser.parse_args()


    try:
        cache.enable(bust_cache=args.bust_cache)
    except Exception:
        print("Could not connect to your local redis server, try `redis-server` and `redis-cli` in a terminal to see what's up.  Otherwise, generations will NOT be cached.")

    # Read off arguments
    model = Model.load_model(args.model)
    eval_model = None if not args.eval_model else Model.load_model(args.eval_model)
    output_folder = Path(args.output_folder)
    dataset_names = args.datasets
    num_samples = args.num_samples
    offset = args.offset
    push_to_hub = args.push_to_hub
    is_4k_model = args.is_4k_model
    is_closed_source = args.is_closed_source
    use_raw_llama3_prompts = args.use_raw_llama3_prompts
    skip_fs_cot = args.skip_fs_cot
    skip_fs_direct = args.skip_fs_direct
    skip_zs_cot = args.skip_zs_cot
    skip_zs_direct = args.skip_zs_direct


    for dataset_name in dataset_names:
        try:
            exp_output_folder = output_folder / f'{dataset_name}'
            exp_output_folder.mkdir(parents=True, exist_ok=True)

            if 'mmlu' == dataset_name or 'mmlu_pro' == dataset_name:
                dataset = name_to_dataset[dataset_name](raw_llama31_prompts=False, use_llama_3_1_prompts=False)
            elif 'gsm8k' == dataset_name:
                dataset = name_to_dataset[dataset_name](used_closed_source_prompting=is_closed_source, use_llama_3_1_prompts=False)
            else:
                dataset = name_to_dataset[dataset_name](use_llama_3_1_prompts=False)


            # run experiment
            print(f"Running the experiment on {dataset_name}")
            run_experiment(
                dataset_name,
                model,
                dataset,
                exp_output_folder,
                offset,
                num_samples,
                push_to_hub,
                is_4k_model,
                is_closed_source,
                skip_fs_cot,
                skip_fs_direct, skip_zs_cot, skip_zs_direct, eval_model=eval_model, taur_rows=datasets.load_dataset('TAUR-Lab/Taur_CoT_Analysis_Project___meta-llama__Meta-Llama-3.1-8B-Instruct' if dataset_name != 'mmlu' else 'TAUR-Lab/Taur_CoT_Analysis_Project___Qwen__Qwen2-7B-Instruct', dataset_name))

        except Exception as e:
            raise e
            print(f"ERRORED: {dataset_name}")




