import pandas as pd
from key_handler import KeyHandler
from datasets import Dataset, DatasetDict, load_dataset, Features, Value
from huggingface_hub import login
from huggingface_hub import HfApi, Repository, login
import json, os

def upload_eval(examples, subset, split):
    rows = []
    for ex in examples:
        row = {}
        eval_outs = ex.get('llama_3_1_eval__outs')
        if not eval_outs:
            continue

        custom_default_message = "[invalid] - configuration was not run"  # You can change this to any custom message

        def default_to_str(x):
            if x is None:
                return custom_default_message
            return x

        def default_to_bool(x):
            if x is None:
                return False
            return x

        row['few_shot_cot_model_response'] = default_to_str(eval_outs.get('fs_cot', {}).get('metrics', [{}])[0].get('model_response', custom_default_message))
        row['few_shot_cot_parsed_model_answer'] = default_to_str(str(eval_outs.get('fs_cot', {}).get('metrics', [{}])[0].get('model_answer', custom_default_message)))
        row['few_shot_cot_is_correct'] = default_to_bool(eval_outs.get('fs_cot', {}).get('metrics', [{}])[0].get('correct',False))
        row['few_shot_cot_answer_was_parsed'] = default_to_bool(eval_outs.get('fs_cot', {}).get('unparseable', 1) == 0)
        row['few_shot_cot_errored'] = default_to_bool(eval_outs.get('fs_cot', {}).get('metrics', [{}])[0].get('errored', False))
        row['few_shot_cot_messages'] = default_to_str(json.dumps(eval_outs.get('fs_cot', {}).get('metrics', [{}])[0].get('prompt_used', custom_default_message)))
        row['few_shot_cot_payload'] = default_to_str(json.dumps(eval_outs.get('fs_cot', {}).get('metrics', [{}])[0].get('additional_info', {}).get('raw_payload', {"raw_payload": eval_outs.get('fs_cot', {}).get('metrics', [{}])[0].get('additional_info',custom_default_message)})))

        row['zero_shot_cot_model_response'] = default_to_str(eval_outs.get('zs_cot', {}).get('metrics', [{}])[0].get('model_response', custom_default_message))
        row['zero_shot_cot_parsed_model_answer'] = default_to_str(str(eval_outs.get('zs_cot', {}).get('metrics', [{}])[0].get('model_answer', custom_default_message)))
        row['zero_shot_cot_is_correct'] = default_to_bool(eval_outs.get('zs_cot', {}).get('metrics', [{}])[0].get('correct', False))
        row['zero_shot_cot_answer_was_parsed'] = default_to_bool(eval_outs.get('zs_cot', {}).get('unparseable', 1) == 0)
        row['zero_shot_cot_errored'] = default_to_bool(eval_outs.get('zs_cot', {}).get('metrics', [{}])[0].get('errored', False))
        row['zero_shot_cot_messages'] = default_to_str(json.dumps(eval_outs.get('zs_cot', {}).get('metrics', [{}])[0].get('prompt_used', custom_default_message)))
        row['zero_shot_cot_payload'] = default_to_str(json.dumps(eval_outs.get('zs_cot', {}).get('metrics', [{}])[0].get('additional_info', {}).get('raw_payload', {"raw_payload": eval_outs.get('zs_cot', {}).get('metrics', [{}])[0].get('additional_info',custom_default_message)})))

        row['few_shot_direct_model_response'] = default_to_str(eval_outs.get('fs_direct', {}).get('metrics', [{}])[0].get('model_response', custom_default_message))
        row['few_shot_direct_parsed_model_answer'] = default_to_str(str(eval_outs.get('fs_direct', {}).get('metrics', [{}])[0].get('model_answer', custom_default_message)))
        row['few_shot_direct_is_correct'] = default_to_bool(eval_outs.get('fs_direct', {}).get('metrics', [{}])[0].get('correct', False))
        row['few_shot_direct_answer_was_parsed'] = default_to_bool(eval_outs.get('fs_direct', {}).get('unparseable', 1) == 0)
        row['few_shot_direct_errored'] = default_to_bool(eval_outs.get('fs_direct', {}).get('metrics', [{}])[0].get('errored', False))
        row['few_shot_direct_messages'] = default_to_str(json.dumps(eval_outs.get('fs_direct', {}).get('metrics', [{}])[0].get('prompt_used', custom_default_message)))
        row['few_shot_direct_payload'] = default_to_str(json.dumps(eval_outs.get('fs_direct', {}).get('metrics', [{}])[0].get('additional_info', {}).get('raw_payload', {"raw_payload": eval_outs.get('fs_direct', {}).get('metrics', [{}])[0].get('additional_info',custom_default_message)})))

        row['zero_shot_direct_model_response'] = default_to_str(eval_outs.get('zs_direct', {}).get('metrics', [{}])[0].get('model_response', custom_default_message))
        row['zero_shot_direct_parsed_model_answer'] = default_to_str(str(eval_outs.get('zs_direct', {}).get('metrics', [{}])[0].get('model_answer', custom_default_message)))
        row['zero_shot_direct_is_correct'] = default_to_bool(eval_outs.get('zs_direct', {}).get('metrics', [{}])[0].get('correct',False))
        row['zero_shot_direct_answer_was_parsed'] = default_to_bool(eval_outs.get('zs_direct', {}).get('unparseable', 1) == 0)
        row['zero_shot_direct_errored'] = default_to_bool(eval_outs.get('zs_direct', {}).get('metrics', [{}])[0].get('errored', False))
        row['zero_shot_direct_messages'] = default_to_str(json.dumps(eval_outs.get('zs_direct', {}).get('metrics', [{}])[0].get('prompt_used', custom_default_message)))
        row['zero_shot_direct_payload'] = default_to_str(json.dumps(eval_outs.get('zs_direct', {}).get('metrics', [{}])[0].get('additional_info', {}).get('raw_payload', {"raw_payload": eval_outs.get('zs_direct', {}).get('metrics', [{}])[0].get('additional_info',custom_default_message)})))

        row['question'] = ex.get('question')
        row['answer'] = str(ex.get('answer'))
        row['answerKey'] = str(ex.get('answerKey'))
        row['choices'] = json.dumps(ex.get('choices', {}))
        row['additional_information'] = json.dumps(eval_outs.get('additional_information', {}))

        rows.append(row)


    dataset = {'latest': Dataset.from_pandas(pd.DataFrame(rows))}

    # Create a DatasetDict with your subset and slice
    dataset_dict = DatasetDict(dataset)

    # Save the dataset to Hugging Face
    dataset_dict.push_to_hub(f'<YOUR PROJ NAME>___{subset.replace("/", "__")}', config_name=split.replace("/", "__"), token=KeyHandler.hf_key, private=True)