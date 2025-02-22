from copy import deepcopy
from eval_datasets import AGIEvalDataset, ARCDataset, \
    CSQADataset, GPQADataset, MuSRDataset, PIQADataset, SocialIQADataset, StrategyQADataset, WinograndeDataset,\
        MMLUDataset, ContextHubDataset, MATHDataset,ProntoQADataset,GSM8KDataset,FOLIODataset, \
    GSM8KSymDataset,BigBenchHardDataset, BigGenBenchDataset, MusiqueDatasetAll, FOLIOSymDataset, MusiqueDataset, ContextHubSymDataset

import concurrent
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from tqdm import tqdm


def get_eval_response(example):
    for ds in [
        AGIEvalDataset, ARCDataset, CSQADataset, GPQADataset, MuSRDataset,
        PIQADataset, SocialIQADataset, StrategyQADataset, WinograndeDataset, MMLUDataset, ContextHubDataset,
        MATHDataset,ProntoQADataset,GSM8KDataset,FOLIODataset,
        GSM8KSymDataset,ContextHubSymDataset,MusiqueDataset,FOLIOSymDataset,BigBenchHardDataset, BigGenBenchDataset,FOLIOSymDataset,MusiqueDatasetAll
    ]:
        if example['dataset_type'] == ds.dataset_type:
            return ds.evaluate_response
    raise ValueError('No dataset type found for example.')


def get_custom_eval_response(example):
    for ds in [
        AGIEvalDataset, ARCDataset, CSQADataset, GPQADataset, MuSRDataset,
        PIQADataset, SocialIQADataset, StrategyQADataset, WinograndeDataset, MMLUDataset, ContextHubDataset,
        ProntoQADataset,GSM8KDataset,FOLIODataset,MusiqueDataset,BigBenchHardDataset,
        BigGenBenchDataset,FOLIOSymDataset,MusiqueDatasetAll,
    ]:
        if example['dataset_type'] == ds.dataset_type:
            return ds.custom_evaluate_response
    raise ValueError('No dataset type found for example.')

def rollout(model, prompt, example, num_rollouts=10, batch_size=10, completion_length=500, temperature=0.8, top_p=0.9, disable_pbar: bool = True, evaluation_model=None, prepend_to_out=None, **inference_kwargs):
    metrics = []
    batch_size = min(batch_size, num_rollouts)
    for i in tqdm(range(0, num_rollouts, batch_size), desc='Rolling', disable=disable_pbar, total=num_rollouts // batch_size):
        raw = model.inference(prompt, num_samples=batch_size, max_tokens=completion_length, temperature=temperature, top_p=top_p, **inference_kwargs)
        outputs = model.parse_out(raw)
        if isinstance(raw, list) and hasattr(raw[0], 'model_extra'):
            try:
                additional_infos = [x.model_extra['additional_info'][0] for x in raw]
            except Exception:
                print('what')
        else:
            additional_infos = [x['additional_info'] for x in raw] if isinstance(raw, list) else getattr(raw, 'additional_info', [])


        # print(outputs)
        if outputs is None:
            metrics.extend([
                {'errored': True, 'correct': False, 'model_response': None, 'answer_line': None, 'model_answer': None, 'answerKey': example.get('answerKey'), 'additional_info': additional_info}
            for additional_info, x in zip(additional_infos, raw) ] )
            continue

        if prepend_to_out:
            outputs = [prepend_to_out + x for x in outputs]

        eval_outs = get_eval_response(example)(outputs, example, model=evaluation_model)

        metrics.extend([{
            'errored': False, 'correct': x.get('correct'), 'model_response': x.get('model_response'), 'answer_line': x.get('answer_line'), 'model_answer': x.get('model_answer'), 'answerKey': example.get('answerKey'),
            'additional_info': additional_info, 'prompt_used': prompt, 'dataset_type': x.get('dataset'), 'question': x.get('question'), 'answer': x.get('answer')
        } for x, additional_info in zip(eval_outs, additional_infos)])

    accuracy = sum([1 for metric in metrics if metric['correct']]) / max(1, len(metrics))
    
    unparsables = sum([1 for metric in metrics if metric['answer_line'] is None]) / max(1,len(metrics))

    return accuracy, unparsables, metrics


def rollout_logprobs(model, prompt, example, num_rollouts=1, batch_size=10, completion_length=500, temperature=0.8, top_p=0.9,
            disable_pbar: bool = True, evaluation_model=None, prepend_to_out=None, **inference_kwargs):

    choices = example['choices']['label']

    results = []
    # with ThreadPoolExecutor(max_workers=1) as executor:
    #     futures = []
    #
    #     for choice in choices:
    #         new_prompt = deepcopy(prompt)
    #         # if new_prompt[-1]['role'] == 'assistant':
    #         #     new_prompt[-1]['content'] += f' {choice}'
    #         # else:
    #         #     new_prompt.append({'role': 'assistant', 'content': choice})
    #
    #         futures.append(executor.submit(model.inference, new_prompt, num_samples=1, max_tokens=completion_length, temperature=temperature, top_p=top_p, vllm_sampling_kwargs={"echo": False, "logprobs": 1000}, **inference_kwargs))
    #
    #     for future in concurrent.futures.as_completed(futures):
    #         raw = future.result()
    #         results.append(raw)

    raw = model.inference(prompt, num_samples=1, max_tokens=1, temperature=temperature, top_p=top_p, vllm_sampling_kwargs={"echo": False, "logprobs": 1000}, **inference_kwargs)
    top_logprobs = list(reversed(raw[0]['choices'][0]['logprobs']['top_logprobs']))[0]
    metrics = {
        'choice_probs': {c: top_logprobs.get(c, -9999) for c in choices}
    }
    # for i, raw in enumerate(results):
    #     top_logprobs = list(reversed(raw[0]['choices'][0]['logprobs']['top_logprobs']))[0]
    #     metrics['choice_probs'][choices[i]] = top_logprobs
    #     for c in choices:

        #
        # for tok in top_logprobs:
        #     token = list(tok.keys())[0]
        #     val = list(tok.values())[0]
        #     if token == choices[i]:
        #         metrics['choice_probs'][choices[i]] = val
        #         break

    # get argmin of choice_probs
    metrics['model_answer'] = max(metrics['choice_probs'], key=metrics['choice_probs'].get)

    metrics['correct'] = metrics['model_answer'] == example['answerKey']
    model_response = "[used logprobs of prompt]"
    answer_line = "[invalid]"

    metrics['model_response'] = model_response
    metrics['answer_line'] = answer_line
    metrics['prompt_used'] = prompt
    metrics['answerKey'] = example['answerKey']
    metrics['errored'] = False

    accuracy = 1 if metrics['correct'] else 0
    unparsables = 0
    metrics = [metrics]

    return accuracy, unparsables, metrics


def rollout_with_answer_logits(model, prompt, example, num_rollouts=10, batch_size=10,
                               completion_length=500,
                               temperature=0.8, top_p=0.9, **inference_kwargs):
    metrics = []
    for i in range(0, num_rollouts, batch_size):
        cost = model.total_cost
        # OPENAI WAY
        # raw = model.inference(prompt, num_samples=batch_size, max_tokens=completion_length, temperature=temperature,
        #                       top_p=top_p, logprobs=True, top_logprobs=10, **inference_kwargs)

        # VLLM WAY
        raw = model.inference(prompt, num_samples=batch_size, max_tokens=completion_length, temperature=temperature,
                              top_p=top_p, logprobs=10, **inference_kwargs)
        log_probs = []
        ans_choice_tokens = example['answer_choice_tokens']

        for x in raw.choices:
            found = False
            for y in x.logprobs.content[::-1]:
                if y.token in ans_choice_tokens:
                    found = True
                    log_probs.append(y.top_logprobs)
                    break
            if not found:
                log_probs.append(None)


        for lidx in range(len(log_probs)):

            if log_probs[lidx] is None or (isinstance(log_probs[lidx], list) and len(log_probs[lidx]) == 0):
                continue
            toks = [[x.token, np.exp(x.logprob)] for x in log_probs[lidx] if x.token in ans_choice_tokens]


            scale = sum([x[1] for x in toks])
            norm = 100 / max(1e-10, scale)
            toks = [[x[0], x[1] * norm] for x in toks]
            log_probs[lidx] = toks

        responses = model.parse_out(raw)

        m = get_eval_response(example)(responses, example)

        for i, metric in enumerate(m):
            metric['logits'] = log_probs[i]
        metrics.extend(m)

    accuracy = sum([1 for metric in metrics if metric['correct']]) / len(metrics)
    unparsables = sum([1 for metric in metrics if metric['answer_line'] is None]) / len(metrics)
    return accuracy, unparsables, metrics


def build_prefix_prompt(prompt, prefix, as_trailing_assistant_message: bool = True):
    prompt = deepcopy(prompt)
    if isinstance(prefix, list):
        prefix = prefix[0]

    if as_trailing_assistant_message:
        if isinstance(prompt, list):
            prompt = [
                *prompt,
                {'role': 'assistant', 'content': prefix}
            ]
        else:
            prompt = [
                {'role': 'user', 'content': prompt},
                {'role': 'assistant', 'content': prefix}
            ]
    else:
        if isinstance(prompt, list):
            prompt[-1]['content'] += f'\n\n{prefix}'
        else:
            prompt = f'{prompt}\n\n{prefix}'
    return prompt


def answer_percentage_from_metrics(metrics, use_logits: bool = False):
    if use_logits:
        answers = [x['logits'] for x in metrics]

        distribution = {}
        for a in answers:
            for k, v in a:
                ct = distribution.get(v, 0)
                ct += v
                distribution[v] = ct

        distribution = {k: v / len(answers) for k, v in distribution.items()}
        return distribution

    answers = [x['model_answer'] for x in metrics]
    counts = {}
    for a in answers:
        ct = counts.get(a, 0)
        ct += 1
        counts[a] = ct

    counts = {k: v / len(answers) for k, v in counts.items()}
    return counts
