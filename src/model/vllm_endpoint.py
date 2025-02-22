import math
import os
from concurrent.futures import ThreadPoolExecutor
import concurrent
import itertools
import time
import openai

from datetime import timedelta
import random
import threading
import jinja2
import requests

from typing import List, Dict, Union, Any, Generator
from tqdm import tqdm
from transformers import GPT2TokenizerFast
from transformers import AutoTokenizer

from src.model.model import Model
from src import cache

class VLLMEndpointModel(Model):
    """
    Wrapper for calling OpenAI with some safety and retry loops as well as a somewhat advanced caching mechanism.
    """

    endpoint: str
    api_max_attempts: int
    api_endpoint: str

    max_tokens: int
    stop_token: str

    log_probs: int
    num_samples: int

    echo: bool

    temperature: float

    def __init__(
            self,
            endpoint: str,
            api_max_attempts: int = 50,
            api_endpoint: str = 'chat',

            temperature: float = 1.0,
            top_p: float = 0.95,
            top_k: int = -1,
            max_tokens: int = 3000,
            stop_token: str = None,
            log_probs: int = 1,
            num_samples: int = 1,
            echo: bool = True,

            prompt_cost: float = None,
            completion_cost: float = None,
            logprobs: int = None,
            **kwargs,
    ):
        """

        :param endpoint: The endpoint you are calling
        :param api_max_attempts: Retry the api call N times.
        :param api_endpoint: Usually differs between completion or chat endpoint
        :param temperature: https://platform.openai.com/docs/api-reference/chat/create#chat/create-temperature
        :param top_p: https://platform.openai.com/docs/api-reference/chat/create#chat/create-top_p
        :param max_tokens: https://platform.openai.com/docs/api-reference/chat/create#chat/create-max_tokens
        :param stop_token: https://platform.openai.com/docs/api-reference/chat/create#chat/create-stop
        :param log_probs: (only for completion) https://platform.openai.com/docs/api-reference/completions/create#completions/create-logprobs
        :param num_samples: https://platform.openai.com/docs/api-reference/chat/create#chat/create-n
        :param echo: (only for completion) https://platform.openai.com/docs/api-reference/completions/create#completions/create-echo
        :param prompt_cost: Pass in the current cost of the api you are calling to track costs (optional)
        :param completion_cost: Pass in the current cost of the api you are calling to track costs (optional)
        """

        self.endpoint = endpoint.split("<model>")[0]
        self.model = endpoint.split("<model>")[1]
        self.cache_key = f"vllm_{self.model}"

        if '/datastor1/shared_resources/nlp-models/llama3/Meta-Llama-3-70B-Instruct' in self.model:
            self.tokenizer = AutoTokenizer.from_pretrained('meta-llama/Meta-Llama-3-70B-Instruct', trust_remote_code=True)
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model, trust_remote_code=True)

        self.api_max_attempts = api_max_attempts
        self.api_endpoint = api_endpoint.lower()

        self.max_tokens = max_tokens
        self.stop_token = stop_token

        self.log_probs = log_probs
        self.num_samples = num_samples

        self.echo = echo

        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k

        self.gpt_waittime = 3

        self.prompt_cost = prompt_cost
        self.completion_cost = completion_cost
        self.total_cost = 0.0

        self.logprobs = logprobs

        # if not openai.api_key:
        #     openai.api_key = os.getenv("OPENAI_API_KEY")


        if 'Llama-2' in self.model or 'llama-2' in self.model.lower():
            self.jinja_template_str = "{% if messages[0]['role'] == 'system' %}{% set loop_messages = messages[1:] %}{% set system_message = messages[0]['content'] %}{% elif USE_DEFAULT_PROMPT == true and not '<<SYS>>' in messages[0]['content'] %}{% set loop_messages = messages %}{% set system_message = 'DEFAULT_SYSTEM_MESSAGE' %}{% else %}{% set loop_messages = messages %}{% set system_message = false %}{% endif %}{% for message in loop_messages %}{% if (message['role'] == 'user') != (loop.index0 % 2 == 0) %}{{ raise_exception('Conversation roles must alternate user/assistant/user/assistant/...') }}{% endif %}{% if loop.index0 == 0 and system_message != false %}{% set content = '<<SYS>>\\n' + system_message + '\\n<</SYS>>\\n\\n' + message['content'] %}{% else %}{% set content = message['content'] %}{% endif %}{% if message['role'] == 'user' %}{{ bos_token +'[INST] ' + content | trim + ' [/INST]' }}{% elif message['role'] == 'system' %}{{ '<<SYS>>\\n' + content | trim + '\\n<</SYS>>\\n\\n' }}{% elif message['role'] == 'assistant' %}{{ ' ' + content | trim + ' ' + eos_token }}{% endif %}{% endfor %}"
        else:
            self.jinja_template_str = """{% set loop_messages = messages %}{% for message in loop_messages %}{% set content = '<|start_header_id|>' + message['role'] + '<|end_header_id|>\n\n'+ message['content'] | trim + '<|eot_id|>' %}{% if loop.index0 == 0 %}{% set content = bos_token + content %}{% endif %}{{ content }}{% endfor %}{% if add_generation_prompt %}{{ '<|start_header_id|>assistant<|end_header_id|>\n\n' }}{% endif %}""".strip()

        self.environment = jinja2.Environment()
        self.template = self.environment.from_string(self.jinja_template_str)

        # self.client = openai.OpenAI(
            # This is the default and can be omitted
            # base_url=endpoint,
            # api_key=os.getenv('HF_API_KEY')
        # )
        self.headers = {
            "Accept": "application/json",
            # "Authorization": f"Bearer {os.getenv('HF_API_KEY')}",
            "Content-Type": "application/json"
        }

    def __update_cost__(self, raw):
        if self.prompt_cost and self.completion_cost:
            cost = raw.usage.completion_tokens * self.completion_cost + raw.usage.prompt_tokens * self.prompt_cost
            self.total_cost += cost

    def query(self, payload):
        try:
            response = requests.post(self.endpoint, headers=self.headers, json=payload, timeout=5 * 60)
            if response.status_code != 200:
                print(response.text)
                msg = response.json()
                if msg['object'] == 'error' and 'reduce the length of the messages' in msg['message']:
                    return response.json(), True
            return response.json(), False

        except Exception:
            return {}, True

    @cache.cached(data_ex=timedelta(days=30), no_data_ex=timedelta(hours=1), prepended_key_attr='cache_key,num_samples,log_probs,echo,temperature=float(0),top_p=float(1.0),stop_token,max_tokens')
    def inference(self, prompt: str, *args, **kwargs) -> Any:
        outs = []
        actual_n_samples = 50
        with ThreadPoolExecutor(max_workers=100) as executor:
            futures = [executor.submit(self.__safe_openai_chat_call__, prompt, *args, num_samples=actual_n_samples, **{k: v for k, v in kwargs.items() if k!='num_samples'}) for _ in range(math.floor(int(kwargs.get('num_samples', self.num_samples) / actual_n_samples)))]
            if kwargs.get('num_samples', self.num_samples) % actual_n_samples:
                futures.append(executor.submit(self.__safe_openai_chat_call__, prompt, *args, num_samples=kwargs.get('num_samples', self.num_samples) % actual_n_samples, **{k: v for k, v in kwargs.items() if k!='num_samples'}))

            outs = []
            for future in concurrent.futures.as_completed(futures):
                outs.append(future.result())

        # for _ in range(kwargs.get('num_samples', self.num_samples)):
        #     t = threading.Thread(target=self.__safe_openai_chat_call__, args=(prompt, *args), kwargs=kwargs)
            # out = self.__safe_openai_chat_call__(
            #     prompt,
            #     *args,
            #     **kwargs
            # )
            # outs.append(t)
        #
        # outs = [x.start() for x in outs]
        # outs = [x.join() for x in outs]

        return outs

    def __safe_openai_chat_call__(
            self,
            prompt: str,
            system_prompt: str = None,
            temperature: float = None,
            top_p: float = None,
            top_k: int = None,
            max_tokens: int = None,
            stop_token: str = None,
            num_samples: int = None,
            only_new_tokens: bool = True,
            do_sample: bool=True,
            logprobs: int = None,
            prompt_logprobs: int = None,
            use_bos_token: bool = True,
            vllm_sampling_kwargs: Dict[str, Any] = None,
            comp_prompt: str = None,
            use_tokenizer: bool = True,
            **kwargs,
    ) -> Dict[str, Union[str, bool]]:
        if vllm_sampling_kwargs is None:
            vllm_sampling_kwargs = {}
        if max_tokens is None:
            max_tokens = self.max_tokens
        if temperature is None:
            temperature = self.temperature
        if top_p is None:
            top_p = self.top_p
        if top_k is None:
            top_k = self.top_k
        if stop_token is None:
            if self.stop_token is not None:
                stop_token = self.stop_token
            else:
                if 'Llama-3' in self.model:
                    stop_token = ["<|start_header_id|>", "<|end_header_id|>", "<|eot_id|>", "<|reserved_special_token","</s>"]
                else:
                    stop_token = ["</s>"]
        if num_samples is None:
            num_samples = self.num_samples
        if logprobs is None:
            logprobs = self.logprobs


        last_exc = None
        payload = None
        for i in range(self.api_max_attempts):
            try:

                def rreplace(s, old, new):
                    return (s[::-1].replace(old[::-1], new[::-1], 1))[::-1]

                # TODO - look at different roles?
                if not isinstance(prompt, str):
                    if use_tokenizer and 'qwen' not in self.model.lower():
                        new_prompt = self.tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True)
                    elif 'reflection-llama-3.1' in self.model.lower() and prompt[0]['role'] == 'system':
                        prompt[0]['content'] = 'You are a world-class AI system, capable of complex reasoning and reflection. Reason through the query inside <thinking> tags, and then provide your final response inside <output> tags. If you detect that you made a mistake in your reasoning at any point, correct yourself inside <reflection> tags. ' + prompt[0]['content']
                    elif 'qwen' in self.model.lower():
                        new_prompt = self.tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True)
                    elif 'olmo' in self.model.lower():
                        new_prompt = self.build_olmo_prompt(prompt)
                    elif 'phi-3' in self.model.lower():
                        new_prompt = self.build_phi_prompt(prompt)
                    elif 'gemma' in self.model.lower():
                        new_prompt = self.build_gemma_prompt(prompt)
                    elif 'llama2' in self.endpoint or 'llama-2' in self.model.lower():
                        new_prompt = self.template.render(messages=prompt, bos_token='<s>', eos_token='</s>')
                    elif 'mistral' in self.model.lower() or 'mixtral' in self.model.lower():
                        new_prompt = self.tokenizer.apply_chat_template([x for x in prompt if x['role'] != 'system'], tokenize=False)
                    else:
                        # TODO - llama 3 evals do not use a BOS Token?
                        new_prompt = self.template.render(messages=prompt, bos_token='<|begin_of_text|>' if use_bos_token else '',
                                                          eos_token='<|eot_id|>')
                        # new_prompt = new_prompt.replace("<|begin_of_text|>", "")

                    if prompt[-1]['role'] == 'assistant':
                        if 'olmo' in self.model.lower():
                            prompt = new_prompt
                        elif 'llama2' in self.endpoint or 'llama-2' in self.model.lower():
                            prompt = rreplace(new_prompt, '</s>', '')
                        elif 'mistral' in self.model.lower() or 'mixtral' in self.model.lower():
                            prompt = new_prompt
                        elif 'phi-3' in self.model.lower() or 'gemma' in self.model.lower():
                            prompt = new_prompt
                        elif 'qwen' in self.model.lower():
                            prompt = rreplace(new_prompt, "<|im_end|>\n<|im_start|>assistant", '')
                        elif 'intern' in self.model.lower():
                            prompt = rreplace(new_prompt, '<|im_end|>\n<|im_start|>assistant\n', '').rstrip()
                        else:
                            prompt = rreplace(new_prompt, '<|eot_id|>', '')  # .rstrip()
                        # else:
                        #     prompt  = new_prompt
                    else:
                        # if 'llama-3' in self.model.lower():
                        #     prompt = new_prompt
                        if 'olmo' in self.model.lower():
                            prompt = new_prompt
                        elif 'phi-3' in self.model.lower() or 'gemma' in self.model.lower():
                            prompt = new_prompt
                        elif 'llama-3' in self.model.lower():
                            prompt = new_prompt + "<|start_header_id|>assistant<|end_header_id|>\n\n"
                        elif 'mistral' in self.model.lower() or 'mixtral' in self.model.lower():
                            prompt = new_prompt
                        else:
                            prompt = new_prompt

                # print(prompt)
                # print('\n\n')
                if prompt != comp_prompt and comp_prompt is not None:
                    # print('hi')
                    pass
                payload = {
                    **vllm_sampling_kwargs,
                    'model': self.model,
                    'prompt': prompt,
                    'max_tokens': max_tokens,
                    'temperature': temperature,
                    'top_p': top_p,
                    # 'echo': True,
                    # 'logprobs': 1,
                    # 'top_k': top_k,
                    'n': num_samples,
                    # 'logprobs': logprobs,
                    # 'min_tokens': 1,  # set to 20 for stratqa for some reason
                    # 'stop': stop_token
                }
                if 'olmo' in self.model.lower():
                    payload['truncate_prompt_tokens'] = 1024

                # print(payload['inputs'])
                # return [{'generated_text': f'{prompt} | TESTING 123'}]

                raw, stop = self.query(payload)
                # print(raw)
                # exit()
                assert raw is not None, "API returned None"
                if only_new_tokens:
                    for x in raw['choices']:
                        x['text'] = x['text'].replace(prompt, '').strip()
                # print('returning')

                raw['additional_info'] = {
                    "raw_payload": payload
                }

                return raw

            except Exception as e:
                # raise e
                time.sleep(1)
                print(f"ERROR: Service Error: {e}")
                import traceback
                print(traceback.format_exc())
        # make a fake response
        return {
            "text": prompt + " OPENAI Error - " + str(last_exc),
            "additional_info": {"raw_payload": payload},
            "API Error": True,
        }

    def parse_out(self, out):
        # return [x["generated_text"] for x in out if "generated_text" in x]
        try:
            return [x['text'] for y in out for x in y['choices']]
        except Exception as e:
            # raise e
            return None


    def build_olmo_prompt(self, messages):
        last_role = None
        prompt = ''
        for msg in messages:
            role = msg['role']
            content = msg['content']
            last_role = role

            if role == 'user':
                prompt += f'<|user|>\n{content}\n'
            elif role == 'assistant':
                prompt += f'<|assistant|>\n{content}\n'

        if last_role == 'assistant':
            prompt = prompt.rstrip()
        else:
            prompt = prompt + '<|assistant|>\n'

        return prompt


    def build_phi_prompt(self, messages):
        last_role = None
        prompt = '<|endoftext|>'
        sys_prompt = ''
        for midx, msg in enumerate(messages):
            role = msg['role']
            content = msg['content']
            if role == 'system':
                sys_prompt = content
                continue

            last_role = role

            if role == 'user':
                prompt += f'<|user|>\n{sys_prompt}\n\n{content}<|end|>\n'
            elif role == 'assistant':
                if midx == len(messages) - 1:
                    prompt += f'<|assistant|>\n{content}'
                else:
                    prompt += f'<|assistant|>\n{content}<|end|>\n'

        if last_role == 'user':
            prompt = prompt + '<|assistant|>\n'

        return prompt

    def build_gemma_prompt(self, messages):
        last_role = None
        prompt = '<bos>'
        sys_prompt = ''
        for midx, msg in enumerate(messages):
            role = msg['role']
            content = msg['content']
            if role == 'system':
                sys_prompt = content
                continue

            last_role = role

            if role == 'user':
                if sys_prompt:
                    prompt += f'<start_of_turn>user\n{sys_prompt}\n\n{content}<end_of_turn>\n'
                else:
                    prompt += f'<start_of_turn>user\n{content}<end_of_turn>\n'
            elif role == 'assistant':
                if midx == len(messages) - 1:
                    prompt += f'<start_of_turn>model\n{content}'
                else:
                    prompt += f'<start_of_turn>model\n{content}<end_of_turn>\n'

        if last_role == 'user':
            prompt = prompt + '<start_of_turn>model\n'

        return prompt

if __name__ == "__main__":
    from src import cache
    from src.model import Model
    from key_handler import KeyHandler


    try:
        cache.enable(bust_cache=True)
    except Exception:
        print(
            "Could not connect to your local redis server, try `redis-server` and `redis-cli` in a terminal to see what's up.  Otherwise, generations will NOT be cached.")


    KeyHandler.set_env_key()
    # model = Model.load_model('openai/gpt-3.5-turbo-16k')
    model = Model.load_model('vllm_endpoint/http://127.0.0.1:8089/v1/completions<model>meta-llama/Meta-Llama-3-8B-Instruct')

    raw = model.parse_out(model.inference([{'role': 'user', 'content':"What is 2+2?"}, {'role': 'assistant', 'content': 'The answer is... 3'}], max_tokens=200, vllm_sampling_kwargs={"logit_bias": {model.tokenizer.encode('but')[1]: -2.0}}))
    print(raw)
