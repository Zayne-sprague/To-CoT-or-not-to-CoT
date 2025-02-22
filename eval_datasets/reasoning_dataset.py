from torch.utils.data import Dataset
import abc


class ReasoningDatasetTypes:
    musr = 'musr'
    csqa = 'csqa'
    socialiqa = 'socialiqa'
    winogrande = 'winogrande'
    piqa = 'piqa'
    copa_balanced = 'copa_balanced'
    mmlu = 'mmlu'
    mmlu_pro = 'mmlu_pro'
    mmlu_redux = 'mmlu_redux'
    musique = 'musique'
    musique_all = 'musique_all'
    gsm8k = 'gsm8k'
    strategyqa = 'stratqa'
    boolq = 'boolq'
    arc = 'arc'
    agieval = 'agieval'
    gpqa = 'gpqa'
    contexthub = 'contexthub'
    math = 'math'
    prontoqa = 'prontoqa'
    gsm8k = 'gsm8k'
    folio = 'folio'
    gsm8ksym = 'gsm8ksym'
    contexthubsym = 'contexthubsym'
    foliosym = 'foliosym'
    bbh = 'bigbenchhard'
    biggen_bench = 'biggen_bench'


class ReasoningDataset(Dataset, abc.ABC):

    dataset_types = ReasoningDatasetTypes
    average_token_len: int = 500

    def __init__(self, path_or_url, generating_paraphrase=False,*args, prompt_fn=None, **kwargs):
        self.requires_eval_model = False
        self.prompt_fn = self.build_messages if prompt_fn is None else prompt_fn
        self.examples = self.load_dataset(path_or_url)
        self.generating_paraphrase = generating_paraphrase

    @classmethod
    @property
    def dataset_type(cls):
        raise NotImplementedError('This method must be implemented by the subclass.')
    ## Return false by default
    @classmethod
    @property
    def is_math(cls):
        return False
    @abc.abstractmethod
    def load_dataset(self, path_or_url):
        raise NotImplementedError('This method must be implemented by the subclass.')




    @classmethod
    def parse_ans(cls, word):
        strip_outs = [':', '.', ',', '<', '>', '!', '?', '(', ')', '[', ']', '{', '}', '\'', '\"', '’', '‘', '“',
                      '”']
        for x in strip_outs:
            word = word.strip().replace(x, '')
        return word

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            return [self.process_example(self.examples[i]) for i in range(*idx.indices(len(self)))]

        return self.process_example(self.examples[idx])

    def build_messages(self, example, include_example: bool = True, icl_as_messages: bool = True, icl_prompt: bool = False):
        user_ctx = example['prompt_parts']['user_context'] if not icl_prompt else example['prompt_parts']['icl_prompt']
        icl_user_ctx = example['prompt_parts'].get('icl_user_contexts')
        icl_assistant_ctx = example['prompt_parts'].get('icl_assistant_contexts')

        messages = []

        if include_example and icl_user_ctx and icl_assistant_ctx:
            if icl_as_messages:
                for uctx, actx in zip(icl_user_ctx, icl_assistant_ctx):
                    messages.append({'role': 'user', 'content': uctx})
                    messages.append({'role': 'assistant', 'content': actx})
            else:
                new_ctx = ''
                for uctx, actx in zip(icl_user_ctx, icl_assistant_ctx):
                    new_ctx += f'{uctx}\n{actx}\n'
                new_ctx += user_ctx
                user_ctx = new_ctx

        messages.append({'role': 'user', 'content': user_ctx})

        return messages

    def process_example(self, example,*args, **kwargs):
        def create_msgs(prompt, sys_prompt=None):
            if prompt is None:
                return None

            if sys_prompt:
                return [{'role': 'system', 'content': sys_prompt}, {'role': 'user', 'content': prompt}]
            return [{'role': 'user', 'content': prompt}]

        prompt_parts = example['prompt_parts']

        if 'zs_cot_prompt' in prompt_parts:
            if not self.generating_paraphrase:
                cot_sys_prompt = prompt_parts.get('cot_system_prompt',  self.default_sys_mc_cot_prompt)
                cotless_sys_prompt = prompt_parts.get('cotless_system_prompt', self.default_sys_mc_cotless_prompt)
            else:
                cot_sys_prompt = prompt_parts.get('cot_system_prompt',  self.basic_sys_prompt)
                cotless_sys_prompt = prompt_parts.get('cotless_system_prompt', self.basic_sys_prompt)


            zs_cot_prompt = create_msgs(prompt_parts.get('zs_cot_prompt'), cot_sys_prompt)
            zs_cotless_prompt = create_msgs(prompt_parts.get('zs_cotless_prompt'), cotless_sys_prompt)

            example['messages'] = zs_cot_prompt
            example['zs_cot_messages'] = zs_cot_prompt
            example['zs_cotless_messages'] = zs_cotless_prompt

            if 'fs_cot_prompt' in prompt_parts:
                fs_cot_prompt = create_msgs(prompt_parts.get('fs_cot_prompt'), cot_sys_prompt)
                fs_cotless_prompt = create_msgs(prompt_parts.get('fs_cotless_prompt'), cotless_sys_prompt)
                example['fs_cot_messages'] = fs_cot_prompt
                example['fs_cotless_messages'] = fs_cotless_prompt

        else:
            if not self.generating_paraphrase:
                example['messages'] = create_msgs(prompt_parts.get('user_context'), self.default_sys_mc_cot_prompt)

                if example["prompt_parts"].get('cotless_user_context'):
                    example['cotless_messages'] = create_msgs(prompt_parts.get('cotless_user_context'), self.default_sys_mc_cotless_prompt)
            else:
                example['messages'] = create_msgs(prompt_parts.get('user_context'), self.basic_sys_prompt)

                if example["prompt_parts"].get('cotless_user_context'):
                    example['cotless_messages'] = create_msgs(prompt_parts.get('cotless_user_context'), self.basic_sys_prompt)


        return example

    def basic_prompt(self, question, choices, direct=False):
        assert isinstance(choices, str), 'Choices must be a string.'

        instructions = """Think step by step before giving your final answer to the question.  When you are ready to answer write the answer in the format: "Answer: <your answer>". You must always give an answer at the end. You may only pick one answer choice, if you think multiple are correct only pick the one you think is best."""
        if direct:
            instructions = """Only write the answer in the format: "Answer: <your answer>". You must always give an answer. You may only pick one answer choice, if you think multiple are correct only pick the one you think is best."""
        return f'''
{question}

{choices}

{instructions}
            '''.strip()

    def basic_cotless_prompt(self, question, choices):
            return f'''
{question}

{choices}

Only write the answer. Write the answer in the following format: "Answer: <your answer>". You must always give an answer. You may only pick one answer choice, if you think multiple are correct only pick the one you think is best.
                '''.strip()
    def paraphrase_prompt(self,question,level='complex'):
        if level == 'complex':
            return f"""
            Paraphrase the following deductive reasonig problem such that: 1. Each sentence is syntactically and lexically more complicated. 2. While the syntax and words are changed after your paraphrase, the semantic meaning of each sentence must remain clear and unchanged without any ambiguity. 3. The order of the sentences remains unchanged. \n {question}
            """
        elif level == 'simple':
            return f"""
            Paraphrase the following deductive reasonig problem such that: 1. Each sentences should be rephrased with different words and even different syntax if possible, but they should not make the original setences too complex. 2. While the syntax and words are changed after your paraphrase, the semantic meaning of each sentence must remain clear and unchanged without any ambiguity. 3. The order of the sentences remains unchanged. \n {question}
            """
    def format_choices(self, choices):
        return '\n'.join([f'( {choices["label"][i]} ) {choices["text"][i]}' for i in range(len(choices["label"]))])

    @abc.abstractmethod
    def evaluate_response(self, model_responses, example, *args, **kwargs):
        raise NotImplementedError('This method must be implemented by the subclass.')

    @abc.abstractmethod
    def custom_evaluate_response(self, model_responses, example, *args, **kwargs):
        return None

    def custom_collate_fn(self, batch):
        return batch

    @property
    def default_sys_mc_cot_prompt(self):
        return 'You are a helpful AI assistant that will answer reasoning questions. You may reason over the question but you will always say at the end "Answer: <Your Answer Letter Choice>". You must only pick one answer and you must end your response with "Answer: <Your Answer Letter Choice>" everytime!'

    @property
    def default_sys_mc_cotless_prompt(self):
        return 'You are a helpful AI assistant that will answer reasoning questions. You will always say at the end "Answer: <Your Answer Letter Choice>". You must only pick one answer and you must end your response with "Answer: <Your Answer Letter Choice>" everytime!'

    @property
    def basic_sys_prompt(self):
        return 'You are a helpful AI assistant.'

    @staticmethod
    def safe_find(text, find):
        if find in text:
            return text.index(find)
        if find.lower() in text.lower():
            return text.lower().index(find.lower())
        # raise Exception(f'Could not find {find} in {text}')
        return -1