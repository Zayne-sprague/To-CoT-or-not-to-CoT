import os
import time
from datetime import timedelta
from typing import Dict, Union, Any
from anthropic import Anthropic


from src.model.model import Model
from src import cache


class AnthropicModel(Model):
    """
    Wrapper for calling OpenAI with some safety and retry loops as well as a somewhat advanced caching mechanism.
    """

    engine: str
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
            model: str = 'claude-3-haiku-20240307',
            api_max_attempts: int = 120,

            temperature: float = 1.0,
            top_p: float = 1.0,
            max_tokens: int = 2049,
            stop_token: str = None,
            num_samples: int = 1,

            prompt_cost: float = None,
            completion_cost: float = None,
            system_prompt: str = "You are a helpful AI assistant.",
            **kwargs,
    ):
        """

        """

        self.model = model
        self.client = Anthropic(api_key=os.getenv('ANTHROPIC_API_KEY'))

        self.api_max_attempts = api_max_attempts

        self.max_tokens = max_tokens
        self.stop_token = stop_token

        self.num_samples = num_samples

        self.system_prompt = system_prompt

        self.temperature = temperature
        self.top_p = top_p

        self.waittime = 60

        self.prompt_cost = prompt_cost
        self.completion_cost = completion_cost
        self.total_cost = 0.0


    def __update_cost__(self, raw):
        if self.prompt_cost and self.completion_cost:
            cost = raw.usage.output_tokens * self.completion_cost + raw.usage.input_tokens * self.prompt_cost
            self.total_cost += cost

    @cache.cached(data_ex=timedelta(days=30), no_data_ex=timedelta(hours=1), prepended_key_attr='model,num_samples,temperature=float(0),top_p=float(1.0),stop_token,max_tokens')
    def inference(self, prompt: str, *args, **kwargs) -> Any:
        outs = []
        for _ in range(kwargs.get('num_samples', self.num_samples)):
            out = self.__safe_anthropic_call__(prompt, **kwargs)
            self.__update_cost__(out)
            outs.append(out)
        return outs

    def __safe_anthropic_call__(
            self,
            prompt: str,
            system_prompt: str = None,
            temperature: float = None,
            top_p: float = None,
            max_tokens: int = None,
            stop_token: str = None,
            **kwargs,
    ) -> Dict[str, Union[str, bool]]:
        if max_tokens is None:
            max_tokens = self.max_tokens
        if temperature is None:
            temperature = self.temperature
        if top_p is None:
            top_p = self.top_p
        if stop_token is None:
            stop_token = self.stop_token
        if system_prompt is None:
            system_prompt = self.system_prompt
        if kwargs.get('num_samples', None):
            # Claude doesn't support this yet :P
            del kwargs['num_samples']

        for i in range(self.api_max_attempts):
            try:
                # TODO - look at different roles?
                if isinstance(prompt, str):
                    messages = [
                        {"role": "user", "content": prompt}
                    ]
                else:
                    messages = prompt
                    if messages[0]['role'] == 'system':
                        system_prompt = messages[0]['content']
                        messages = messages[1:]
                    if messages[-1]['role'] != 'user':
                        messages[-1]['content'] = messages[-1]['content'].rstrip()

                raw = self.client.messages.create(
                    model=self.model,
                    messages=messages,
                    temperature=temperature,
                    top_p=top_p,
                    system=system_prompt,
                    max_tokens=max_tokens,
                    stop_sequences=stop_token,
                    **kwargs.get('api_kwargs', {})
                )
                setattr(raw, "additional_info", [{'raw_payload': {
                    'engine': self.model,
                    'messages': messages,
                    'temperature': temperature,
                    'top_p': top_p,
                    'max_tokens': max_tokens,
                    'system': system_prompt,
                    'n': 1,
                    'stop': stop_token,
                    **kwargs
                }} for _ in raw.content])
                return raw
            except Exception as e:
                last_exc = e
                time.sleep(self.waittime)
                print(f"ERROR: ANTHROPIC Error: {e}")
        # make a fake response
        return None

    def parse_out(self, out):
        return [x.text for y in out for x in y.content]


if __name__ == "__main__":
    from src import cache
    cache.enable()

    from key_handler import KeyHandler
    KeyHandler.set_env_key()


    model = AnthropicModel(
        model='claude-3-haiku-20240307',
        api_max_attempts=120,
        temperature=1.0,
        top_p=1.0,
        max_tokens=2049,
        stop_token=None,
        num_samples=1,
        prompt_cost=0.25/1_000_000,
        completion_cost=1.25/1_000_000,
    )

    prompt = [{'role': 'user', 'content': "What is the meaning of life?"}, {'role': 'assistant', 'content': "The meaning of life is to be the "}]


    outs = model.parse_out(model.inference(prompt, num_samples=3))
    for o in outs:
        print(o)
        print('---')

    print(model.total_cost)