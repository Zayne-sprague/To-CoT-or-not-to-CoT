from abc import abstractmethod, ABC
from typing import List, Dict, Any, Generator


class Model(ABC):

    @abstractmethod
    def inference(self, prompt: str, *args, **kwargs) -> Any:
        """
        Most simple inference to a model that takes a prompt and gets an output object back.

        :param prompt: The prompt to give to the language model
        :param args: If the specific model needs more arguments
        :param kwargs: If the specific model needs more keyword arguments
        :return: Generated response from the language model.
        """
        raise NotImplementedError("All models need an inference call implemented.")

    @classmethod
    def load_model(cls, name: str, **kwargs):
        from src.model import OpenAIModel, AnthropicModel, VLLMEndpointModel,\
            GeminiModel

        tag = name.split('/')[0].strip()
        model_name = name.split('/', 1)[-1]

        if tag == 'anthropic':
            prompt_price = 0.0
            completion_price = 0.0

            if 'haiku' == model_name.lower():
                model_name = 'claude-3-haiku-20240307'
            elif 'sonnet' == model_name.lower():
                model_name = 'claude-3-sonnet-20240229'
            elif 'opus' == model_name.lower():
                model_name = 'claude-3-opus-20240229'

            if 'claude-3-5-sonnet-20240620' == model_name:
                prompt_price = 3 / 1_000_000
                completion_price = 15 / 1_000_000
            elif 'haiku' in model_name.lower():
                prompt_price = 0.25 / 1_000_000
                completion_price = 1.25 / 1_000_000
            elif 'sonnet' in model_name.lower():
                prompt_price = 3. / 1_000_000
                completion_price = 15 / 1_000_000
            elif 'opus' in model_name.lower():
                prompt_price = 15. / 1_000_000
                completion_price = 75 / 1_000_000



            kwargs['prompt_cost'] = prompt_price
            kwargs['completion_cost'] = completion_price
            return AnthropicModel(model_name, **kwargs)
        if tag == 'gemini':
            if 'prompt_cost' not in kwargs and 'completion_cost' not in kwargs:
                prompt_price = 0.0
                completion_price = 0.0

                if model_name == 'google/gemini-1.5-flash-001':
                    prompt_price = 0.075 / 1_000_000
                    completion_price = 0.30 / 1_000_000
                elif model_name == 'google/gemini-1.5-pro-001':
                    prompt_price = 3.5 / 1_000_000
                    completion_price = 10.5 / 1_000_000
                kwargs['prompt_cost'] = prompt_price
                kwargs['completion_cost'] = completion_price
            return GeminiModel(model_name, **kwargs)

        if tag == 'openai':

            if 'prompt_cost' not in kwargs and 'completion_cost' not in kwargs:
                prompt_price = 0.0
                completion_price = 0.0
                if model_name == 'gpt-3.5-turbo':
                    prompt_price = 0.001 / 1000
                    completion_price = 0.02 / 1000
                elif model_name == 'gpt-3.5-turbo-0125':
                    prompt_price = 0.00050 / 1000
                    completion_price = 0.00150 / 1000
                elif model_name == 'gpt-3.5-turbo-16k':
                    prompt_price = 0.003 / 1000
                    completion_price = 0.004 / 1000
                elif model_name == 'gpt-4':
                    prompt_price = 0.03 / 1000
                    completion_price = 0.06 / 1000
                elif model_name == 'gpt-4-1106-preview' or model_name == 'gpt-4-0125-preview':
                    prompt_price = 0.01 / 1000
                    completion_price = 0.03 / 1000
                elif model_name == 'gpt-4o-2024-05-13':
                    prompt_price = 0.0050 / 1000
                    completion_price = 0.0150 / 1000
                elif model_name == 'gpt-4o-mini-2024-07-18':
                    prompt_price = 0.000150 / 1000
                    completion_price = 0.000600 / 1000
                elif model_name == 'gpt-4o-2024-08-06':
                    prompt_price = 0.00250  / 1000
                    completion_price = 0.01000 / 1000
                elif model_name == 'gpt-3.5-turbo-instruct':
                    prompt_price = 0.0015 / 1000
                    completion_price = 0.0020 / 1000
                    kwargs['api_endpoint'] = 'completion'
                    kwargs['echo'] = False
                elif model_name == 'text-davinci-003':
                    prompt_price = 0.02 / 1000
                    completion_price = 0.02 / 1000
                    kwargs['api_endpoint'] = 'completion'
                    kwargs['echo'] = False
                elif 'o1-preview' in model_name:
                    prompt_price = 15 / 1_000_000
                    completion_price = 60 / 1_000_000
                elif 'o1-mini' in model_name:
                    prompt_price = 3 / 1_000_000
                    completion_price = 12 / 1_000_000
                kwargs['prompt_cost'] = prompt_price
                kwargs['completion_cost'] = completion_price


            return OpenAIModel(model_name, **kwargs)
        elif tag == 'vllm_endpoint':
            return VLLMEndpointModel(model_name, **kwargs)


    def parse_out(self, out):
        from src.model import OpenAIModel, GeminiModel
        if isinstance(self, OpenAIModel) or isinstance(self, GeminiModel):
            if self.api_endpoint == 'completion':
                return [x['text'] for x in out.choices]
            else:
                return [x.message.content if x.message is not None else '' for x in out.choices]
        else:
            return out