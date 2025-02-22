import random

from datasets import load_dataset

from eval_datasets import ReasoningDataset


class BigGenBenchDataset(ReasoningDataset):
    def __init__(self, path_or_url='prometheus-eval/BiGGen-Bench', split='test', *args, **kwargs):
        super().__init__(path_or_url + ':' + split, *args, **kwargs)
        self.requires_eval_model = True

    @classmethod
    @property
    def dataset_type(cls):
        return cls.dataset_types.biggen_bench

    def load_dataset(self, path_or_url):
        examples = []
        dataset_url, split = path_or_url.split(':')
        dataset = [x for x in load_dataset(dataset_url)[split]]

        for ex in dataset:
            answer = ex['reference_answer']
            answerKey = 'E'
            choices = {'text': ['1','2','3','4','5'], 'label': ['A', 'B', 'C', 'D', 'E']}

            # zs_cot = f'{ex["input"]}\n\nBefore you answer the question, you will think step by step how you will answer. When you are ready, you will say "[Final Answer] <Direct response to the question>", where <Direct response to the question> is a full response you would give to the question that you compose from your reasoning.  You may think of this as an exercise where someone has asked you to perform a task, first you will think about how to solve the task, then you will write out a thoughtful answer for it. Remember, you must always reason and plan your response first, then say [FINAL ANSWER] and then give your full final answer.  Let\'s think step by step.'
            zs_cot = f'Read the question below then think before you speak.  You should assume that a reader will only see what you speak, so use your thoughts to guide what you will say at a high level (remember some people still like to hear the answer broken down step by step).  When you are done thinking and planning what you will say, output [Final Answer] followed by your full response to the question.  This is a task where you will think about how to solve the task, then you will write out a thoughtful answer for it.  Remember, you must always reason and plan your response first, then say [FINAL ANSWER] and then give your full final answer.\n\n{ex["input"]}\n\nRemember, first think and plan on what you will say, then say [Final Answer] and give your full final answer that you would want someone to read as a response to the given task or question. Your thoughts should be a high level reasoning and plans of what to say. What you say can include step by step breakdowns of the answer if applicable. Let\'s think step by step before we speak.'
            zs_direct = f'{ex["input"]}\n\nGive a direct answer to the question.'

            examples.append({
                **ex,
                'question': ex['input'],
                'answerKey': answerKey,
                'choices': choices,
                'dataset_type': self.dataset_types.biggen_bench,
                'prompt_parts': {
                    'zs_cot_prompt': zs_cot,
                    'zs_cotless_prompt': zs_direct,
                    'cot_system_prompt': ex["system_prompt"],
                    'cotless_system_prompt': ex["system_prompt"]
                },
                'answer_choice_tokens': None,
                'answer': answer,
                'answer_index': 0,
            })

        examples = self.build_fs_examples(examples)

        # TODO - remove this later.
        # examples = [x for x in examples if x['capability'] == 'reasoning']

        random.shuffle(examples)
        return examples

    def build_fs_examples(self, examples):

        for ex_idx, example in enumerate(examples):
            task = example['task']
            prev_ex = None
            next_ex = None
            if ex_idx != 0:
                prev_ex = examples[ex_idx - 1]
            if ex_idx != len(examples) - 1:
                next_ex = examples[ex_idx + 1]

            comparison = None
            if prev_ex and prev_ex['task'] == task:
                comparison = prev_ex
            elif next_ex and next_ex['task'] == task:
                comparison = next_ex
            else:
                raise Exception('No comparison found')

            fs_cot_prompt = f'Think before you speak.  You will be given a question and you must give a thoughtful answer. To give a thoughtful answer you should think, then give a final open ended response.  Here is an example of a question and a correct open ended response.\n\n###Question:\n{comparison["question"]}\n\n###An example of a correct open ended response (after applying thought):\n{comparison["reference_answer"]}\n\n---\nUsing the example above as a guide, craft a good opened response by first thinking and planning what you will say, then say [Final Answer] and give your open ended response.\n\n###Question:\n{example["input"]}\n\nRemember, first think and plan on what you will say, then say [Final Answer] and give your full final answer that you would want someone to read as a response to the given task or question. Your thoughts should be a high level reasoning and plans of what to say. What you say can include step by step breakdowns of the answer if applicable. Let\'s think step by step before we speak.'
            fs_direct_prompt = f'###Question: {comparison["input"]}\n\nGive a direct answer to the question.\n\n###Example answer:\n{comparison["reference_answer"]}\n\nUse the example above to guide how you should answer the question below.  Please answer the question as directly as possible.\n\n###Question you should answer:\n{example["question"]}\n\nGive a direct answer to the question.\n###Your answer:'

            example['prompt_parts']['fs_cot_prompt'] = fs_cot_prompt
            example['prompt_parts']['fs_cotless_prompt'] = fs_direct_prompt

        return examples
    @classmethod
    def evaluate_response(
            cls,
            model_responses,
            example,
            randomly_select_when_unparsable: bool = False,
            *args, model=None, **kwargs
    ):
        assert model is not None, 'Model must be provided for evaluation'

        returned_answers = []

        for response in model_responses:
            part_to_eval = response
            if '[Final Answer]' in response:
                part_to_eval = response.split('[Final Answer]')[1].strip()
            elif '[FINAL ANSWER]' in response:
                part_to_eval = response.split('[FINAL ANSWER]')[1].strip()
            elif '**Final Answer**' in response:
                part_to_eval = response.split('**Final Answer**')[1].strip()
            elif 'Final Answer:' in response:
                part_to_eval = response.split('Final Answer:')[1].strip()

            evaluation_prompt = f"""
###Task Description:
An instruction (might include an Input inside it), a response to evaluate, a reference answer that gets a score of 5, and a score rubric representing a evaluation criteria are given.
1. Write a detailed feedback that assess the quality of the response strictly based on the given score rubric, not evaluating in general.
2. After writing a feedback, write a score that is an integer between 1 and 5. You should refer to the score rubric.
3. The output format should look as follows: "Feedback: (write a feedback for criteria) [RESULT] (an integer number between 1 and 5)"
4. Please do not generate any other opening, closing, and explanations.

###The instruction to evaluation:
{example["input"]}

###Response to evaluate:
{part_to_eval}

###Reference Answer (Score 5):
{example["reference_answer"]}

###Score Rubrics:
Criteria:
{example["score_rubric"]["criteria"]}

Description of a Score 1 response:
{example["score_rubric"]["score1_description"]}

Description of a Score 2 response:
{example["score_rubric"]["score2_description"]}

Description of a Score 3 response:
{example["score_rubric"]["score3_description"]}

Description of a Score 4 response:
{example["score_rubric"]["score4_description"]}

Description of a Score 5 response:
{example["score_rubric"]["score5_description"]}

###Feedback:
Remember, you must strictly evaluate the response based on the given score rubric, and finish your output in the format of "(...) [RESULT] <score>", where <score> is a number between 1 and 5.
            """.strip()
            out = model.parse_out(model.inference(evaluation_prompt, temperature=0.0))
            out = out[0]
            try:
                score = int(out.split('[RESULT]')[1].strip())
            except Exception:
                returned_answers.append({'errored': True, 'correct': False, 'model_response': None, 'answer_line': None, 'model_answer': None, 'raw_model_answer': None})
                continue
            correct = score >= 4

            returned_answers.append(
                {'answer_label': 'E', 'model_response': response, 'answer_line': part_to_eval,
                 'correct': correct, 'answer_span': [response.index(part_to_eval), len(response)], 'evaluator_response': out, 'evaluator_model': repr(model), 'raw_model_answer': response + "\n\n[EVALUATOR OUTPUT]\n\n" + out,
                 'answer_randomly_sampled': False, 'model_answer': str(score), **example})

        return returned_answers

    @classmethod
    def custom_evaluate_response(cls, model_responses, example, *args, model=None, **kwargs):
        return None




if __name__ == '__main__':
    from torch.utils.data import DataLoader

    dataset = BigGenBenchDataset()

    ex = dataset[0]


    responses = [
        """
To watch a movie without leaving home, we need a device or service that can stream or play movies. The options given are:

( A ) drive in movie - This is not a device or service for home use.
( B ) drive in movie - Again, this is not a device or service for home use.
( C ) television - While a television can be used to watch movies, it requires a separate device like a DVD player, cable box, or streaming device to play movies at home.
( D ) video store - This is a physical location where you can rent or buy movies on DVD or Blu-ray, but it doesn't allow you to watch movies at home without additional equipment.
( E ) show - This term is too vague to be the correct answer.

The best answer is:

Answer: C) television (with a streaming device or cable box) or E) (a streaming service, such as Netflix, Hulu, Amazon Prime Video, etc.)

        """
    ]


    metrics = dataset.evaluate_response(responses, ex)
    print([x['model_answer'] for x in metrics])
