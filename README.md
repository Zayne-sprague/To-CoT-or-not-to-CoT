# TO COT OR NOT TO COT? CHAIN-OF-THOUGHT HELPS MAINLY ON MATH AND SYMBOLIC REASONING

[Paper Link](https://arxiv.org/abs/2409.12183)

[Data from meta-analysis](https://docs.google.com/spreadsheets/d/1zCxzKUg9BrbNfqJY1BEmNgCN0d38jaF8ads4Dyt5nbE/edit?usp=sharing)

# install and Setup

## Python
We used python3.9

```terminal
pip install -r requirements.txt
```

## Keys

Copy and paste the key_handler__template.py file into key_handler.py

```terminal
cp key_handler__template.py key_handler.py
```

Then put in all your keys for OpenAI / Huggingface Token / Google / Claude etc. depending on the models you want to run.

# Running Stuff

You can run all our experiments we did in `experiments/section_4__cot_evals`

## Zero-shot vs Few-shot vs CoT vs Direct Answer

This analysis is done per model and can be uploaded to your Huggingface repo but will also be stored locally.

To run it:
```terminal
python -m zeroshot_cot_experiments.py --model=openai/gpt-4o-mini-2024-07-18 --output_folder=./outs/test --eval_model=openai/gpt-4o-mini-2024-07-18 --num_samples=10 --is_closed_source=True --skip_fs_direct --skip_fs_cot --datasets agieval_lsat_lr agieval_lsat_ar agieval_lsat_rc
```

- You can see the models that are available in `src/model` and how they are initialized in `model.py` load_model fn.

  - OpenAI/Claude/Gemini models use their normal API names prefaced with `openai`
  - Huggingface models must be hosted somewhere with vLLM and then you can call them via `--model=vllm_endpoint/http://127.0.0.1:60271/v1/completions<model>deepseek-ai/DeepSeek-R1-Distill-Llama-70B`
  
- See all datasets that are allowed in the file `zeroshot_cot_experiments.py` at the bottom.

- eval_model is the model we use as an LLM-as-a-judge (for Biggen Bench)
 
- You can skip individual experiment settings like FewShot CoT via `--skip_fs_cot`
 
### Note
Because the datasets are changing on Huggingface and Llama 3.1 evals are no longer there (Llama changes their eval repos) our script uses the prompts stored in our own HF repo to keep everything reproducible.

## Plots Charts and Analyses 

We include in this repo all the main figures and analyses from our paper.  However, they all pull from our google sheets or Huggingface Repo.  If you want to reproduce our results with your own data, you'll have to update how we load in the data (though that should be pretty easy). We are noting this here just so people know that the outputs from `zeroshot_cot_experiments.py` are not automatically hooked into all the plotting scripts.

## Calling Models

We made our own special way to call models :P 

You can see examples in `scripts/example__calling_models.py` for details on how to do it. Really it was just a way for us to unify calling stuff back before a ton of other packages came out that did this.

## Installing Gemini

To get Gemini working you need a `google_service_key.json`, a project ID, a project location, threshold limits setup (on the console) and then to be logged in via the CLI.  There has to be an easier way, but for now that's how this was setup.