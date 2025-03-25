import os
import lm_eval as evaluator
from transformers import AutoConfig, AutoModel, AutoModelForCausalLM
from tensorized_filters.models.transformer_500M.model import Transformer as MiniTransformer, TransformerConfig as MiniTransformerConfig

os.environ["HF_ALLOW_CODE_EVAL"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

AutoConfig.register("minitransformer", MiniTransformerConfig)
AutoModel.register(MiniTransformerConfig, MiniTransformer)
AutoModelForCausalLM.register(MiniTransformerConfig, MiniTransformer)

print("Registered MiniTransformer model and configuration.")

tasks = [
    # "mmlu",
    "hellaswag",
    # "piqa",
    # "siqa",
    # "boolq",
    # "winogrande",
    # "commonsense_qa",
    # "openbookqa",
    # "arc",
    # "arc_easy",
    # "arc_challenge",
    # "triviaqa",
    # "nq_open",
    # "humaneval",
    # "mbpp",
    # "gms8k",
    # "hendrycks_math",
    # "mathqa",
    # "minerva_math",
    # "score",
    # "asdiv",
    # "agieval",
    # "bigbench",
]

tasks_fewshot = {
    "hellaswag": 0,
    # "mmlu": 5,
    # "piqa": 0,
    # "siqa": 0,
    # "boolq": 0,
    # "winogrande": -1,
    # "commonsense_qa": 7,
    # "openbookqa": -1,
    # "arc": -1,
    # "arc_easy": -1,
    # "arc_challenge": -1,
    # "triviaqa": 5,
    # "nq_open": 5,
    # "humaneval": -1,
    # "mbpp": 3,
    # "gms8k": -1,
    # "hendrycks_math": 4,
    # "mathqa": -1,
    # "minerva_math": -1,
    # "score": -1,
    # "asdiv": -1,
    # "agieval": -1,
    # "bigbench": -1,
}

all_results = {}

for task in tasks:
    print(f"Evaluating task: {task}")
    eval_kwargs = dict(
        model="hf",
        model_args=(
            "pretrained=Hazan-Lab/Transformer_564M,"
            "trust_remote_code=True,"
            "dtype=bfloat16,"
            "cache_dir=/scratch/gpfs/mn4560/hazan-lab/tensorized_filters/tensorized_filters/eval/cache"
        ),
        tasks=[task],
        batch_size="auto",
        device="cuda:0"
    )
    few_shot_value = tasks_fewshot.get(task, -1)
    if few_shot_value != -1:
        eval_kwargs["num_fewshot"] = few_shot_value
    results = evaluator.simple_evaluate(**eval_kwargs)
    task_result = results["results"].get(task, {})
    all_results[task] = task_result
    print(f"Results for {task}:")
    print(task_result)
    print("\n" + "="*50 + "\n")

print("All Evaluation Results:")
for task, result in all_results.items():
    print(f"{task}: {result}")
