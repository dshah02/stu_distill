from transformers import CONFIG_MAPPING, MODEL_MAPPING
from transformers.models.auto.configuration_auto import _LazyConfigMapping
from transformers.models.auto.auto_factory import _LazyAutoMapping
from transformers.models.auto import MODEL_FOR_CAUSAL_LM_MAPPING

from Transformer_564M import MiniTransformer, MiniTransformerConfig
import lm_eval as evaluator
import os
os.environ["HF_ALLOW_CODE_EVAL"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Dynamically register MiniTransformer
if isinstance(CONFIG_MAPPING, _LazyConfigMapping):
    CONFIG_MAPPING._extra_content["minitransformer"] = MiniTransformerConfig
else:
    CONFIG_MAPPING.update({"minitransformer": MiniTransformerConfig})

if isinstance(MODEL_MAPPING, _LazyAutoMapping):
    MODEL_MAPPING._extra_content[MiniTransformerConfig] = MiniTransformer
else:
    MODEL_MAPPING.update({MiniTransformerConfig: MiniTransformer})

if isinstance(MODEL_FOR_CAUSAL_LM_MAPPING, _LazyAutoMapping):
    MODEL_FOR_CAUSAL_LM_MAPPING._extra_content[MiniTransformerConfig] = MiniTransformer
else:
    MODEL_FOR_CAUSAL_LM_MAPPING.update({MiniTransformerConfig: MiniTransformer})

print("Registered MiniTransformer model and configuration.")

# Define the list of tasks you want to evaluate
tasks = [
    "mmlu",
    "hellaswag",
    "piqa",
    #"siqa",
    "boolq",
    "winogrande",
    "commonsense_qa",
    "openbookqa",
    #"arc",
    "arc_easy",
    "arc_challenge",
    #"triviaqa",
    #"nq_open",
    #"humaneval",
    #"mbpp",
    #"gms8k",
    #"hendrycks_math",
    #"mathqa",
    #"minerva_math",
    #"score",
    #"asdiv",
    #"agieval",
    #"bigbench",
]

# Few-shot settings dictionary
tasks_fewshot = {
    "hellaswag": 0,       # 0-shot per table
    "mmlu": 5,            # MMLU → 5-shot
    "piqa": 0,            # 0-shot
    #"siqa": 0,            # SocialIQA → 0-shot
    "boolq": 0,           # BooIQ → 0-shot
    "winogrande": -1,     # partial score only (no few-shot info)
    "commonsense_qa": 7,  # 7-shot
    "openbookqa": -1,     # no few-shot info given
    #"arc": -1,            # no few-shot info (table only had ARC-e, ARC-c)
    "arc_easy": -1,       # ARC-e (no few-shot info)
    "arc_challenge": -1,  # ARC-c (no few-shot info)
    #"triviaqa": 5,        # 5-shot
    #"nq_open": 5,         # Natural Questions → 5-shot
    #"humaneval": -1,      # pass@1 only, no few-shot info
    #"mbpp": 3,            # 3-shot
    #"gms8k": -1,          # GSM8K mention, but no shot info
    #"hendrycks_math": 4,  # MATH → 4-shot
    #"mathqa": -1,         # no few-shot info
    #"minerva_math": -1,   # no few-shot info
    #"score": -1,          # no mention
    #"asdiv": -1,          # no mention
    #"agieval": -1,        # no few-shot info
    #"bigbench": -1,        # no few-shot info
}

# Initialize a dictionary to store all results
all_results = {}

for task in tasks:
    print(f"Evaluating task: {task}")
    # Prepare the arguments for evaluation
    eval_kwargs = dict(
        model="hf",
        model_args=(
            "pretrained=Hazan-Lab/Transformer_564M,"
            "trust_remote_code=True,"
            "dtype=bfloat16,"
            "cache_dir=/scratch/gpfs/hd0216/New Project/hugging_face/.cache"
        ),
        tasks=[task],
        batch_size=1,
        device="cuda:0"
    )

    # Determine if we should specify num_fewshot
    few_shot_value = tasks_fewshot.get(task, -1)  # default to -1 if not in dict
    if few_shot_value != -1:
        eval_kwargs["num_fewshot"] = few_shot_value

    # Evaluate each task individually
    results = evaluator.simple_evaluate(**eval_kwargs)

    # Extract results for the current task
    task_result = results["results"].get(task, {})
    all_results[task] = task_result

    print(f"Results for {task}:")
    print(task_result)
    print("\n" + "="*50 + "\n")



# Optionally, print all results at once
print("All Evaluation Results:")
for task, result in all_results.items():
    print(f"{task}: {result}")
