"""
Validation loss: 2.8789, trained on ~59.7B FineWeb-Edu tokens.
This is a base model and can frequently hallucinate.
"""

import json

from time import time

import torch
import torch.nn.functional as F
import tiktoken

from safetensors import safe_open

from model import FlashSTU, FlashSTUConfig, get_spectral_filters


CHECKPOINT_PATH = "model_step-114000.safetensors"
CONFIG_PATH = "config.json"


def apply_compile(model: nn.Module) -> None:
    """
    Apply torch.compile to each layer. This makes compilation efficient
    due to repeated structure. Alternatively, one can just compile the whole model.
    """
    print(f"Compiling each {model.__class__.__name__} layer with torch.compile...")
    start = time.perf_counter()
    for idx, layer in model.layers.named_children():
        compiled_layer = torch.compile(layer, mode="max-autotune", fullgraph=True)
        model.layers.register_module(idx, compiled_layer)
    end = time.perf_counter()
    print(f"Finished compiling each {model.__class__.__name__} layer in {end - start:.4f} seconds.")


def load_stu_model(config_path: str, checkpoint_path: str, device: torch.device):
    with open(config_path, "r") as f:
        config_data = json.load(f)

    dim = config_data["dim"]
    num_heads = config_data["num_heads"]
    num_layers = config_data["num_layers"]
    num_eigh = config_data["num_eigh"]
    seq_len = config_data["seq_len"]
    use_hankel_L = config_data["use_hankel_L"]
    window_size = config_data["window_size"]
    vocab_size = config_data["vocab_size"]
    mlp_scale = config_data["mlp_scale"]
    bias = config_data["bias"]
    dropout = config_data["dropout"]
    softcap = config_data["softcap"]
    theta = config_data["theta"]
    torch_compile = config_data["torch_compile"]
    torch_dtype = getattr(torch, config_data["torch_dtype"])

    model_config = FlashSTUConfig(
        dim=dim,
        num_heads=num_heads,
        num_layers=num_layers,
        seq_len=seq_len,
        window_size=window_size,
        vocab_size=vocab_size,
        mlp_scale=mlp_scale,
        bias=bias,
        dropout=dropout,
        softcap=softcap,
        theta=theta,
        torch_dtype=torch_dtype,
    )

    spectral_filters = get_spectral_filters(seq_len, num_eigh, use_hankel_L, device, torch_dtype)
    model = FlashSTU(model_config, spectral_filters)
    model = model.to(device=device, dtype=torch_dtype)

    print(f"Loading checkpoint from {checkpoint_path}...")
    state_dict = {}
    start_time = time()

    if checkpoint_path.endswith(".safetensors"):
        with safe_open(checkpoint_path, framework="pt", device=device.type) as f:
            for k in f.keys():
                state_dict[k] = f.get_tensor(k)
    elif checkpoint_path.endswith(".pt"):
        state_dict = torch.load(checkpoint_path, map_location="cpu")
    else:
        raise ValueError(f"Unsupported checkpoint format: {checkpoint_path}")
    print(f"Checkpoint loaded in {time() - start_time:.2f} seconds.")

    model.load_state_dict(state_dict, strict=True)
    print("Model weights loaded successfully!")

    if torch_compile:
        model = apply_compile(model)

    model.eval()

    return model, config_data


def generate_text(
    model,
    tokenizer,
    prompt,
    num_return_sequences=1,
    max_length=512,
    device="cuda",
    temperature=1.0,
    top_k=50,
):
    """
    Generate text from the given prompt using top-k sampling.

    Args:
        model: The FlashSTU model instance.
        tokenizer: The tokenizer used for encoding/decoding.
        prompt (str): Input prompt text.
        num_return_sequences (int): How many sequences to return.
        max_length (int): Maximum length of generated tokens.
        device: torch device.
        temperature (float): Sampling temperature. Higher = more random.
        top_k (int): Top-K sampling parameter.

    Returns:
        list[str]: A list of generated text sequences.
    """
    model.eval()

    # Encode prompt tokens.
    tokens = torch.tensor(
        [tokenizer.encode(prompt, allowed_special={"<|endoftext|>"})],
        device=device,
    )
    tokens = tokens.repeat(num_return_sequences, 1)

    sample_rng = torch.Generator(device=device)
    sample_rng.manual_seed(1746)

    eos_token_id = tokenizer.encode(
        "<|endoftext|>", allowed_special={"<|endoftext|>"}
    )[0]

    with torch.no_grad():
        for _ in range(max_length - tokens.size(1)):
            with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                # Fwd pass. Inspect logits here.
                logits = model(tokens)     # shape: [batch, seq, vocab]
                logits = logits[:, -1, :]  # last token logits

                # Apply temperature scaling.
                if temperature > 0:
                    logits = logits / temperature

            # Compute probabilities.
            probs = F.softmax(logits, dim=-1)

            # Top-K sampling.
            top_k_probs, top_k_indices = torch.topk(probs, top_k, dim=-1)
            ix = torch.multinomial(top_k_probs, 1, generator=sample_rng)
            next_token = torch.gather(top_k_indices, -1, ix)

            # Append next token.
            tokens = torch.cat((tokens, next_token), dim=1)

            # Stop if EOS token is generated.
            if (next_token == eos_token_id).any():
                break

    # Decode all sequences.
    generated_sequences = []
    for i in range(num_return_sequences):
        decoded = tokenizer.decode(tokens[i].tolist())
        generated_sequences.append(decoded)

    return generated_sequences


def main():
    device = torch.device("cuda")

    # Load model and config.
    model, config_data = load_stu_model(CONFIG_PATH, CHECKPOINT_PATH, device)
    tokenizer = tiktoken.get_encoding("o200k_base")

    # Collect prompt(s) from user.
    prompts = []
    while True:
        prompt = input("Enter a prompt (press enter with no text to finish): ")
        if not prompt:
            break
        prompts.append(prompt.strip())

    if len(prompts) == 0:
        print("No prompts provided. Exiting.")
        return

    # -------------------------------------------------------------------
    # BASE SETTINGS:
    BASE_TEMPERATURE = 0.7  # Increase for more randomness.
    BASE_TOP_K = 50         # Limit sampling to the top k tokens.
    MAX_LENGTH = 512        # Maximum number of tokens to generate.
    # -------------------------------------------------------------------

    total_tokens = 0
    start_time = time()

    for i, prompt in enumerate(prompts, 1):
        print(f"Generating text for prompt {i}: {prompt}")
        generated_texts = generate_text(
            model,
            tokenizer,
            prompt,
            num_return_sequences=1,
            max_length=MAX_LENGTH,
            device=device,
            temperature=BASE_TEMPERATURE,
            top_k=BASE_TOP_K,
        )
        for gen_text in generated_texts:
            print(f"\nPrompt: {prompt}")
            print(f"Generated Text: {gen_text}\n")
            total_tokens += len(
                tokenizer.encode(gen_text, allowed_special={"<|endoftext|>"})
            )
    end_time = time()
    tokens_per_second = total_tokens / (end_time - start_time)
    print(f"Tokens per second: {tokens_per_second:.2f}")


if __name__ == "__main__":
    main()
