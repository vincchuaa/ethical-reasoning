"""
Provides a single `call_model()` interface for calling different LLM backends.

Uses native SDKs for API models (OpenAI, Anthropic) and direct HuggingFace
transformers loading for local models. No litellm or vLLM dependency.
"""

import os
import threading
from typing import Dict, List, Optional, Union

import torch
from dotenv import load_dotenv
from openai import OpenAI
from transformers import AutoModelForCausalLM, AutoTokenizer

import anthropic

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
DEEPINFRA_API_KEY = os.getenv("DEEPINFRA_API_KEY")

MessageList = List[Dict[str, str]]

MODEL_MAPPING = {
    "gpt": "gpt-4o",
    "claude": "claude-sonnet-4-20250514",
    "d-ds": "deepseek-ai/DeepSeek-V3.2",
    "llama": "meta-llama/Llama-3.1-8B-Instruct",
    "qwen": "Qwen/Qwen3-8B",
}

_hf_model_cache: dict = {}
_hf_load_lock = threading.Lock()


def load_huggingface_model(model_path: str) -> dict:
    """Load a HuggingFace model + tokenizer, with caching.

    Thread-safe: concurrent workers will wait for the first loader to finish,
    then share the cached model.
    """
    if model_path in _hf_model_cache:
        return _hf_model_cache[model_path]

    with _hf_load_lock:
        if model_path in _hf_model_cache:
            return _hf_model_cache[model_path]

        if torch.cuda.is_available():
            n_gpus = torch.cuda.device_count()
            print(
                f"Loading model: {model_path} → auto "
                f"(HF model #{len(_hf_model_cache) + 1}, {n_gpus} visible GPU(s))"
            )
        else:
            print(f"Loading model: {model_path} → cpu")

        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )

        _hf_model_cache[model_path] = {
            "model": model,
            "tokenizer": tokenizer,
        }
        return _hf_model_cache[model_path]


def call_huggingface_model(
    messages: MessageList,
    model_path: str,
    max_tokens: int = 4096,
    force_prefix: Optional[str] = None,
) -> str:
    """Run inference on a local HuggingFace model."""
    try:
        cached = load_huggingface_model(model_path)
        model = cached["model"]
        tokenizer = cached["tokenizer"]

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        inputs = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        original_prompt_len = inputs["input_ids"].shape[-1]

        if force_prefix:
            prefix_ids = tokenizer.encode(
                force_prefix, add_special_tokens=False, return_tensors="pt"
            ).to(model.device)
            inputs["input_ids"] = torch.cat([inputs["input_ids"], prefix_ids], dim=-1)
            inputs["attention_mask"] = torch.ones_like(inputs["input_ids"])

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=False,
                temperature=0.7,
                pad_token_id=tokenizer.eos_token_id,
            )

        response_text = tokenizer.decode(
            outputs[0][original_prompt_len:],
            skip_special_tokens=True,
        )
        return response_text

    except Exception as e:
        print(f"Error in call_huggingface_model for {model_path}: {e}")
        return f"ERROR: {str(e)}"


def call_gpt(
    prompt: MessageList,
    model: str = "gpt-4o",
    temperature: float = 0.7,
    max_tokens: int = 4096,
    force_prefix: Optional[str] = None,
) -> str:
    if not OPENAI_API_KEY:
        return "ERROR: OPENAI_API_KEY environment variable not set."
    client = OpenAI(api_key=OPENAI_API_KEY)
    messages = list(prompt)
    if force_prefix:
        messages.append({"role": "assistant", "content": force_prefix})
    completion = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    text = completion.choices[0].message.content or ""
    return (force_prefix + text) if force_prefix else text


def call_deepinfra_model(
    prompt: MessageList,
    model: str,
    temperature: float = 0.7,
    max_tokens: int = 4096,
    force_prefix: Optional[str] = None,
) -> str:
    """Call a model hosted on DeepInfra via its OpenAI-compatible endpoint."""
    if not DEEPINFRA_API_KEY:
        return "ERROR: DEEPINFRA_API_KEY environment variable not set."
    try:
        client = OpenAI(
            api_key=DEEPINFRA_API_KEY,
            base_url="https://api.deepinfra.com/v1/openai",
        )
        messages = list(prompt)
        if force_prefix:
            messages.append({"role": "assistant", "content": force_prefix})
        completion = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        text = completion.choices[0].message.content or ""
        return (force_prefix + text) if force_prefix else text
    except Exception as e:
        print(f"Error calling DeepInfra model '{model}': {e}")
        return f"ERROR: {str(e)}"


def call_claude_model(
    prompt: MessageList,
    model: str = "claude-sonnet-4-20250514",
    temperature: float = 0.7,
    max_tokens: int = 4096,
    force_prefix: Optional[str] = None,
) -> str:
    if not ANTHROPIC_API_KEY:
        return "ERROR: ANTHROPIC_API_KEY environment variable not set."
    try:
        client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

        system_text = ""
        user_messages = []
        for msg in prompt:
            if msg["role"] == "system":
                system_text += msg["content"] + "\n"
            else:
                user_messages.append(msg)

        if not user_messages:
            user_messages = [{"role": "user", "content": ""}]

        if force_prefix:
            user_messages = list(user_messages) + [{"role": "assistant", "content": force_prefix}]

        kwargs = {
            "model": model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": user_messages,
        }
        if system_text.strip():
            kwargs["system"] = system_text.strip()

        response = client.messages.create(**kwargs)
        text = response.content[0].text
        return (force_prefix + text) if force_prefix else text

    except Exception as e:
        print(f"Error calling Claude model '{model}': {e}")
        return f"ERROR: {str(e)}"


def call_model(
    model_alias: str,
    prompt: Union[str, List[Dict]],
    system_prompt: Optional[str] = None,
    temperature: float = 0.7,
    max_tokens: int = 4096,
    force_prefix: Optional[str] = None,
) -> str:
    """
    Call an LLM and return the response text.

    Args:
        model_alias: Key in MODEL_MAPPING (e.g. "gpt", "claude", "hf-llama3.1-8b").
        prompt: Either a plain string or a list of {"role": ..., "content": ...} dicts.
        system_prompt: Optional system message prepended to the conversation.
        temperature: Sampling temperature.
        max_tokens: Maximum tokens in the response.
        force_prefix: If set, the model's response is forced to begin with this
            string (assistant prefill). The returned text always includes the prefix.

    Returns:
        The assistant's response text.
    """
    model_alias = model_alias.lower()
    if model_alias not in MODEL_MAPPING:
        raise ValueError(
            f"Unknown model alias '{model_alias}'. "
            f"Available: {', '.join(MODEL_MAPPING.keys())}"
        )

    full_model_name = MODEL_MAPPING[model_alias]

    messages: MessageList = []
    if isinstance(prompt, str):
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
    elif isinstance(prompt, list):
        if system_prompt and (not prompt or prompt[0].get("role") != "system"):
            messages.append({"role": "system", "content": system_prompt})
        messages.extend(prompt)
    else:
        return "ERROR: Prompt must be a string or a list of messages."

    if "gpt" in model_alias:
        return call_gpt(
            prompt=messages,
            model=full_model_name,
            temperature=temperature,
            max_tokens=max_tokens,
            force_prefix=force_prefix,
        )
    elif "claude" in model_alias:
        return call_claude_model(
            prompt=messages,
            model=full_model_name,
            temperature=temperature,
            max_tokens=max_tokens,
            force_prefix=force_prefix,
        )
    elif model_alias.startswith("d-"):
        return call_deepinfra_model(
            prompt=messages,
            model=full_model_name,
            temperature=temperature,
            max_tokens=max_tokens,
            force_prefix=force_prefix,
        )
    else:
        return call_huggingface_model(
            messages=messages,
            model_path=full_model_name,
            max_tokens=max_tokens,
            force_prefix=force_prefix,
        )
