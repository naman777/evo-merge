"""Loading the merge target model and scoring it on math problems."""
import time

import torch

from .answers import extract_prediction, is_correct
from .merge import LayerwiseMerger, load_state_dict

DTYPES = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}


def set_rope_theta(config, theta):
    """Set the RoPE base on both the old (rope_theta) and new (rope_parameters) config layouts."""
    config.rope_theta = theta
    params = getattr(config, "rope_parameters", None)
    if isinstance(params, dict):
        params["rope_theta"] = theta


def get_rope_theta(config):
    params = getattr(config, "rope_parameters", None)
    if isinstance(params, dict) and "rope_theta" in params:
        return params["rope_theta"]
    return getattr(config, "rope_theta", None)


def load_merge_setup(cfg):
    """Load the live model (on cfg.device), its tokenizer and a merger over base + both parents."""
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    skeleton = cfg.model_a if cfg.config_from == "a" else cfg.model_b
    model_config = AutoConfig.from_pretrained(skeleton)
    if cfg.rope_theta is not None:
        set_rope_theta(model_config, cfg.rope_theta)
    try:
        model = AutoModelForCausalLM.from_pretrained(skeleton, config=model_config, dtype=DTYPES[cfg.dtype])
    except TypeError:  # transformers < 4.56 calls it torch_dtype
        model = AutoModelForCausalLM.from_pretrained(skeleton, config=model_config, torch_dtype=DTYPES[cfg.dtype])
    model.to(cfg.device).eval()

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_a)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"loading weights: base={cfg.base_model} a={cfg.model_a} b={cfg.model_b}", flush=True)
    base = load_state_dict(cfg.base_model)
    parents = [load_state_dict(cfg.model_a), load_state_dict(cfg.model_b)]
    merger = LayerwiseMerger(model, base, parents, model_config.num_hidden_layers,
                             method=cfg.merge_method, density=cfg.density, seed=cfg.seed)
    print(f"model ready: {model_config.num_hidden_layers} layers, rope_theta={get_rope_theta(model_config)}, "
          f"genome size {merger.genome_size}", flush=True)
    return model, tokenizer, merger


class Evaluator:
    """Greedy-decodes answers to a fixed problem set and reports accuracy."""

    def __init__(self, model, tokenizer, problems, system_prompt, max_new_tokens=512, batch_size=32):
        self.model, self.tokenizer = model, tokenizer
        self.max_new_tokens, self.batch_size = max_new_tokens, batch_size
        prompts = [self._prompt(system_prompt, p["question"]) for p in problems]
        # Sort by prompt length so each batch carries little padding.
        order = sorted(range(len(problems)), key=lambda i: len(prompts[i]))
        self.problems = [problems[i] for i in order]
        self.prompts = [prompts[i] for i in order]
        stop = {tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|im_end|>"),
                tokenizer.convert_tokens_to_ids("<|endoftext|>")}
        self.stop_ids = sorted(i for i in stop if isinstance(i, int) and i >= 0)

    def _prompt(self, system_prompt, question):
        messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": question}]
        return self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    @torch.no_grad()
    def evaluate(self, keep_outputs=False):
        start = time.time()
        records = []
        for i in range(0, len(self.prompts), self.batch_size):
            batch = self.prompts[i : i + self.batch_size]
            enc = self.tokenizer(batch, return_tensors="pt", padding=True).to(self.model.device)
            out = self.model.generate(
                **enc, max_new_tokens=self.max_new_tokens, do_sample=False,
                eos_token_id=self.stop_ids, pad_token_id=self.tokenizer.pad_token_id,
            )
            texts = self.tokenizer.batch_decode(out[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
            for problem, text in zip(self.problems[i : i + self.batch_size], texts):
                pred = extract_prediction(text)
                record = {"id": problem["id"], "gold": problem["gold"], "pred": pred,
                          "correct": is_correct(pred, problem["gold"])}
                if keep_outputs:
                    record["output"] = text
                records.append(record)
        accuracy = sum(r["correct"] for r in records) / max(1, len(records))
        return accuracy, records, time.time() - start
