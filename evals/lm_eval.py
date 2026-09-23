"""Zero-shot evaluation through the pinned official lm-eval harness."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from lm_eval.api.model import TemplateLM
from lm_eval.models.utils import Collator, normalize_gen_kwargs
from transformers import AutoTokenizer, PreTrainedModel, StopStringCriteria

from evals.loading import load_complete_model
from evals.scoring import score_tokens
from training.compile import compile_feedback
from training.precision import attention_kernel_context
from white_matter.models import register_models
from white_matter.modules.precision import model_autocast_context

register_models()

PAPER_TASKS = [
    "lambada_openai",
    "wikitext",
    "piqa",
    "winogrande",
    "boolq",
    "hellaswag",
    "arc_easy",
    "arc_challenge",
    "openbookqa",
]
MAX_EVAL_CONTEXT_TOKENS = 1_024


def configure_eval_compiler() -> None:
    # WikiText's rolling windows add shape variants after batched likelihood tasks.
    # The compiled pool writer needs more than PyTorch's default eight graph variants.
    from torch._dynamo import config as dynamo_config

    dynamo_config.recompile_limit = max(dynamo_config.recompile_limit, 64)


class _ContinuationStopStrings(StopStringCriteria):
    def __init__(self, tokenizer, stops: list[str], prompt_length: int):
        super().__init__(tokenizer, stops)
        self.prompt_length = prompt_length

    def __call__(self, input_ids, scores, **kwargs):
        return super().__call__(input_ids[:, self.prompt_length:], scores, **kwargs)


class WhiteMatterHarnessLM(TemplateLM):
    """Likelihood scoring and cached HF generation for the evaluation harness."""

    backend = "causal"

    def __init__(
        self,
        *,
        model: PreTrainedModel,
        tokenizer,
        batch_size: int,
        logits_chunk: int = 256,
        max_length: int = MAX_EVAL_CONTEXT_TOKENS,
    ) -> None:
        super().__init__()
        self.model = model.eval()
        self.tokenizer = tokenizer
        self._device = next(model.parameters()).device
        self.batch_size = int(batch_size)
        self.logits_chunk = int(logits_chunk)
        if self.batch_size <= 0 or self.logits_chunk <= 0 or max_length < 2:
            raise ValueError("batch_size and logits_chunk must be positive; max_length must be at least 2")
        self.max_length_val = min(
            int(model.config.max_position_embeddings),
            MAX_EVAL_CONTEXT_TOKENS, max_length,
        )
        self.num_passes = int(getattr(model.config, "num_passes", 1))
        self.cyclic_groups = int(getattr(model.config, "cyclic_groups", 8))
        self.is_cyclic = model.config.execution_mode == "cyclic"
        self._tokenizer_name = str(getattr(tokenizer, "name_or_path", "unknown"))

    @property
    def eot_token_id(self) -> int:
        return int(self.tokenizer.eos_token_id)

    @property
    def max_length(self) -> int:
        return self.max_length_val

    @property
    def max_gen_toks(self) -> int:
        return min(256, self.max_length - 1)

    @property
    def tokenizer_name(self) -> str:
        return self._tokenizer_name

    def tok_encode(self, string: str, **_: Any) -> list[int]:
        return self.tokenizer.encode(string, add_special_tokens=False)

    def tok_decode(self, tokens: list[int]) -> str:
        return self.tokenizer.decode(tokens, skip_special_tokens=True)

    @torch.inference_mode()
    def _hidden_pre_lm_head(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.is_cyclic:
            original_length = input_ids.shape[1]
            quantum = self.cyclic_groups * 64
            padded_length = ((original_length + quantum - 1) // quantum) * quantum
            if padded_length > original_length:
                pad_width = padded_length - original_length
                pad_id = int(self.tokenizer.pad_token_id)
                input_ids = F.pad(input_ids, (0, pad_width), value=pad_id)
                if attention_mask is not None:
                    attention_mask = F.pad(attention_mask, (0, pad_width))
        with attention_kernel_context(str(self._device)):
            with model_autocast_context(str(self._device)):
                return self.model.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    num_passes=self.num_passes,
                    return_dict=True,
                ).last_hidden_state

    def _prepare_full_ids(self, context: list[int], continuation: list[int]) -> tuple[list[int], int]:
        if len(continuation) >= self.max_length_val:
            raise ValueError("continuation does not fit in one likelihood window")
        full = (context + continuation)[-self.max_length_val :]
        continuation_start = len(full) - len(continuation)
        if continuation_start == 0:
            full = ([self.eot_token_id] + full)[-self.max_length_val :]
            continuation_start = len(full) - len(continuation)
        return full, continuation_start

    @torch.inference_mode()
    def _loglikelihood_tokens(self, requests, disable_tqdm: bool = False, **_: Any):
        del disable_tqdm
        results = [(0.0, True)] * len(requests)
        prepared = []
        for index, (_, context, continuation) in enumerate(requests):
            if not continuation:
                continue
            full, start = self._prepare_full_ids(list(context), list(continuation))
            prepared.append((index, full, start))
        prepared.sort(key=lambda item: len(item[1]), reverse=True)

        for batch_start in range(0, len(prepared), self.batch_size):
            batch = prepared[batch_start : batch_start + self.batch_size]
            max_length = max(len(full) for _, full, _ in batch)
            pad_id = int(self.tokenizer.pad_token_id)
            ids = [full + [pad_id] * (max_length - len(full)) for _, full, _ in batch]
            masks = [[1] * len(full) + [0] * (max_length - len(full)) for _, full, _ in batch]
            input_ids = torch.tensor(ids, dtype=torch.long, device=self._device)
            attention_mask = torch.tensor(masks, dtype=torch.long, device=self._device)

            hidden = self._hidden_pre_lm_head(input_ids, attention_mask)
            selected = torch.cat([hidden[row, start - 1 : len(full) - 1]
                                  for row, (_, full, start) in enumerate(batch)])
            targets = [token for _, full, start in batch for token in full[start:]]
            token_log_probs, greedy = score_tokens(
                selected, self.model.lm_head.weight, torch.tensor(targets, device=self._device),
                chunk_size=self.logits_chunk,
            )
            lengths = [len(full) - start for _, full, start in batch]
            for (request_index, _, _), scores, matches in zip(
                batch, token_log_probs.split(lengths), greedy.split(lengths), strict=True,
            ):
                results[request_index] = (float(scores.sum()), bool(matches.all()))
        return results

    @torch.inference_mode()
    def loglikelihood_rolling(self, requests, disable_tqdm: bool = False) -> list[float]:
        del disable_tqdm
        results = []
        for request in requests:
            tokens = [self.eot_token_id] + self.tok_encode(request.args[0])
            total = 0.0
            stride = self.max_length_val - 1
            for start in range(0, max(0, len(tokens) - 1), stride):
                chunk = tokens[start : start + self.max_length_val]
                input_ids = torch.tensor([chunk], dtype=torch.long, device=self._device)
                hidden = self._hidden_pre_lm_head(input_ids)[:, : len(chunk)]
                scores, _ = score_tokens(
                    hidden[0, :-1], self.model.lm_head.weight, input_ids[0, 1:], chunk_size=self.logits_chunk,
                )
                total += float(scores.sum())
            results.append(total)
        return results

    @torch.inference_mode()
    def generate_until(self, requests, disable_tqdm: bool = False) -> list[str]:
        del disable_tqdm
        if self.is_cyclic and getattr(self.model.config, "prefill_mode", None) is None:
            raise ValueError("choose prefill_mode='cyclic' or 'autoregressive' explicitly for generation")
        ordered = Collator(
            [request.args for request in requests],
            sort_fn=lambda item: -len(self.tok_encode(item[0])),
            group_by="gen_kwargs", group_fn=lambda item: item[1],
        )
        results = []
        for batch in ordered.get_batched(n=self.batch_size):
            options = normalize_gen_kwargs(batch[0][1], self.max_gen_toks)
            stops = options.pop("until")
            if any(not isinstance(stop, str) or not stop for stop in stops):
                raise ValueError("until must contain nonempty strings")
            limit = options.pop("max_gen_toks")
            if not 0 < limit < self.max_length:
                raise ValueError("max_gen_toks must be positive and smaller than max_length")
            if options.get("num_return_sequences", 1) != 1 or options.get("num_beams", 1) != 1:
                raise ValueError("evaluation supports one continuation per prompt without beam search")
            if not options["do_sample"]:
                options.pop("temperature", None)
            tokens = [(self.tok_encode(context) or [self.eot_token_id])[-(self.max_length - limit):]
                      for context, _ in batch]
            width = max(map(len, tokens))
            pad = self.tokenizer.pad_token_id
            pad = self.eot_token_id if pad is None else pad
            ids = torch.tensor([[pad] * (width - len(row)) + row for row in tokens], device=self._device)
            mask = torch.tensor([[0] * (width - len(row)) + [1] * len(row) for row in tokens], device=self._device)
            # Native HF stopping handles each row independently; dynamic KV supports left padding.
            options.update(use_cache=True, cache_implementation="dynamic", max_new_tokens=limit,
                           pad_token_id=pad, eos_token_id=self.eot_token_id, return_dict_in_generate=False)
            if stops:
                options["stopping_criteria"] = [*options.get("stopping_criteria", []),
                                                _ContinuationStopStrings(self.tokenizer, stops, width)]
            with attention_kernel_context(str(self._device)), model_autocast_context(str(self._device)):
                output = self.model.generate(ids, attention_mask=mask, num_passes=self.num_passes, **options)
            for row, (context, original_options) in zip(output[:, width:], batch, strict=True):
                text = self.tok_decode(row.tolist())
                for stop in stops:
                    text = text.split(stop, 1)[0]
                results.append(text)
                self.cache_hook.add_partial("generate_until", (context, original_options), text)
        return ordered.get_original(results)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the paper's zero-shot lm-eval suite.")
    parser.add_argument("--model", required=True, help="HF checkpoint directory or Hub ID.")
    parser.add_argument("--tokenizer", default="Qwen/Qwen3-0.6B-Base")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tasks", nargs="+", default=PAPER_TASKS)
    parser.add_argument("--include-path", type=Path, help="Directory of additional lm-eval task YAMLs.")
    parser.add_argument("--limit", type=int, help="Examples per task, for smoke tests only.")
    parser.add_argument("--max-length", type=int, default=MAX_EVAL_CONTEXT_TOKENS)
    parser.add_argument("--no-compile", action="store_true", help="Skip optional feedback compilation on CUDA.")
    parser.add_argument("--prefill-mode", choices=["cyclic", "autoregressive"])
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-fewshot", type=int, default=0)
    args = parser.parse_args()

    from lm_eval import simple_evaluate

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        configure_eval_compiler()
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    model = load_complete_model(args.model, dtype=dtype).to(device)
    if args.prefill_mode is not None:
        model.config.prefill_mode = args.prefill_mode
    if device.type == "cuda" and not args.no_compile:
        compile_feedback(model, mode="default")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    harness_model = WhiteMatterHarnessLM(
        model=model, tokenizer=tokenizer, batch_size=args.batch_size, max_length=args.max_length,
    )
    task_manager = None
    if args.include_path is not None:
        from lm_eval.tasks import TaskManager
        task_manager = TaskManager(include_path=str(args.include_path))
    result = simple_evaluate(
        model=harness_model,
        tasks=args.tasks,
        num_fewshot=args.num_fewshot,
        limit=args.limit,
        task_manager=task_manager,
        log_samples=False,
        cache_requests=False,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({
        "model": args.model, "tasks": args.tasks, "limit": args.limit,
        "num_fewshot": args.num_fewshot,
        "max_length": harness_model.max_length, "results": result["results"],
    }, indent=2, default=str))


if __name__ == "__main__":
    main()
