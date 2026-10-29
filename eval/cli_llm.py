"""A LangChain chat model that runs a subscription CLI (Codex CLI, Claude Code) — the RAGAS judge without an API bill.

RAGAS only needs "prompt in, text out", so the judge LLM can be a command-line tool logged in with a ChatGPT or
Claude subscription. Everything else in the RAGAS run stays identical (same prompts, same metric maths, same input
and output files); only the model that answers the judge prompts changes:

    uv run python -m eval.rescore_ragas eval/results/<run>.json --judge-cli codex [--judge-cli-model gpt-5]
    uv run python -m eval.rescore_ragas eval/results/<run>.json --judge-cli claude [--judge-cli-model sonnet]

Each call runs in a fresh empty temporary directory with tools disabled / a read-only sandbox, so the judge sees
only the prompt — never the repository, the golden set or other runs. One process per call is slow (a few seconds
of start-up each); keep `--workers` low (2–4) to stay inside the subscription's rate limits.
"""

from __future__ import annotations

import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult

SYSTEM = ("You are an evaluation assistant. Follow the instructions in the prompt exactly and reply with only the "
          "requested output (JSON when JSON is asked for), no commentary.")


def _command(cli: str, model: str, out_file: Path) -> list[str]:
    if cli == "codex":
        exe = shutil.which("codex") or "codex"
        cmd = [exe, "exec", "--skip-git-repo-check", "--sandbox", "read-only", "--color", "never",
               "--output-last-message", str(out_file)]
        return cmd + (["--model", model] if model else []) + ["-"]  # "-": prompt from stdin
    if cli == "claude":
        exe = shutil.which("claude") or "claude"
        cmd = [exe, "-p", "--output-format", "text", "--tools", "", "--system-prompt", SYSTEM,
               "--setting-sources", "", "--strict-mcp-config"]
        return cmd + (["--model", model] if model else [])
    raise ValueError(f"unknown judge CLI {cli!r}; choose codex or claude")


def _strip_fences(text: str) -> str:
    m = re.fullmatch(r"\s*```(?:json)?\s*(.*?)\s*```\s*", text, flags=re.S)
    return m.group(1) if m else text.strip()


class CliChatModel(BaseChatModel):
    cli: str = "codex"
    model: str = ""
    timeout: int = 600

    @property
    def _llm_type(self) -> str:
        return f"cli-{self.cli}"

    def _call(self, prompt: str) -> str:
        with tempfile.TemporaryDirectory(prefix="ragas_judge_") as tmp:
            out_file = Path(tmp) / "last_message.txt"
            if self.cli == "codex":
                prompt = f"{SYSTEM}\n\n{prompt}"
            proc = subprocess.run(_command(self.cli, self.model, out_file), input=prompt, cwd=tmp,
                                  capture_output=True, text=True, encoding="utf-8", timeout=self.timeout)
            text = out_file.read_text(encoding="utf-8") if out_file.exists() else proc.stdout
            if proc.returncode != 0 or not text.strip():
                raise RuntimeError(f"{self.cli} exited {proc.returncode}: {(proc.stderr or proc.stdout)[-500:]}")
            return _strip_fences(text)

    def _generate(self, messages: list[BaseMessage], stop: list[str] | None = None, run_manager: Any = None,
                  **kwargs: Any) -> ChatResult:
        prompt = "\n\n".join(str(m.content) for m in messages)
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=self._call(prompt)))])
