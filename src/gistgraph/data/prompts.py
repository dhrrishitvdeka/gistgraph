"""QA prompt construction.

The prompt is rendered with the model's chat template once, with a placeholder where the context
goes. Splitting on the placeholder gives a ``pre`` and a ``post`` string. For text contexts we
simply join ``pre + context + post``. For compressed contexts, the soft-prompt embeddings are
placed between the embeddings of ``pre`` and ``post``, so the LLM sees the same surrounding text
in both cases.
"""

from __future__ import annotations

_PLACEHOLDER = "@@CONTEXT@@"
_INSTRUCTION = (
    "Answer the question based on the context. Reply with only the answer, no explanation."
)


def _user_message(question: str) -> str:
    return f"{_INSTRUCTION}\n\nContext:\n{_PLACEHOLDER}\n\nQuestion: {question}"


def split_prompt(tokenizer, question: str) -> tuple[str, str]:
    """Return the text before and after the context slot for ``question``."""
    messages = [{"role": "user", "content": _user_message(question)}]
    rendered = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    pre, sep, post = rendered.partition(_PLACEHOLDER)
    if not sep:
        raise ValueError("chat template dropped the context placeholder")
    return pre, post


def full_prompt(tokenizer, context: str, question: str) -> str:
    pre, post = split_prompt(tokenizer, question)
    return pre + context + post
