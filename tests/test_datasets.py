from gistgraph.data.datasets import (
    fit_context,
    parse_2wiki,
    parse_hotpotqa,
    parse_longbench,
    parse_squad,
)
from gistgraph.data.prompts import full_prompt, split_prompt
from gistgraph.data.schema import Example

WIKI_CTX = {
    "title": ["A", "B", "C"],
    "sentences": [["a1.", "a2."], ["b1."], ["c1.", "c2."]],
}


def test_parse_hotpotqa():
    row = {
        "id": "h1",
        "question": "q?",
        "answer": "yes",
        "context": WIKI_CTX,
        "supporting_facts": {"title": ["A", "C"], "sent_id": [0, 1]},
    }
    ex = parse_hotpotqa(row)
    assert ex.paragraphs == ["A\na1. a2.", "B\nb1.", "C\nc1. c2."]
    assert ex.context == "\n\n".join(ex.paragraphs)
    assert ex.supporting == [0, 2] and ex.hops == 2 and ex.answers == ["yes"]


def test_parse_2wiki_hops_from_evidences():
    row = {
        "id": "w1",
        "question": "q?",
        "answer": "x",
        "context": WIKI_CTX,
        "supporting_facts": {"title": ["B"], "sent_id": [0]},
        "evidences": [["a", "r", "b"], ["b", "r", "c"], ["c", "r", "d"]],
    }
    ex = parse_2wiki(row)
    assert ex.hops == 3 and ex.supporting == [1]


def test_parse_squad_dedupes_answers():
    row = {
        "id": "s1",
        "context": "ctx text",
        "question": "q?",
        "answers": {"text": ["Denver", "Denver", "Broncos"], "answer_start": [0, 0, 1]},
    }
    ex = parse_squad(row)
    assert ex.answers == ["Denver", "Broncos"] and ex.hops == 1 and ex.supporting == [0]


def test_parse_longbench_splits_passages():
    row = {
        "_id": "l1",
        "input": "q?",
        "answers": ["x"],
        "context": "Passage 1:\nTitle one\nBody one.\n\nPassage 2:\nTitle two\nBody two.",
    }
    ex = parse_longbench(row, "longbench_hotpotqa")
    assert len(ex.paragraphs) == 2 and ex.paragraphs[0].startswith("Title one")


def test_fit_context_truncates_and_keeps_short(example, tok):
    short = fit_context(example, tok, 10_000)
    assert short is example
    cut = fit_context(example, tok, 5)
    assert len(tok.encode(cut.context)) == 5
    assert isinstance(cut, Example) and cut.supporting == []


class _ChatTok:
    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        return f"<u>{messages[0]['content']}</u><a>"


def test_split_prompt_places_context_between_pre_and_post():
    t = _ChatTok()
    pre, post = split_prompt(t, "Why?")
    assert pre.endswith("Context:\n") and post.startswith("\n\nQuestion: Why?")
    assert full_prompt(t, "CTX", "Why?") == pre + "CTX" + post
