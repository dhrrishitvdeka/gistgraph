"""Small builders shared by several test modules."""

from gistgraph.data.schema import Example


def toy_examples(n=4):
    return [
        Example(
            id=f"e{i}",
            context=f"the answer is {'abcdefg'[i]} and more filler text goes here for length",
            question="what is the answer?",
            answers=["abcdefg"[i]],
            dataset="toy",
        )
        for i in range(n)
    ]
