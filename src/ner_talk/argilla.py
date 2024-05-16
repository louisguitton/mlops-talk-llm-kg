from typing import Callable, Dict, Iterator, List

import argilla as rg
from argilla.client.feedback.schemas import SpanValueSchema
from datasets import Dataset
from tqdm import tqdm


def template_for_token_classification(labels: Dict[str, str]) -> rg.FeedbackDataset:
    """Create a dataset with a span question for NER or POS tagging or information retrieval tasks.

    There is no pre-defined template in argilla yet, so we define a custom dataset instead.
    The high-level API of this method is TBD.
    cf Argilla docs > "create update dataset" > "define questions" > "span"
    """
    dataset = rg.FeedbackDataset(
        fields=[rg.TextField(name="text")],
        questions=[
            rg.SpanQuestion(
                name="entities",
                title="Highlight the entities in the text:",
                labels=labels,
                field="text",  # the field where you want to do the span annotation
                required=True,
                allow_overlapping=True,
            )
        ],
    )
    return dataset


def dataset_to_records(
    dataset: Dataset,
    tokens_field: str = "tokens",
    responses_lambda: Callable[[dict], List[SpanValueSchema]] = None,
    suggestions_lambda: Callable[[dict], List[SpanValueSchema]] = None,
    suggestions_agent: str = None,
) -> Iterator[rg.FeedbackRecord]:
    row: dict
    for row in tqdm(dataset):
        # we assume the tokens are clean, and we disregard more tokenization details
        text = " ".join(row[tokens_field])

        yield rg.FeedbackRecord(
            fields={"text": text},
            responses=(
                [
                    {
                        "values": {
                            "entities": {
                                "value": responses_lambda(row),
                            }
                        }
                    }
                ]
                if responses_lambda
                else []
            ),
            suggestions=(
                [
                    {
                        "question_name": "entities",
                        "value": suggestions_lambda(row),
                        "agent": suggestions_agent,
                    }
                ]
                if suggestions_lambda
                else []
            ),
        )
