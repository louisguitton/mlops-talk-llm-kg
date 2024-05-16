import types
from typing import Callable, Dict, Iterator, List, Tuple, Union

import argilla as rg
from argilla.client.feedback.schemas import SpanValueSchema
from datasets import (
    ClassLabel,
    Dataset,
    DatasetDict,
    Features,
    Sequence,
    Value,
    load_dataset,
)
from spacy.tokens import Doc
from spacy.training.iob_utils import biluo_tags_to_offsets, iob_to_biluo
from spacy.vocab import Vocab
from tqdm import tqdm


def template_for_token_classification(labels: Dict[str, str]) -> rg.FeedbackDataset:
    """Create a dataset with a span question for NER or POS tagging or information retrieval tasks.

    There is no pre-defined template in argilla yet, so we define a custom dataset instead.
    The high-level API of this method is TBD.
    ref: https://docs.argilla.io/en/latest/practical_guides/create_update_dataset/create_dataset.html#define-questions + click on Span
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


def iob_to_argilla(
    row: dict, tokens_field: str = "tokens", tags_field: str = "parsed_ner_tags"
) -> List[SpanValueSchema]:
    """Generate argilla compatible annotations from IOB tags.

    IOB stands for inside-outside-beginning and is a tagging format
    commonly used in research datasets.
    """
    doc = Doc(Vocab(), words=row[tokens_field])
    offsets = biluo_tags_to_offsets(doc, iob_to_biluo(row[tags_field]))

    return [
        SpanValueSchema(
            start=start,  # position of the first character of the span
            end=stop,  # position of the character right after the end of the span
            label=entity,
            score=1.0,
        )
        for start, stop, entity in offsets
    ]


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
