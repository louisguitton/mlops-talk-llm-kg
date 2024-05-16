from typing import Callable, List, Type

from argilla.client.feedback.schemas import SpanValueSchema
from spacy.language import Language
from spacy.tokens import Doc, Span
from spacy.training.iob_utils import biluo_tags_to_offsets, iob_to_biluo
from spacy.vocab import Vocab


def iob_to_argilla(
    row: dict, tokens_field: str = "tokens", tags_field: str = "parsed_ner_tags"
) -> List[SpanValueSchema]:
    """Generate argilla-compatible annotations from IOB tags.

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


def spacy_to_argilla(
    row: dict,
    nlp: Type[Language],
    tokens_field: str = "tokens",
    score: Callable[[Span], float] = None,
) -> List[SpanValueSchema]:
    """Generate argilla-compatible annotations from a spaCy NER model."""
    text = " ".join(row[tokens_field])
    doc = nlp(text)
    return [
        SpanValueSchema(
            start=ent.start_char,
            end=ent.end_char,
            label=ent.label_,
            score=score(ent),
        )
        for ent in doc.ents
    ]
