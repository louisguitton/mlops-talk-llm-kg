"""SpanMarker"""

from typing import Iterator

import argilla as rg
import spacy

from ner_talk.annotators import spacy_to_argilla
from ner_talk.argilla import dataset_to_records, template_for_token_classification
from ner_talk.datasets import SPANMARKER_LABELS, load_ontonotes

rg.init(workspace="admin", api_url="http://localhost:6900", api_key="admin.apikey")

ontonotes = load_ontonotes()["validation"].select(range(100))

nlp = spacy.load("en_core_web_sm", exclude=["ner"])
nlp.add_pipe("span_marker", config={"model": "tomaarsen/span-marker-mbert-base-multinerd"})

records: Iterator[rg.FeedbackRecord] = dataset_to_records(
    ontonotes,
    responses_lambda=None,
    suggestions_lambda=lambda row: spacy_to_argilla(row, nlp),
)

dataset: rg.FeedbackDataset = template_for_token_classification(labels=SPANMARKER_LABELS)

dataset.add_records(list(records))
dataset.push_to_argilla(name="ner-lvl3", workspace="admin")
