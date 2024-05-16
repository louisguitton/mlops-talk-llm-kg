"""Add suggestions from a simple spacy pipeline and responses from gold labels."""

from typing import Iterator

import argilla as rg
import spacy

from ner_talk.annotators import iob_to_argilla, spacy_to_argilla
from ner_talk.argilla import dataset_to_records, template_for_token_classification
from ner_talk.datasets import ONTONOTES5_LABELS, load_ontonotes

rg.init(workspace="admin", api_url="http://localhost:6900", api_key="admin.apikey")

ontonotes = load_ontonotes()["validation"].select(range(100))

nlp = spacy.load("en_core_web_sm")

records: Iterator[rg.FeedbackRecord] = dataset_to_records(
    ontonotes,
    responses_lambda=iob_to_argilla,
    suggestions_lambda=lambda row: spacy_to_argilla(row, nlp),
)

dataset: rg.FeedbackDataset = template_for_token_classification(labels=ONTONOTES5_LABELS)

dataset.add_records(list(records))
dataset.push_to_argilla(name="ner-lvl2", workspace="admin")
