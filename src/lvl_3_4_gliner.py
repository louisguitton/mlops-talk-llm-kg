"""EL with DBPedia."""

from typing import Iterator

import argilla as rg
import spacy
from gliner_spacy.pipeline import (  # noqa: F401 because we need to register the factory with spacy
    GlinerSpacy,
)

from ner_talk.annotators import spacy_to_argilla
from ner_talk.argilla import dataset_to_records, template_for_token_classification
from ner_talk.datasets import ONTONOTES5_LABELS, load_ontonotes

rg.init(workspace="admin", api_url="http://localhost:6900", api_key="admin.apikey")

ontonotes = load_ontonotes()["validation"].select(range(100))

# Configuration for GLiNER integration
custom_spacy_config = {
    "gliner_model": "urchade/gliner_multi-v2.1",
    "chunk_size": 250,
    "labels": list(ONTONOTES5_LABELS.keys()),
    "style": "ent",
    "threshold": 0.3,
}

# Initialize a blank English spaCy pipeline and add GLiNER
nlp = spacy.blank("en")
nlp.add_pipe("gliner_spacy", config=custom_spacy_config)

records: Iterator[rg.FeedbackRecord] = dataset_to_records(
    ontonotes,
    responses_lambda=None,
    suggestions_lambda=lambda row: spacy_to_argilla(row, nlp),
)

dataset: rg.FeedbackDataset = template_for_token_classification(labels=ONTONOTES5_LABELS)

dataset.add_records(list(records))
dataset.push_to_argilla(name="ner-lvl3-gliner", workspace="admin")
