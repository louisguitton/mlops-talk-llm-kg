"""EL with DBPedia."""

from typing import Iterator

import argilla as rg
from spacy_llm.util import Config, assemble_from_config

from ner_talk.annotators import spacy_to_argilla
from ner_talk.argilla import dataset_to_records, template_for_token_classification
from ner_talk.datasets import ONTONOTES5_LABELS, load_ontonotes

rg.init(workspace="admin", api_url="http://localhost:6900", api_key="admin.apikey")

ontonotes = load_ontonotes()["validation"].select(range(100))

cfg_string = """
[nlp]
lang = "en"
pipeline = ["llm"]

[components]

[components.llm]
factory = "llm"

[components.llm.model]
@llm_models = "langchain.Ollama.v1"
name = "mistral"
context_length = 2048
config = {"temperature": 0.0}

[components.llm.task]
@llm_tasks = "spacy.NER.v3"
labels = ["CARDINAL", "DATE", "EVENT", "FAC", "GPE", "LANGUAGE", "LAW", "LOC", "MONEY", "NORP", "ORDINAL", "ORG", "PERCENT", "PERSON", "PRODUCT", "QUANTITY", "TIME", "WORK_OF_ART"]
description = Labelling named “real-world” objects, like persons, companies or locations.

[components.llm.task.label_definitions]
CARDINAL = "Numerals that do not fall under another type"
DATE = "Absolute or relative dates or periods"
EVENT = "Named hurricanes, battles, wars, sports events, etc."
FAC = "Buildings, airports, highways, bridges, etc."
GPE = "Countries, cities, states"
LANGUAGE = "Any named language"
LAW = "Named documents made into laws"
LOC = "Non-GPE locations, mountain ranges, bodies of water"
MONEY = "Monetary values, including unit"
NORP = "Nationalities or religious or political groups"
ORDINAL = "Ordinal number, i.e., first, second, etc."
ORG = "Companies, agencies, institutions, etc."
PERCENT = "Percentage (including “%”)"
PERSON = "People, including fictional"
PRODUCT = "Vehicles, weapons, foods, etc. (Not services)"
QUANTITY = "Measurements, as of weight or distance"
TIME = "Times smaller than a day"
WORK_OF_ART = "Titles of books, songs, etc."
"""
config = Config().from_str(cfg_string)
nlp = assemble_from_config(config)

records: Iterator[rg.FeedbackRecord] = dataset_to_records(
    ontonotes,
    responses_lambda=None,
    suggestions_lambda=lambda row: spacy_to_argilla(row, nlp),
)

dataset: rg.FeedbackDataset = template_for_token_classification(labels=ONTONOTES5_LABELS)

dataset.add_records(list(records))
dataset.push_to_argilla(name="ner-lvl3-llm", workspace="admin")
