"""Load a NER research dataset and upload it to Argilla for human annotation."""

import argilla as rg

from ner_talk.argilla import dataset_to_records, template_for_token_classification
from ner_talk.datasets import CONLL2003_LABELS, load_conll

rg.init(workspace="admin", api_url="http://localhost:6900", api_key="admin.apikey")

conll2003 = load_conll()["validation"].select(range(100))
records = dataset_to_records(conll2003, responses_lambda=None, suggestions_lambda=None)

dataset = template_for_token_classification(labels=CONLL2003_LABELS)
dataset.add_records(list(records))
dataset.push_to_argilla(name="ner-lvl1", workspace="admin")
