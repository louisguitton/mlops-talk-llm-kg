from datasets import ClassLabel, DatasetDict, load_dataset

CONLL2003_LABELS = {
    "PER": "Person",
    "ORG": "Organization",
    "LOC": "Location",
    "MISC": "Other",
}


def load_conll() -> DatasetDict:
    classmap = ClassLabel(
        names=[
            "O",
            "B-PER",
            "I-PER",
            "B-ORG",
            "I-ORG",
            "B-LOC",
            "I-LOC",
            "B-MISC",
            "I-MISC",
        ]
    )
    return load_dataset("conll2003", trust_remote_code=True).map(
        lambda sample: {"parsed_ner_tags": classmap.int2str(sample["ner_tags"])}
    )
