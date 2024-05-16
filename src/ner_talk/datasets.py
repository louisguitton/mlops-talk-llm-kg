import collections

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


ONTONOTES5_LABELS = {
    "CARDINAL": "CARDINAL",  # Numerals that do not fall under another type
    "DATE": "DATE",  # Absolute or relative dates or periods
    "PERSON": "PERSON",  # People, including fictional
    "NORP": "NORP",  # Nationalities or religious or political groups
    "GPE": "GPE",  # Countries, cities, states
    "LAW": "LAW",  # Named documents made into laws
    "ORG": "ORGANIZATION",  # Companies, agencies, institutions, etc.
    "PERCENT": "PERCENT",  # Percentage (including “%”)
    "ORDINAL": "ORDINAL",  # “first”, “second”
    "MONEY": "MONEY",  # Monetary values, including unit
    "WORK_OF_ART": "WORK OF ART",  # Titles of books, songs, etc.
    "FAC": "FACILITY",  # Facilities like Buildings, airports, highways, bridges, etc.
    "TIME": "TIME",  # Times smaller than a day
    "LOC": "LOCATION",  # Non-GPE locations, mountain ranges, bodies of water
    "QUANTITY": "QUANTITY",  # Measurements, as of weight or distance
    "PRODUCT": "PRODUCT",  # Vehicles, weapons, foods, etc. (Not services)
    "EVENT": "EVENT",  # Named hurricanes, battles, wars, sports events, etc.
    "LANGUAGE": "LANGUAGE",  # Any named language
}


def load_ontonotes() -> DatasetDict:
    ontonotes5_labels_raw = {
        "O": 0,
        "B-CARDINAL": 1,
        "B-DATE": 2,
        "I-DATE": 3,
        "B-PERSON": 4,
        "I-PERSON": 5,
        "B-NORP": 6,
        "B-GPE": 7,
        "I-GPE": 8,
        "B-LAW": 9,
        "I-LAW": 10,
        "B-ORG": 11,
        "I-ORG": 12,
        "B-PERCENT": 13,
        "I-PERCENT": 14,
        "B-ORDINAL": 15,
        "B-MONEY": 16,
        "I-MONEY": 17,
        "B-WORK_OF_ART": 18,
        "I-WORK_OF_ART": 19,
        "B-FAC": 20,
        "B-TIME": 21,
        "I-CARDINAL": 22,
        "B-LOC": 23,
        "B-QUANTITY": 24,
        "I-QUANTITY": 25,
        "I-NORP": 26,
        "I-LOC": 27,
        "B-PRODUCT": 28,
        "I-TIME": 29,
        "B-EVENT": 30,
        "I-EVENT": 31,
        "I-FAC": 32,
        "B-LANGUAGE": 33,
        "I-PRODUCT": 34,
        "I-ORDINAL": 35,
        "I-LANGUAGE": 36,
    }
    ontonotes5_labels = collections.OrderedDict(
        sorted(ontonotes5_labels_raw.items(), key=lambda x: x[1])
    )
    classmap = ClassLabel(names=list(ontonotes5_labels.keys()))
    return (
        load_dataset("tner/ontonotes5")
        .rename_column("tags", "ner_tags")
        .map(lambda sample: {"parsed_ner_tags": classmap.int2str(sample["ner_tags"])})
    )


SPANMARKER_LABELS = {
    "PER": "person",  # People
    "ORG": "organization",  # Associations, companies, agencies, institutions, nationalities and religious or political groups
    "LOC": "location",  # Physical locations (e.g. mountains, bodies of water), geopolitical entities (e.g. cities, states), and facilities (e.g. bridges, buildings, airports).
    "ANIM": "animal",  # Breeds of dogs, cats and other animals, including their scientific names.
    "BIO": "biological",  # Genus of fungus, bacteria and protoctists, families of viruses, and other biological entities.
    "CEL": "celestial",  # Planets, stars, asteroids, comets, nebulae, galaxies and other astronomical objects.
    "DIS": "disease",  # Physical, mental, infectious, non-infectious, deficiency, inherited, degenerative, social and self-inflicted diseases.
    "EVE": "event",  # Sport events, battles, wars and other events.
    "FOOD": "food",  # Foods and drinks.
    "INST": "instrument",  # Technological instruments, mechanical instruments, musical instruments, and other tools.
    "MEDIA": "media",  # Titles of films, books, magazines, songs and albums, fictional characters and languages.
    "PLANT": "plant",  # Types of trees, flowers, and other plants, including their scientific names.
    "MYTH": "mythological",  # Mythological and religious entities.
    "TIME": "time",  # Specific and well-defined time intervals, such as eras, historical periods, centuries, years and important days. No months and days of the week.
    "VEHI": "vehicle",  # Cars, motorcycles and other vehicles.
}

DBPEDIA_LABELS = {"DBPEDIA_ENT": "DBpedia"}
