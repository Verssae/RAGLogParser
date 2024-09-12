import re

from qdrant_client import QdrantClient, models
import pandas as pd

SEED = 42
   
def to_tradional(template: str):
    pattern = r"{(.*?)}"
    replaced_string = re.sub(pattern, r"<*>", template)
    return replaced_string


def traditional_to_regex(template: str):
    template = re.escape(template)
    template = template.replace(r"<\*>", r"(.*?)")
    return template


def validate_template(log_message, template):
    template = to_tradional(template)
    template = traditional_to_regex(template)
    regex = re.compile(template)
    return regex.match(log_message) is not None


def init_qdrant(
    client: QdrantClient, ndim, collection_name="logs", from_scratch=True
) -> bool:

    existance = client.collection_exists(collection_name)
    if not existance:
        operation_info = client.create_collection(
            collection_name=collection_name,
            vectors_config=models.VectorParams(
                size=ndim, distance=models.Distance.COSINE
            ),
        )
        return operation_info

    if from_scratch and existance:
        client.delete_collection(collection_name)
        operation_info = client.create_collection(
            collection_name=collection_name,
            vectors_config=models.VectorParams(
                size=ndim, distance=models.Distance.COSINE
            ),
        )
        return operation_info

    return True


def records_to_df(records: list[models.Record]):
    df = pd.DataFrame([{**record.payload, "id": record.id} for record in records])
    df = df.set_index("id")
    return df


def load_project(project: str):
    df = pd.read_csv(
        f"logs/{project}/{project}_2k.log_structured_corrected.csv",
        index_col="LineId",
        usecols=["LineId", "Content", "EventTemplate"],
    )
    df = df.rename(columns={"Content": "log", "EventTemplate": "template"})
    df["seen"] = False
    df["semantic_template"] = None

    df["log"] = df["log"].apply(change_brackets)
    df["template"] = df["template"].apply(change_brackets)

    # Replace whitespace with a single space
    df["log"] = df["log"].str.replace(r"\s+", " ", regex=True)
    df["template"] = df["template"].str.replace(r"\s+", " ", regex=True)

    # strip
    df["log"] = df["log"].str.strip()
    df["template"] = df["template"].str.strip()

    return df


def change_brackets(text):
    # '{' -> '(' and '}' -> ')'
    return text.replace("{", "(").replace("}", ")")
