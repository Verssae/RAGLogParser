from argparse import ArgumentParser
import os

from qdrant_client import QdrantClient, models
from dotenv import load_dotenv
from fastembed import TextEmbedding
from sklearn.metrics import pairwise_distances_argmin_min

import pandas as pd
import numpy as np
import openai

from utils import validate_template, load_project, init_qdrant, SEED


np.random.seed(SEED)

LLM_CALLS = 0
PROMPT_TOKENS = 0
COMPLETION_TOKENS = 0


def select_diverse_vectors(vectors, k=5):
    diverse_indices = [np.random.choice(range(len(vectors)))]
    for _ in range(k - 1):
        remaining_indices = list(set(range(len(vectors))) - set(diverse_indices))
        distances = pairwise_distances_argmin_min(
            vectors[remaining_indices], vectors[diverse_indices], metric="cosine"
        )[1]
        next_index = remaining_indices[np.argmax(distances)]
        diverse_indices.append(next_index)
    return diverse_indices

instruction = """As an expert in system log analysis, your task is to generate a semantic variable name for the given log messages and their corresponding templates.

### Instructions:
1. Replace all instances of `<*>` in the templates with `{variable_name}` that accurately reflects the semantics of the variable.
2. If a template consists of only static text (i.e., no variables `<*>`), leave it unchanged.
3. Output only the updated semantic template.

Be specific and ensure that the variable names are descriptive and contextually appropriate."""

def ask(logs, template):
    log_prompt = "\n".join([f"- {log}" for log in logs])
    print(log_prompt)
    completion = openai.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": instruction,
            },
            {
                "role": "user",
                "content": f"[logs]\n{log_prompt}\n\n[template]\n{template}",
            },
        ],
        seed=SEED,
        temperature=0.2,
    )
    return completion.choices[0].message.content


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "project",
        type=str,
        help="Mac,Android,Thunderbird,HealthApp,OpenStack,OpenSSH,Proxifier,HPC,Zookeeper,Hadoop,Linux,HDFS,BGL,Windows,Apache,Spark",
    )
    parser.add_argument(
        "--shot",
        type=int,
        help="Number of templates to see",
        default=4,
    )

    args = parser.parse_args()
    project = args.project

    load_dotenv()
    openai.api_key = os.getenv("OPENAI_API_KEY")

    client = QdrantClient("http://localhost:6333")
    embedding_model = TextEmbedding()
    client.set_model(embedding_model.model_name)
    ndim = list(
        filter(
            lambda x: x["model"] == embedding_model.model_name,
            embedding_model.list_supported_models(),
        )
    )[0]["dim"]

    assert init_qdrant(client, ndim, project, True) == True
    snapshots = client.list_snapshots(project)
    for snapshot in snapshots:
        client.delete_snapshot(project, snapshot.name)

    df = load_project(project)

    embedding_generator = embedding_model.embed(df.log.values)
    client.upload_collection(
        project,
        vectors=embedding_generator,
        ids=df.index,
        payload=df.to_dict(orient="records"),
    )

    templates = df.template.unique()
    print(f"Total templates: {len(templates)}")

    templates = np.random.choice(templates, args.shot, replace=False)
    print(f"Randomly selected templates: {len(templates)}")

    for template in templates:
        records, pointer = client.scroll(
            project,
            models.Filter(
                must=[models.FieldCondition(key="template", match=models.MatchValue(value=template))],
            ),
            with_vectors=True,
        )
        log_vectors = [log_vector.vector for log_vector in records]
        log_vectors = np.array(log_vectors)

        if len(log_vectors) < args.shot:
            logs = [record.payload["log"] for record in records]
        else:
            indicies = select_diverse_vectors(log_vectors, args.shot)
            logs = [records[i].payload["log"] for i in indicies]

        semantic_template = ask(logs, template)
        print(template)
        print(semantic_template)

        # Save the semantic template and 'seen' flag
        client.set_payload(
            project,
            payload={"seen": True, "semantic_template": semantic_template},
            points=models.Filter(
                must=[models.FieldCondition(key="template", match=models.MatchValue(value=template))]
            ),
        )

    snapshot = client.create_snapshot(project)
    print(f"Snapshot created: {snapshot.name}")

    # [using training data]
    # 1. diverse 3 logs for each template
    # 2. ask the expert to generate the semantic template
    # 3. save the semantic template
    # 4. use the semantic template to prompt the model
