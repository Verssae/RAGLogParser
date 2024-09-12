from argparse import ArgumentParser
import os
import re
import uuid

from qdrant_client import QdrantClient
from qdrant_client.models import (
    VectorParams,
    Distance,
    Filter,
    FieldCondition,
    PointStruct,
    MatchValue,
    PointGroup,
)
from dotenv import load_dotenv
from fastembed import TextEmbedding
from sklearn.metrics import pairwise_distances_argmin_min

import pandas as pd
import numpy as np
import openai

from evaluator import evaluate

SEED = 42
np.random.seed(SEED)

llm_calls = 0
prompt_tokens = 0
completion_tokens = 0


def init_qdrant(
    client: QdrantClient, ndim, collection_name="logs", from_scratch=True
) -> bool:
    existance = client.collection_exists(collection_name)
    if not existance:
        operation_info = client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=ndim, distance=Distance.COSINE),
        )
        return operation_info

    if from_scratch and existance:
        client.delete_collection(collection_name)
        operation_info = client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=ndim, distance=Distance.COSINE),
        )
        return operation_info
    
    return True


def update_db(client: QdrantClient, id, vector, payload):
    client.upsert(
        collection_name="logs",
        points=[PointStruct(id=id, vector=vector, payload=payload)],
    )


def find_similar_templates(client, query_vector, project, k=5):
    result = client.search_groups(
        collection_name="logs",
        query_vector=query_vector,
        query_filter=Filter(
            must=[
                FieldCondition(key="project", match=MatchValue(value=project)),
                FieldCondition(key="validated", match=MatchValue(value=True)),
            ]
        ),
        group_by="template",
        limit=k,
    )
    groups = result.groups

    return [[group.hits[0].payload["message"], group.id] for group in groups]


def select_diverse_vectors(vectors, k=5):
    diverse_indices = [0]
    for _ in range(k - 1):
        remaining_indices = list(set(range(len(vectors))) - set(diverse_indices))
        distances = pairwise_distances_argmin_min(
            vectors[remaining_indices], vectors[diverse_indices], metric="cosine"
        )[1]
        next_index = remaining_indices[np.argmax(distances)]
        diverse_indices.append(next_index)
    return diverse_indices


def collect_diverse_messages(client: QdrantClient, template: str, project: str, k):
    records = client.scroll(
        collection_name="logs",
        scroll_filter=Filter(
            must=[
                FieldCondition(key="project", match=MatchValue(value=project)),
                FieldCondition(key="template", match=MatchValue(value=template)),
            ]
        ),
        with_vectors=True,
    )

    records = records[0]

    if len(records) < k:
        return []

    vectors = [record.vector for record in records]
    vectors = np.array(vectors)

    diverse_indices = select_diverse_vectors(vectors, k)
    messages = [records[i].payload["message"] for i in diverse_indices]

    return messages


def check_update(client: QdrantClient, template: str, project: str):
    global llm_calls, prompt_tokens, completion_tokens

    diverse_messages = collect_diverse_messages(client, template, project, k=5)
    if len(diverse_messages) == 0:
        return

    user_prompt = (
        "Log messages:\n"
        + "\n".join(diverse_messages)
        + "\n\nMatched Template:\n"
        + template
    )

    messages = [
        {
            "role": "system",
            "content": "You are an expert in analyzing system log messages and extracting patterns to create new log templates. Templates consist of static text and placeholders for dynamic values. Placeholder names are enclosed in curly braces with semantic meaning. Your task is to determine if the given template is needed to be updated. For example, same values for some variable may indicate it is static text part, or more meaningful name needed. If so, provide a new template that can be used to match log messages given as input. The new template should different with existing template. Otherwise, just say 'OK'. Do not explain the reason.  Output only the new template or OK.",
        },
        {"role": "user", "content": user_prompt},
    ]

    response = openai.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        seed=SEED,
        temperature=0,
    )

    llm_calls += 1
    prompt_tokens += response.usage.prompt_tokens
    completion_tokens += response.usage.completion_tokens

    output = response.choices[0].message.content
    if output == template:
        exit()
    if output != "OK":

        for message in diverse_messages:
            validated = validate_template(message, output)
            if not validated:
                print(f"[Update Fail] {output}")
                return

        print(f"[Update] {template} -> {output}")
        client.set_payload(
            collection_name="logs",
            payload={"template": output},
            points=Filter(
                must=[
                    FieldCondition(key="project", match=MatchValue(value=project)),
                    FieldCondition(key="template", match=MatchValue(value=template)),
                ]
            ),
        )


def gen_id(line_id, project):
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{line_id}-{project}"))


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


def load_project(project: str):
    df = pd.read_csv(
        f"logs/{project}/{project}_2k.log_structured_corrected.csv",
        index_col="LineId",
        usecols=["LineId", "Content", "EventTemplate"],
    )
    return df


def ask_openai(log_message, examples=[]):
    global llm_calls, prompt_tokens, completion_tokens

    history = [
        [
            {"role": "user", "content": example[0]},
            {"role": "assistant", "content": example[1]},
        ]
        for example in examples
    ]

    history = [item for sublist in history for item in sublist]

    messages = [
        {
            "role": "system",
            "content": "You are an expert in analyzing system log messages and extracting patterns to create new log templates. Templates consist of static parts and dynamic variable parts. Variable names are enclosed in curly braces with semantic meaning. Make the number of variables small as possible. Your output should be a template that can be used to match log messages given as input. Output should be a single line.",
        },
        *history,
        {"role": "user", "content": log_message},
    ]

    response = openai.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        seed=SEED,
        temperature=0,
    )

    llm_calls += 1
    prompt_tokens += response.usage.prompt_tokens
    completion_tokens += response.usage.completion_tokens

    return response.choices[0].message.content


def check_cross_update(client, log_message, query_vector):
    global llm_calls, prompt_tokens, completion_tokens
    result = client.search_groups(
        collection_name="logs",
        query_vector=query_vector,
        query_filter=Filter(
            must=[
                FieldCondition(key="project", match=MatchValue(value=project)),
                FieldCondition(key="validated", match=MatchValue(value=True)),
            ]
        ),
        group_by="template",
        limit=2,
    )
    groups = result.groups

    if len(groups) < 2:
        return

    if groups[1].hits[0].score < 0.98:
        return
    examples = [[group.hits[0].payload["message"], group.id] for group in groups]
    history = [
        [
            {"role": "user", "content": example[0]},
            {"role": "assistant", "content": example[1]},
        ]
        for example in examples
    ]

    history = [item for sublist in history for item in sublist]

    messages = [
        {
            "role": "system",
            "content": "You are an expert in analyzing system log messages and extracting patterns to create new log templates. Templates consist of static text and placeholders for dynamic values. Placeholder names are enclosed in curly braces with semantic meaning. Your output should be a template that can be used to match log messages given as input. Output should be a single line.",
        },
        *history,
        {
            "role": "user",
            "content": "Above two log messages are very similar but you've classified them into different templates. Please provide a new template that can be used to match both log messages. If you think they are correct, just say OK. Output should be a single line.",
        },
    ]

    response = openai.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        seed=SEED,
        temperature=0,
    )

    output = response.choices[0].message.content

    if output != "OK":
        for example in examples:
            validated = validate_template(example[0], output)
            if not validated:
                print(f"[Cross Update Fail] {output}")
                return

        print(f"[Cross Update] {examples[0][1]} + {examples[1][1]} -> {output}")
        client.set_payload(
            collection_name="logs",
            payload={"template": output},
            points=Filter(
                must=[
                    FieldCondition(key="project", match=MatchValue(value=project)),
                    FieldCondition(
                        key="template", match=MatchValue(value=examples[0][1])
                    ),
                ]
            ),
        )

        client.set_payload(
            collection_name="logs",
            payload={"template": output},
            points=Filter(
                must=[
                    FieldCondition(key="project", match=MatchValue(value=project)),
                    FieldCondition(
                        key="template", match=MatchValue(value=examples[0][1])
                    ),
                ]
            ),
        )

    llm_calls += 1
    prompt_tokens += response.usage.prompt_tokens
    completion_tokens += response.usage.completion_tokens


def pipeline(df, embedding_model):
    embeddings_generator = embedding_model.embed(df.Content.values)
    for i, (row, embedding) in enumerate(zip(df.itertuples(), embeddings_generator)):
        # if (i+1) % 200 == 0:
        #     for template in collect_templates(db, project):
        #         check_update(db, template, project)
        log_message = row.Content
        vector = embedding.tolist()
        id = gen_id(row.Index, project)
        validated = False

        similar_history = find_similar_templates(db, vector, project, k=5)
        if len(similar_history) > 0:
            for history in similar_history:

                template = history[1]
                validated = validate_template(log_message, template)
                if validated:
                    update_db(
                        db,
                        id,
                        vector,
                        {
                            "line_id": row.Index,
                            "project": project,
                            "message": log_message,
                            "validated": validated,
                            "template": template,
                        },
                    )
                    # print(f"\t[Old] {template}")
                    # check_cross_update(db, log_message, vector)
                    break

        if validated:
            continue

        new_template = ask_openai(log_message, examples=similar_history)
        validated = validate_template(log_message, new_template)
        update_db(
            db,
            id,
            vector,
            {
                "line_id": row.Index,
                "project": project,
                "message": log_message,
                "validated": validated,
                "template": new_template,
            },
        )

        if validated:
            print(f"[New] {new_template}")
            rematch_similar_messages(db, vector, new_template, project)
        else:
            print(f"[Fail] {new_template}")
            print(f"\t{id}")
            print(f"\t{log_message}")


def rematch_similar_messages(client: QdrantClient, query, template, project: str):
    result = client.search(
        collection_name="logs",
        query_vector=query,
        query_filter=Filter(
            must=[
                FieldCondition(key="project", match=MatchValue(value=project)),
                FieldCondition(key="validated", match=MatchValue(value=False)),
            ]
        ),
        score_threshold=0.98,
    )

    for scored_point in result:
        message = scored_point.payload["message"]
        if validate_template(message, template):
            print(f"[Re-matched]: {message}")
            client.set_payload(
                collection_name="logs",
                payload={"validated": True},
                points=Filter(
                    must=[
                        FieldCondition(key="project", match=MatchValue(value=project)),
                        FieldCondition(
                            key="line_id",
                            match=MatchValue(value=scored_point.payload["line_id"]),
                        ),
                    ]
                ),
            )


def unmatched_messages(client: QdrantClient, project: str):
    records, pointer = client.scroll(
        collection_name="logs",
        scroll_filter=Filter(
            must=[
                FieldCondition(key="project", match=MatchValue(value=project)),
                FieldCondition(key="validated", match=MatchValue(value=False)),
            ]
        ),
    )

    while pointer is not None:
        res, pointer = client.scroll(collection_name="logs", offset=pointer)
        records.extend(res)

    return [
        {
            "LineId": record.payload["line_id"],
            "Content": record.payload["message"],
            "EventTemplate": record.payload["template"],
        }
        for record in records
    ]


def collect(client: QdrantClient, project: str):
    records, pointer = client.scroll(
        collection_name="logs",
        scroll_filter=Filter(
            must=[
                FieldCondition(key="project", match=MatchValue(value=project)),
            ]
        ),
    )

    while pointer is not None:
        res, pointer = client.scroll(collection_name="logs", offset=pointer)
        records.extend(res)

    return [
        {
            "LineId": record.payload["line_id"],
            "Content": record.payload["message"],
            "Generated": record.payload["template"],
            "Predicted": to_tradional(record.payload["template"]),
            "Validated": record.payload["validated"],
        }
        for record in records
    ]


def collect_templates(client: QdrantClient, project: str):
    records, pointer = client.scroll(
        collection_name="logs",
        scroll_filter=Filter(
            must=[
                FieldCondition(key="project", match=MatchValue(value=project)),
            ]
        ),
    )

    while pointer is not None:
        res, pointer = client.scroll(collection_name="logs", offset=pointer)
        records.extend(res)

    return list({record.payload["template"] for record in records})


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "project",
        type=str,
        help="Mac,Android,Thunderbird,HealthApp,OpenStack,OpenSSH,Proxifier,HPC,Zookeeper,Hadoop,Linux,HDFS,BGL,Windows,Apache,Spark",
    )
    parser.add_argument("--test_size", type=float, default=0)
    args = parser.parse_args()

    load_dotenv()
    openai.api_key = os.getenv("OPENAI_API_KEY")

    db = QdrantClient("http://localhost:6333")
    embedding_model = TextEmbedding()
    db.set_model(embedding_model.model_name)
    ndim = list(
        filter(
            lambda x: x["model"] == embedding_model.model_name,
            embedding_model.list_supported_models(),
        )
    )[0]["dim"]

    init_qdrant(db, ndim)

    project = args.project
    df = load_project(project)

    print(f"Project: {project}")
    df = df.sample(frac=args.test_size, replace=False)
    print(f"Sample size: {len(df)}")

    pipeline(df, embedding_model)

    # unmatches = unmatched_messages(db, project)

    # if len(unmatches) > 0:
    #     unmatched_df = pd.DataFrame(
    #         unmatches, index=[record["LineId"] for record in unmatches]
    #     )
    #     pipeline(unmatched_df, embedding_model)

    print(
        f"LLM calls: {llm_calls} (expected cost: {prompt_tokens / 1000000 * 0.15 + completion_tokens / 1000000 * 0.6} USD)"
    )
    unmatches = unmatched_messages(db, project)
    print(f"Unmatched messages: {len(unmatches)}")
    print(unmatches)

    records = collect(db, project)
    records = pd.DataFrame(records)
    records.set_index("LineId", inplace=True)

    # combine with df
    records["EventTemplate"] = df["EventTemplate"]

    # Save
    records.to_csv(f"output/{project}_{args.test_size}.csv")
    GA, PA, ED, ED_std = evaluate(f"output/{project}_{args.test_size}.csv")
    print(GA, PA, ED, ED_std)
