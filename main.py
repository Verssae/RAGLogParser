# %% IMPORTS
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
)
from dotenv import load_dotenv
from fastembed import TextEmbedding

import pandas as pd
import numpy as np
import openai

SEED = 42
np.random.seed(SEED)

llm_calls = 0
prompt_tokens = 0
completion_tokens = 0

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# %% QDRANT
db = QdrantClient("http://localhost:6333")
embedding_model = TextEmbedding()
db.set_model(embedding_model.model_name)
ndim = list(
    filter(
        lambda x: x["model"] == embedding_model.model_name,
        embedding_model.list_supported_models(),
    )
)[0]["dim"]

assert db.collection_exists("test_collection")
if db.collection_exists("logs"):
    db.delete_collection("logs")

db.create_collection(
    collection_name="logs",
    vectors_config=VectorParams(size=ndim, distance=Distance.COSINE),
)


# %%  PROJECTS: Mac,Android,Thunderbird,HealthApp,OpenStack,OpenSSH,Proxifier,HPC,Zookeeper,Hadoop,Linux,HDFS,BGL,Windows,Apache,Spark
def load_project(project: str):
    df = pd.read_csv(
        f"logs/{project}/{project}_2k.log_structured_corrected.csv",
        index_col="LineId",
        usecols=["LineId", "Content", "EventTemplate"],
    )
    return df


# %%
new_topk_instruction = """
You are an expert in analyzing log messages and extracting patterns to create new log templates. Below are the log message and similar but not matched log templates with our input log messsage. Use these templates as a reference to create a new log template matches exactly with the input log message. The new template should use meaningful variable names in the placeholders.Answer should be a single line.

Example log templates:
{top_templates}

[INPUT] {log_message}
[OUTPUT] 
"""

new_diverse_instruction = """
Below are the log message and similar but not matched log templates with our input log messsage. Use these templates as a reference to create a new log template matches exactly with the input log message. The new template should use meaningful variable names in the placeholders. Answer should be a single line.

Simliar log templates:
{example_templates}

[INPUT] {log_message}
[OUTPUT] 
"""

update_instruction = """
You are an expert in analyzing log messages and extracting patterns to update log templates. Below are the input log message and the existing log template. Use the existing template as a reference to create an updated log template. The updated template should use meaningful variable names and should match both the input log message and the log messages previously matched by the existing template.

Input log message:
{log_message}

Existing log template:
{existing_template}

Log messages previously matched by the existing template:
{existing_matches}

Based on this information, please create an updated log template matches with input log message. Answer should be a single line.
"""
# %% Apache dataset is for reference
df_apache = load_project("Apache")
apache_templates = df_apache.EventTemplate.unique()


def sample_log_with_template(df: pd.DataFrame, template: str):
    return df[df.EventTemplate == template].sample(1).Content.values[0]


# for t in apache_templates[:3]:
#     print(t)
#     print(sample_log_with_template(df_apache, t))

# %% Manual labeling

example_templates = [
    "[INPUT] workerEnv.init() ok /etc/httpd/conf/workers2.properties" +
    "[OUTPUT] workerEnv.init() ok {file_path}",
    "[INPUT] mod_jk child workerEnv in error state 6",
    "[OUTPUT] mod_jk child workerEnv in error state {error_state}" +
    "[INPUT] jk2_init() Found child 5433 in scoreboard slot 9" +
    "[OUTPUT] k2_init() Found child {child_id} in scoreboard slot {slot_id}",
]


# %%
def generate_with_examples(log_message, example_templates):
    completion = openai.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "user",
                "content": new_diverse_instruction.format(
                    log_message=log_message,
                    example_templates="\n\n".join(example_templates),
                ),
            }
        ],
        temperature=0,
        seed=SEED,
    )

    global llm_calls, prompt_tokens, completion_tokens
    llm_calls += 1
    prompt_tokens += completion.usage.prompt_tokens
    completion_tokens += completion.usage.completion_tokens

    return completion.choices[0].message.content


def generate_with_similar_k(log_message, top_templates, k):
    completion = openai.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "user",
                "content": new_topk_instruction.format(
                    k=k,
                    log_message=log_message,
                    top_templates="\n".join(map(lambda x: '- ' + x, top_templates)),
                ),
            }
        ],
        temperature=0,
        seed=SEED,
    )
    
    global llm_calls, prompt_tokens, completion_tokens
    llm_calls += 1
    prompt_tokens += completion.usage.prompt_tokens
    completion_tokens += completion.usage.completion_tokens

    return completion.choices[0].message.content


# %%
project = "HDFS"
df = load_project(project)
# print(df.describe())
# templates = df.EventTemplate.unique()
# for t in templates:
#     print(t, end="\n\t")
#     print(sample_log_with_template(df, t))
# %% Online evaluation


def find_similar_templates(query_vector, project, k=5):
    result = db.search_groups(
        collection_name="logs",
        query_vector=query_vector,
        query_filter=Filter(
            must=[FieldCondition(key="project", match=MatchValue(value=project)), FieldCondition(key="validated", match=MatchValue(value=True))]
        ),
        group_by="template",
        limit=k,
    )
    groups = result.groups
    if len(groups) == 0:
        return []
    return [group.id for group in groups]


def gen_id(line_id, project):
    # make uuid from line_id and project
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{line_id}-{project}"))


def to_tradional(template: str):
    pattern = r"{(.*?)}"
    replaced_string = re.sub(pattern, r"<*>", template)
    return replaced_string


def traditional_to_regex(template: str):
    template = re.escape(template)
    template = template.replace(r"<\*>", r"(.*)")
    return template


def template_matcher(log_message, template):
    template = to_tradional(template)
    template = traditional_to_regex(template)
    regex = re.compile(template)
    return regex.match(log_message) is not None


def update_db(id, vector, payload):
    db.upsert(
        collection_name="logs",
        points=[PointStruct(id=id, vector=vector, payload=payload)],
    )


def pipeline(df, k=5):
    embeddings_generator = embedding_model.embed(df.Content.values)

    for row, embedding in zip(df.itertuples(), embeddings_generator):
        query_vector = embedding.tolist()
        print(f"Input: {row.Content}")
        similar_templates = find_similar_templates(query_vector, project, k)
        if len(similar_templates) == 0:
            new_template = generate_with_examples(row.Content, example_templates)
            validated = template_matcher(row.Content, new_template)

            update_db(
                id=gen_id(row.Index, project),
                vector=query_vector,
                payload={
                    "project": project,
                    "message": row.Content,
                    "validated": validated,
                    "template": new_template if validated else None,
                },
            )

            if validated:
                print(f"\t[E][New] {new_template}")
            else:
                print(f"\t[E][Fail] {new_template}")
        else:
            
            for template in similar_templates:
                validated = template_matcher(row.Content, template)
                if validated:
                    update_db(
                        id=gen_id(row.Index, project),
                        vector=query_vector,
                        payload={
                            "project": project,
                            "message": row.Content,
                            "validated": validated,
                            "template": template,
                        },
                    )
                    print(f"\t[S][Matched] {template}")
                    break
            if not validated:
                new_template = generate_with_similar_k(
                    row.Content, similar_templates, k
                )
                validated = template_matcher(row.Content, new_template)
                update_db(
                    id=gen_id(row.Index, project),
                    vector=query_vector,
                    payload={
                        "project": project,
                        "message": row.Content,
                        "validated": validated,
                        "template": new_template if validated else None,
                    },
                )
                if validated:
                    print(f"\t[S][New] {new_template}")
                else:
                    print(f"\t[S][Fail] {new_template}")


# %%
pipeline(df.sample(10))
print(f"LLM calls: {llm_calls} (expected cost: {prompt_tokens / 1000000 * 0.15 + completion_tokens / 1000000 * 0.6} USD)")

# %%
import re

def to_traditional(template: str):
    # Replace placeholders like {IP}, {Port} etc. with <*>
    pattern = r"{(.*?)}"
    replaced_string = re.sub(pattern, r"<*>", template)
    return replaced_string

def traditional_to_regex(template: str):
    # Escape all characters in the template string
    template = re.escape(template)
    # Replace escaped <*> with regex pattern (.*?) for non-greedy match
    template = template.replace(r"\<\*\>", r"(.*?)")
    return template

def template_matcher(log_message, template):
    # Convert template to traditional format
    traditional_template = to_traditional(template)
    # Convert traditional template to regex pattern
    regex_pattern = traditional_to_regex(traditional_template)
    # Compile the regex pattern
    regex = re.compile(regex_pattern)
    # Match the log message against the regex pattern
    match = regex.match(log_message)
    return match is not None

# Testing the functions
log_message = "10.251.126.22:50010:Got exception while serving blk_1686195200514944346 to /10.250.6.223:"
template = "{IP}:{Port}: Got exception while serving {BlockID} to {ClientIP}:{ClientPort}"

print(template_matcher(log_message, template))  # This should return True

import re

def to_traditional(template: str):
    # Replace placeholders like {IP}, {Port} etc. with <*>
    pattern = r"{(.*?)}"
    replaced_string = re.sub(pattern, r"<*>", template)
    print(f"Traditional Template: {replaced_string}")
    return replaced_string

def traditional_to_regex(template: str):
    # Escape all characters in the template string
    template = re.escape(template)
    # Replace escaped <*> with regex pattern (.*?) for non-greedy match
    template = template.replace(r"\<\*\>", r"(.*?)")
    print(f"Regex Template: {template}")
    return template

def template_matcher(log_message, template):
    # Convert template to traditional format
    traditional_template = to_traditional(template)
    # Convert traditional template to regex pattern
    regex_pattern = traditional_to_regex(traditional_template)
    # Compile the regex pattern
    regex = re.compile(regex_pattern)
    # Match the log message against the regex pattern
    match = regex.match(log_message)
    print(f"Matching Log Message: {log_message}")
    print(f"Regex Pattern: {regex_pattern}")
    print(f"Match Result: {match}")
    return match is not None

# Testing the functions
log_message = "10.251.126.22:50010:Got exception while serving blk_1686195200514944346 to /10.250.6.223:"
template = "{IP}:{Port}: Got exception while serving {BlockID} to {ClientIP}:{ClientPort}"

print(template_matcher(log_message, template))  # This should return True
# %%
