from argparse import ArgumentParser
import os
from typing import List
from dotenv import load_dotenv
from fastembed import TextEmbedding
import openai
import pandas as pd
from qdrant_client import QdrantClient, models
import numpy as np
from rich.progress import (
    Progress,
    BarColumn,
    DownloadColumn,
    TextColumn
)

from evaluate import evaluate
from utils import validate_template, to_tradional, init_qdrant, records_to_df, init_qdrant, SEED

LLM_CALLS = 0
PROMPT_TOKENS = 0
COMPLETION_TOKENS = 0
COST = lambda : PROMPT_TOKENS/1000000*0.15 + COMPLETION_TOKENS/1000000*0.06

np.random.seed(SEED)

progress = Progress(
    TextColumn("[bold blue]{task.fields[project]}", justify="right"),
    BarColumn(bar_width=None),
    "[progress.percentage]{task.percentage:>3.1f}%",
    "•",
    TextColumn("[bold yellow]{task.fields[status]}", justify="right"),
    "•",
    TextColumn("[bold cyan]{task.fields[cost]}", justify="right"),

)

def scroll(
    client: QdrantClient,
    project: str,
    must=[models.FieldCondition(key="seen", match=models.MatchValue(value=False))],
):
    result, pointer = client.scroll(
        project,
        scroll_filter=models.Filter(must=must),
        with_payload=True,
        with_vectors=True,
    )
    while pointer is not None:
        res, pointer = client.scroll(
            project,
            offset=pointer,
            scroll_filter=models.Filter(must=must),
            with_payload=True,
            with_vectors=True,
        )
        result += res

    return result


def gen_semantic_template(input_log, examples):
    instruction = """### Instructions:
You are an expert in system log analysis. Your task is to create a new log template corresponding to the input log message. 

1. Convert the input log into standardized templates by identifying and replacing the variable parts with {variable_name}.
2. If a template consists of only static text (i.e., no variables), you don't have to create a template.
3. Do not correct any typos or spacing errors in the log messages.
4. Do not change any static text, including special tokens such as ':', '='. This is necessary to create templates that accurately match the log messages.

### Context:
Your output should be a semantic template that accurately reflects the structure of the input log message.

### Example:
Input Log: "Error: User john_doe failed to login at 10:45 AM"
Output Template: "Error: User {username} failed to login at {time}"

### Your Task:
Generate the semantic template for the given input log message. Output only the semantic template."""

    examples, memory = examples[:2], examples[2:]
    example_prompt = (
        "\n\n[EXAMPLES]\n\n"
        + "\n\n".join(
            [f"[INPUT]\t{log}\n[OUTPUT]\t{template}" for log, template in examples]
        )
        + "\n\n[/EXAMPLES]\n\n"
    )
    # example_prompt = "\n\n".joint([f"Input Log: \"{log}\"\nOutput Template: \"{template}\"" for log, template in examples])
    variable_name = "{variable_name}"
    memory = [
        [
            {"role": "user", "content": f"[INPUT]\t{log}"},
            {"role": "system", "content": f"[OUTPUT]\t{template}"},
        ]
        for log, template in memory
    ]

    memory = [item for sublist in memory for item in sublist]

    completion = openai.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": f"You are an expert about system log analysis. Your task is to create a new log template corresponding to input log message.{example_prompt} Convert the input log into standardized templates by identifying and replacing the variable parts with {variable_name}. If a template consist of only static text (i.e, no variables), leave it unchanged. Do not correct any typos, or spacing errors in the log messages. Do not change any static text including special tokens such as ':', '='. This is necessary to create templates that accurately match the log messages. Output only the semantic template.",
            },
            *memory,
            {"role": "user", "content": f"[INPUT]\t{input_log}"},
        ],
        seed=SEED,
        temperature=0.2,
    )

    global LLM_CALLS, PROMPT_TOKENS, COMPLETION_TOKENS
    LLM_CALLS += 1
    PROMPT_TOKENS += completion.usage.prompt_tokens
    COMPLETION_TOKENS += completion.usage.completion_tokens

    output = completion.choices[0].message.content
    if output.startswith("[OUTPUT]"):
        output = output.split("\t")[1]

    return output


def find_similar_memory(client: QdrantClient, query_vector, project, k):
    if k == 0:
        return []
    result = client.search_groups(
        project,
        query_vector,
        query_filter=models.Filter(
            must=[
                models.FieldCondition(key="seen", match=models.MatchValue(value=False)),
                models.FieldCondition(key="valid", match=models.MatchValue(value=True)),
            ],
            must_not=[
                models.IsNullCondition(
                    is_null=models.PayloadField(key="semantic_template")
                )
            ],
        ),
        group_by="semantic_template",
        limit=k,
    )

    groups = result.groups

    return [[group.hits[0].payload["log"], group.id] for group in groups]


def pipeline(client:QdrantClient, records: List[models.Record], k, examples):
    with progress:
        task_id = progress.add_task(description="[cyan]Parsing...", total=len(records), start=True, project=f"{project}_{k}", status="[.]", cost=f"{COST():.4f}$")
        for record in records:
            input_log = record.payload["log"]

            # Get similar templates
            memory = find_similar_memory(client, record.vector, project, 4) 

            valid = False
            for _, template in memory:
                valid = validate_template(input_log, template)

                # If valid, update the record without LLM call
                if valid:
                    client.set_payload(
                        project,
                        payload={"valid": True, "semantic_template": template},
                        points=[record.id],
                    )
                    break

            if valid:
                progress.update(task_id, status="[R]", cost=f"{COST():.4f}$", advance=1)
                continue

            # Combine the similar templates and the examples
            np.random.shuffle(examples)
            history = memory[:k]
            history = history + examples[:k-len(history)]

            # reverse the order of history
            history = history[::-1]

            semantic_template = gen_semantic_template(input_log, history)
            valid = validate_template(input_log, semantic_template)
            if valid:
                client.set_payload(
                    project,
                    payload={"valid": True, "semantic_template": semantic_template},
                    points=[record.id],
                )
                progress.update(task_id, status="[+]", cost=f"{COST():.4f}$", advance=1)
                # print(f"[ADD] {input_log} -> {semantic_template}")
                progress.console.log(f"[green][+][/green] {input_log} -> [green]{semantic_template}[/green]")
                # Revalidate the similar unvalidated records
                unmatched_logs = client.search(
                    project,
                    record.vector,
                    models.Filter(
                        must=[
                            models.FieldCondition(
                                key="seen", match=models.MatchValue(value=False)
                            ),
                            models.FieldCondition(
                                key="valid", match=models.MatchValue(value=False)
                            ),
                        ]
                    ),
                    score_threshold=0.96,
                )
                if len(unmatched_logs) == 0:
                    continue
                progress.console.log(f"[Revalidating {len(unmatched_logs)} unmatched logs]")
                
                for unmatched_log in unmatched_logs:
                    progress.update(task_id, status=f"{len(unmatched_logs)}..", advance=0)
                    if validate_template(
                        unmatched_log.payload["log"], semantic_template
                    ):
                        client.set_payload(
                            project,
                            payload={
                                "valid": True,
                                "semantic_template": semantic_template,
                            },
                            points=[unmatched_log.id],
                        )
                        progress.console.log(
                            f"[cyan][R][/cyan] {unmatched_log.payload['log']} -> [cyan]{semantic_template}[/cyan]"
                        )
                        progress.update(task_id, status="[R]", cost=f"{COST():.4f}$", advance=1)

            else:
                client.set_payload(
                    project,
                    payload={"valid": False, "semantic_template": semantic_template},
                    points=[record.id],
                )
                progress.console.log(f"[red][X][/red] {input_log} -> [red]{semantic_template}[/red]")
                progress.update(task_id, status="[X]", cost=f"{COST():.4f}$", advance=1)



if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "project",
        type=str,
        help="Mac,Android,Thunderbird,HealthApp,OpenStack,OpenSSH,Proxifier,HPC,Zookeeper,Hadoop,Linux,HDFS,BGL,Windows,Apache,Spark",
    )
    parser.add_argument(
        "shot",
        type=int,
        help="Number of templates to see",
    )
    parser.add_argument(
        "--load",
        action="store_true",
        help="Continue from the last snapshot",
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

    snapshots = client.list_snapshots(project)
    assert len(snapshots) == 1, "Only one snapshot is allowed"
    ok = client.recover_snapshot(
        project,
        f"http://localhost:6333/collections/{project}/snapshots/{snapshots[0].name}",
        priority=models.SnapshotPriority.SNAPSHOT,
    )
    progress.console.log(f"Start from {snapshots[0].name}")

    # Main pipeline
    # unseen_logs = scroll(client, project)
    seen_records = scroll(
        client,
        project,
        must=[models.FieldCondition(key="seen", match=models.MatchValue(value=True))],
    )

    seen_df = records_to_df(seen_records)

    # group by semantic template and sample one log from each group
    examples = []
    for template in seen_df.semantic_template.unique():
        group = seen_df[seen_df.semantic_template == template]
        examples.append([group.log.iloc[0], template])

    # sample unseen logs
    unseen_logs = scroll(
        client,
        project,
        must=[
            models.FieldCondition(key="seen", match=models.MatchValue(value=False)),
        ],
    )

    progress.console.log(f"Unseen logs: {len(unseen_logs)}")


    # Online log parsing
    pipeline(client, unseen_logs, args.shot, examples)

    unmatched_logs = scroll(
        client,
        project,
        must=[
            models.FieldCondition(key="seen", match=models.MatchValue(value=False)),
            models.FieldCondition(key="valid", match=models.MatchValue(value=False)),
        ],
    )

    progress.console.log(f"Unmatched logs: {len(unmatched_logs)}")
    if len(unmatched_logs) > 0:
        progress.console.log(f"Second round of parsing")
        pipeline(client, unmatched_logs, args.shot, examples)

    unmatched_logs = scroll(
        client,
        project,
        must=[
            models.FieldCondition(key="seen", match=models.MatchValue(value=False)),
            models.FieldCondition(key="valid", match=models.MatchValue(value=False)),
        ],
    )
    progress.console.log(f"Unmatched logs: {len(unmatched_logs)}")
    # Evaluate the model
    unseen_logs = scroll(
        client,
        project,
        must=[models.FieldCondition(key="seen", match=models.MatchValue(value=False))],
    )

    unseen_df = records_to_df(unseen_logs)

    unseen_df["predict"] = unseen_df.apply(
        lambda x: to_tradional(x.semantic_template), axis=1
    )
    unseen_df.to_csv(f"output/{project}_{args.shot}.csv")

    GA, PA, ED, ED_std = evaluate(unseen_df)

    stats = {
        "project": project,
        "shot": args.shot,
        "LLM_CALLS": LLM_CALLS,
        "PROMPT_TOKENS": PROMPT_TOKENS,
        "COMPLETION_TOKENS": COMPLETION_TOKENS,
        "cost": COST(),
        "GA": GA,
        "PA": PA,
        "ED": ED,
        "ED_std": ED_std,
    }
    progress.console.log(stats)

    # save stats
    # if no csv file exists, create a new one
    if not os.path.exists("output/stats.csv"):
        stats_df = pd.DataFrame(columns=stats.keys())
        stats_df.to_csv("output/stats.csv", index=False)

    stats_df = pd.read_csv("output/stats.csv")
    stats_df = pd.concat([stats_df, pd.DataFrame([stats])], ignore_index=True, axis=0)
    stats_df.to_csv("output/stats.csv", index=False)


