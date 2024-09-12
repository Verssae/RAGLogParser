import pandas as pd
import scipy.special
from nltk.metrics.distance import edit_distance
from sklearn.metrics import accuracy_score
import numpy as np
import csv
import os
import argparse


def evaluate(predic_file):
    df_parsedlog = pd.read_csv(
        predic_file, index_col='LineId'
    )
    df_parsedlog["Predict_NoSpaces"] = df_parsedlog["Predicted"].str.replace(
        r"\s+", "", regex=True
    )
    df_parsedlog["EventTemplate_NoSpaces"] = df_parsedlog["EventTemplate"].str.replace(
        r"\s+", "", regex=True
    )
    accuracy_exact_string_matching = accuracy_score(
        np.array(df_parsedlog.EventTemplate_NoSpaces.values, dtype="str"),
        np.array(df_parsedlog.Predict_NoSpaces.values, dtype="str"),
    )
    edit_distance_result = []
    for i, j in zip(
        np.array(df_parsedlog.EventTemplate_NoSpaces.values, dtype="str"),
        np.array(df_parsedlog.Predict_NoSpaces.values, dtype="str"),
    ):
        edit_distance_result.append(edit_distance(i, j))

    edit_distance_result_mean = np.mean(edit_distance_result)
    edit_distance_result_std = np.std(edit_distance_result)
    (precision, recall, f_measure, accuracy_GA) = get_accuracy(
        df_parsedlog["EventTemplate_NoSpaces"], df_parsedlog["Predict_NoSpaces"]
    )
    return (
        accuracy_GA,
        accuracy_exact_string_matching,
        edit_distance_result_mean,
        edit_distance_result_std,
    )


def get_accuracy(series_groundtruth, series_parsedlog, debug=False):
    series_groundtruth_valuecounts = series_groundtruth.value_counts()
    real_pairs = 0
    for count in series_groundtruth_valuecounts:
        if count > 1:
            real_pairs += scipy.special.comb(count, 2)

    series_parsedlog_valuecounts = series_parsedlog.value_counts()
    parsed_pairs = 0
    for count in series_parsedlog_valuecounts:
        if count > 1:
            parsed_pairs += scipy.special.comb(count, 2)

    accurate_pairs = 0
    accurate_events = 0  # determine how many lines are correctly parsed
    for parsed_eventId in series_parsedlog_valuecounts.index:
        logIds = series_parsedlog[series_parsedlog == parsed_eventId].index
        series_groundtruth_logId_valuecounts = series_groundtruth[logIds].value_counts()
        error_eventIds = (
            parsed_eventId,
            series_groundtruth_logId_valuecounts.index.tolist(),
        )
        error = True
        if series_groundtruth_logId_valuecounts.size == 1:
            groundtruth_eventId = series_groundtruth_logId_valuecounts.index[0]
            if (
                logIds.size
                == series_groundtruth[series_groundtruth == groundtruth_eventId].size
            ):
                accurate_events += logIds.size
                error = False
        if error and debug:
            print(
                "(parsed_eventId, groundtruth_eventId) =",
                error_eventIds,
                "failed",
                logIds.size,
                "messages",
            )
        for count in series_groundtruth_logId_valuecounts:
            if count > 1:
                accurate_pairs += scipy.special.comb(count, 2)
    precision = float(accurate_pairs) / parsed_pairs
    recall = float(accurate_pairs) / real_pairs
    f_measure = 2 * precision * recall / (precision + recall)
    accuracy = float(accurate_events) / series_groundtruth.size
    return precision, recall, f_measure, accuracy

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("file_path", type=str, help="file path")
    args = parser.parse_args()

    GA, PA, ED, ED_std = evaluate(args.file_path)
    print(GA, PA, ED, ED_std)
    
