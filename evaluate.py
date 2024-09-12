
from argparse import ArgumentParser

import pandas as pd
import numpy as np
import scipy
from sklearn.metrics import accuracy_score
from nltk.metrics.distance import edit_distance

from utils import validate_template


def evaluate(df):
    accuracy_exact_string_matching = accuracy_score(np.array(df['template'].values, dtype='str'),
                                                    np.array(df['predict'].values, dtype='str'))

    edit_distance_result = []
    for i, j in zip(np.array(df['template'].values, dtype='str'),
                    np.array(df['predict'].values, dtype='str')):
        edit_distance_result.append(edit_distance(i, j))

    edit_distance_result_mean = np.mean(edit_distance_result)
    edit_distance_result_std = np.std(edit_distance_result)

    (precision, recall, f_measure, accuracy_GA) = get_accuracy(df['template'], df['predict'])

    return accuracy_GA, accuracy_exact_string_matching, edit_distance_result_mean, edit_distance_result_std

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
        error_eventIds = (parsed_eventId, series_groundtruth_logId_valuecounts.index.tolist())
        error = True
        if series_groundtruth_logId_valuecounts.size == 1:
            groundtruth_eventId = series_groundtruth_logId_valuecounts.index[0]
            if logIds.size == series_groundtruth[series_groundtruth == groundtruth_eventId].size:
                accurate_events += logIds.size
                error = False
        if error and debug:
            print('(parsed_eventId, groundtruth_eventId) =', error_eventIds, 'failed', logIds.size, 'messages')
        for count in series_groundtruth_logId_valuecounts:
            if count > 1:
                accurate_pairs += scipy.special.comb(count, 2)

    precision = float(accurate_pairs) / parsed_pairs
    recall = float(accurate_pairs) / real_pairs
    f_measure = 2 * precision * recall / (precision + recall)
    accuracy = float(accurate_events) / series_groundtruth.size
    return precision, recall, f_measure, accuracy
    

    
if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "file_path",
        type=str,
    )

    args = parser.parse_args()

    df = pd.read_csv(args.file_path, index_col='id')


    # original template validation test
    df['original_valid'] = df[['log', 'template']].apply(lambda x: validate_template(x['log'], x['template']), axis=1)
    print(df['original_valid'].value_counts())
    falses = df[df['original_valid'] == False]
    print(falses[['log', 'template']])

    print(df['template'].isnull().sum())

    # predicted template validation test

    unseen_df = df[df['seen'] == False]

    GA, PA, ED, ED_std = evaluate(unseen_df)



    