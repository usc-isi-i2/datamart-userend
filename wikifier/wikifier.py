import os
import typing
import numpy as np
import copy
import math
import hashlib
import json
import logging
import requests
import pandas as pd
import io
import csv
from .find_identity import FindIdentity
from collections import Counter
from datamart_isi import config

try:
    from datamart_isi.config import default_temp_path
    DEFAULT_TEMP_PATH = default_temp_path
except:
    DEFAULT_TEMP_PATH = "/tmp"

_logger = logging.getLogger(__name__)
NEW_WIKIFIER_SERVER = config.new_wikifier_server



def produce(inputs, target_columns: typing.List[int]=None, target_p_nodes: typing.List[str]=None, input_type: str="pandas",
            wikifier_choice: typing.List[str]=None, threshold: float=0.7):
    if input_type == "pandas":
        if wikifier_choice is None:
            return produce_by_automatic(input_df=inputs, target_columns=target_columns,
                                        target_p_nodes=target_p_nodes, threshold_for_converage=threshold)
        elif target_columns is None:
            if wikifier_choice[0] == "identifier":
                return produce_for_pandas(inputs, target_columns, target_p_nodes, threshold)
            elif wikifier_choice[0] == "new_wikifier":
                return produce_by_new_wikifier(inputs, target_columns, threshold)
            else:
                return produce_by_automatic(inputs, target_columns, target_p_nodes, threshold)
        else:
            col_name = inputs.columns.tolist()
            col_idt, col_new, col_at = [], [], []
            for i in range(len(wikifier_choice)):
                if wikifier_choice[i] == "identifier":
                    col_idt.append(target_columns[i])
                elif wikifier_choice[i] == "new_wikifier":
                    col_new.append(target_columns[i])
                else:
                    col_at.append(target_columns[i])
            col_res = list(set([i for i in range(len(col_name))]).difference(set(col_new + col_idt + col_at)))
            return_df = copy.deepcopy(inputs.iloc[:, col_res])

            if col_idt:
                return_df_idt = produce_for_pandas(inputs.iloc[:, col_idt], [i for i in range(len(col_idt))], target_p_nodes, threshold)
                # change the column name index
                col_tmp = return_df_idt.columns.tolist()
                col_name.extend(list(set(col_tmp).difference(set(col_name).intersection(set(col_tmp)))))
                return_df = pd.concat([return_df, return_df_idt], axis=1)
            if col_new:
                return_df_new = produce_by_new_wikifier(inputs.iloc[:, col_new], [i for i in range(len(col_new))], threshold)
                col_tmp = return_df_new.columns.tolist()
                col_name.extend(list(set(col_tmp).difference(set(col_name).intersection(set(col_tmp)))))
                return_df = pd.concat([return_df, return_df_new], axis=1)
            if col_at:
                return_df_at = produce_by_automatic(inputs.iloc[:, col_at], [i for i in range(len(col_at))], target_p_nodes, threshold)
                col_tmp = return_df_at.columns.tolist()
                col_name.extend(list(set(col_tmp).difference(set(col_name).intersection(set(col_tmp)))))
                return_df = pd.concat([return_df, return_df_at], axis=1)

            return return_df[col_name]

    # elif input_type == "d3m_ds":
    #     return produce_for_d3m_dataset(input_ds=inputs, target_columns=target_columns)
    # elif input_type == "d3m_df":
    #     return produce_for_d3m_dataframe(input_df=inputs, target_columns=target_columns)
    else:
        raise ValueError("unknown type of input!")


def all_in_range_0_to_100(inputs):
    min_val = min(inputs)
    max_val = max(inputs)
    if min_val <= 100 and min_val >= 0 and max_val >= 0 and max_val <= 100:
        return True
    else:
        return False


def are_almost_continues_numbers(inputs, threshold=0.7):
    min_val = min(inputs)
    max_val = max(inputs)
    if (max_val - min_val) * threshold <= len(inputs):
        return True
    else:
        return False


def one_character_alphabet(inputs):
    if all(x.isalpha() and len(x) == 1 for x in inputs):
        return True
    return False


def produce_for_pandas(input_df, target_columns: typing.List[int]=None, target_p_nodes: dict=None, threshold_for_converage=0.7):
    """
    function used to produce for input type is pandas.dataFrame
    :param input_df: input pd.dataFrame
    :param target_columns: target columns to find with wikidata
    :param target_p_nodes: user-speicified P node want to get, can be None if want automatic search
    :param threshold_for_converage: minimum coverage ratio for a wikidata columns to be appended
    :return: a pd.dataFrame with updated columns from wikidata
    """
    _logger.debug("[INFO] Start to produce Q-nodes by identifier")
    # if no target columns given, just try every str columns
    if target_columns is None:
        target_columns = list(range(input_df.shape[1]))

    return_df = copy.deepcopy(input_df)
    column_to_p_node_dict = dict()
    for column in target_columns:
        current_column_name = input_df.columns[column]
        # curData = [str(x) for x in list(input_df[column])]
        _logger.debug('Current column: ' + current_column_name)
        try:
            temp = set()
            for each in input_df.iloc[:, column].dropna().tolist():
                temp.add(int(each))
            if all_in_range_0_to_100(temp) or are_almost_continues_numbers(temp, threshold_for_converage):
                _logger.debug("Column with all numerical values and useless detected, skipped")
                continue
        except:
            pass

        curData = []
        for each in input_df.iloc[:, column].tolist():
            if type(each) is str:
                curData.append(each)
            elif each is np.nan or math.isnan(each):
                curData.append("")
            else:
                curData.append(str(each))

        if coverage(curData) < threshold_for_converage:
            _logger.debug("Coverage of data is " + str(coverage(curData)) + " which is less than threshold " + str(threshold_for_converage))
            continue
        # for each column, try to find corresponding possible P nodes id first
        if target_p_nodes is not None and current_column_name in target_p_nodes:
            target_p_node_to_send = target_p_nodes[current_column_name]
        else:
            target_p_node_to_send = None

        for idx, res in enumerate(FindIdentity.get_identifier_3(strings=curData, column_name=current_column_name,
                                                                target_p_node=target_p_node_to_send)):
            # res[0] is the send input P node
            top1_dict = res[1]
            new_col = [""] * len(curData)
            for i in range(len(curData)):
                if curData[i] in top1_dict:
                    new_col[i] = top1_dict[curData[i]]
            if coverage(new_col) < threshold_for_converage:
                _logger.debug("[WARNING] Coverage of Q nodes is " + str(coverage(new_col)) +
                              " which is less than threshold " + str(threshold_for_converage))
                continue
            column_to_p_node_dict[current_column_name] = res[0]
            col_name = current_column_name + '_wikidata'
            return_df[col_name] = new_col
            break

    save_specific_p_nodes(input_df, column_to_p_node_dict)
    return return_df


def produce_by_new_wikifier(input_df, target_columns: typing.List[int]=None, threshold_for_converage = 0.7):
    _logger.debug("[INFO] Start to produce Q-nodes by new wikifier")
    return_df = copy.deepcopy(input_df)
    if target_columns is None:
        target_columns = list(range(input_df.shape[1]))

    col_names = []
    for column in target_columns:
        current_column_name = input_df.columns[column]
        _logger.debug('Current column: ' + current_column_name)
        col_names.append(current_column_name)

    # input_df.to_csv('wikifier.csv', index=False)
    data = io.BytesIO()
    data_bytes = input_df.to_csv(index=False).encode("utf-8")
    data.write(data_bytes)
    data.seek(0)

    files = {
        'file': ('wikifier.csv', data),
        'type': (None, 'text/csv'),
        'columns': (None, json.dumps({'names': col_names})),
        'wikifyPercentage': (None, str(threshold_for_converage))
    }

    response = requests.post(NEW_WIKIFIER_SERVER, files=files)
    if response.status_code == 200:
        data = response.content.decode("utf-8")
        data = list(csv.reader(data.splitlines(), delimiter=','))
        return_df = pd.DataFrame(data[1:], columns=data[0])
        _logger.debug("Successfully getting data from the new wikifier")
    else:
        _logger.debug('Error: ' + response.text)

    return return_df


def produce_by_automatic(input_df, target_columns: typing.List[int]=None, target_p_nodes: dict=None, threshold_for_converage = 0.7):
    _logger.debug("[INFO] Start to produce Q-nodes by automatic")
    if target_columns is None:
        target_columns = list(range(input_df.shape[1]))

    col_new, col_idt = [], []
    for column in target_columns:
        current_column_name = input_df.columns[column]
        _logger.debug('Current column: ' + current_column_name)
        try:
            if input_df.iloc[:, column].astype(float).dtypes == "float64" or input_df.iloc[:, column].astype(int).dtypes == "int64":
                _logger.debug(current_column_name + ' is numeric column, will choose identifier')
                col_idt.append(column)
        except:
            _logger.debug(current_column_name + ' is text column, will choose new wikifier')
            col_new.append(column)

    col_name = input_df.columns.tolist()
    return_df_idt = input_df.iloc[:, col_idt]
    return_df_new = input_df.iloc[:, col_new]
    col_res = list(set([i for i in range(len(col_name))]).difference(set(col_new + col_idt)))
    return_df = copy.deepcopy(input_df.iloc[:, col_res])

    if col_idt:
        return_df_idt = produce_for_pandas(return_df_idt, [i for i in range(len(col_idt))], target_p_nodes, threshold_for_converage)
        col_tmp = return_df_idt.columns.tolist()
        col_name.extend(list(set(col_tmp).difference(set(col_name).intersection(set(col_tmp)))))
    if col_new:
        return_df_new = produce_by_new_wikifier(return_df_new, [i for i in range(len(col_new))], threshold_for_converage)
        col_tmp = return_df_new.columns.tolist()
        col_name.extend(list(set(col_tmp).difference(set(col_name).intersection(set(col_tmp)))))
    return_df = pd.concat([return_df, return_df_idt, return_df_new], axis=1)

    return return_df[col_name]


def coverage(column):
    count_stats = Counter(column)
    return (len(column)-count_stats[''])/len(column)


def save_specific_p_nodes(original_dataframe, column_to_p_node_dict) -> bool:
    try:
        original_columns_list = original_dataframe.columns.tolist()
        original_columns_list.sort()
        hash_generator = hashlib.md5()

        hash_generator.update(str(original_columns_list).encode('utf-8'))
        hash_key = str(hash_generator.hexdigest())
        temp_path = os.getenv('D3MLOCALDIR', DEFAULT_TEMP_PATH)
        specific_q_nodes_file = os.path.join(temp_path, hash_key + "_column_to_P_nodes")
        if os.path.exists(specific_q_nodes_file):
            _logger.warning("The specific p nodes file already exist! Will replace the old one!")

        with open(specific_q_nodes_file, 'w') as f:
            json.dump(column_to_p_node_dict, f)

        return True

    except Exception as e:
        _logger.debug(e, exc_info=True)
        return False
