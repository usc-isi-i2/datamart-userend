import os
import typing
import numpy as np
import copy
import math
import hashlib
import json
import logging
from .find_identity import FindIdentity
from collections import Counter

try:
    from datamart_isi.config import default_temp_path
    DEFAULT_TEMP_PATH = default_temp_path
except:
    DEFAULT_TEMP_PATH = "/tmp"

_logger = logging.getLogger(__name__)


def produce(inputs, target_columns: typing.List[int]=None, target_p_nodes: typing.List[str]=None, input_type: str="pandas"):
    if input_type == "pandas":
        return produce_for_pandas(input_df=inputs, target_columns=target_columns, target_p_nodes=target_p_nodes)
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


def produce_for_pandas(input_df, target_columns: typing.List[int]=None, target_p_nodes: dict=None, threshold_for_converage=0.7):
    """
    function used to produce for input type is pandas.dataFrame
    :param input_df: input pd.dataFrame
    :param target_columns: target columns to find with wikidata
    :param target_p_nodes: user-speicified P node want to get, can be None if want automatic search
    :param threshold_for_converage: minimum coverage ratio for a wikidata columns to be appended
    :return: a pd.dataFrame with updated columns from wikidata
    """
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
        for each in input_df.iloc[:, column]:
            if type(each) is str:
                curData.append(each)
            elif each is np.nan or math.isnan(each):
                curData.append("")
            else:
                curData.append(str(each))

        # for each column, try to find corresponding specific P nodes required
        if target_p_nodes is not None and current_column_name in target_p_nodes:
            # if found, we should always create the extra column, even the coverage is less than threshold
            target_p_node_to_send = target_p_nodes[current_column_name]
            _logger.info("Specific P node given, will wikifier this column anyway.")
        else:
            # if not found, check if coverage reach the threshold
            target_p_node_to_send = None
            if coverage(curData) < threshold_for_converage:
                _logger.debug("Coverage of data is " + str(coverage(curData)) + " which is less than threshold " + str(
                    threshold_for_converage))
                continue

        # get wikifiered result for this column
        for idx, res in enumerate(FindIdentity.get_identifier_3(strings=curData, column_name=current_column_name,
                                                                target_p_node=target_p_node_to_send)):
            # res[0] is the send input P node
            top1_dict = res[1]
            new_col = [""] * len(curData)
            for i in range(len(curData)):
                if curData[i] in top1_dict:
                    new_col[i] = top1_dict[curData[i]]
            # same as previous, only check when no specific P nodes given
            if not target_p_node_to_send and coverage(new_col) < threshold_for_converage:
                _logger.debug("[WARNING] Coverage of Q nodes is " + str(coverage(new_col)) +
                              " which is less than threshold " + str(threshold_for_converage))
                continue
            column_to_p_node_dict[current_column_name] = res[0]
            col_name = current_column_name + '_wikidata'
            return_df[col_name] = new_col
            break

    save_specific_p_nodes(input_df, column_to_p_node_dict)
    return return_df


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
