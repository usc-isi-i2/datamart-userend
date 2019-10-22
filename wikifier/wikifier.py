import typing
import numpy as np
import copy
import math
import json
import logging
import requests
import pandas as pd
import io
import csv
from .find_identity import FindIdentity
from collections import Counter
from datamart_isi.cache.metadata_cache import MetadataCache
from datamart_isi.cache.general_search_cache import GeneralSearchCache
from datamart_isi.utilities import connection
from .utils import check_wikifier_choice

CACHE_MANAGER = GeneralSearchCache()
_logger = logging.getLogger(__name__)
NEW_WIKIFIER_SERVER = connection.get_wikifier_knowledge_graph_server_url()


def produce(inputs, target_columns: typing.List[int] = None, target_p_nodes: typing.List[str] = None, input_type: str = "pandas",
            wikifier_choice: typing.List[str] = None, threshold: float = 0.7, use_cache=True):
    if target_columns is None and target_p_nodes is None and wikifier_choice is None and use_cache:
        do_wikifier = check_wikifier_choice(inputs)
        if do_wikifier is False:
            _logger.info("Detect dataset which should not wikified!")
            return inputs

    if input_type == "pandas":
        # use general search cache system to cache the wikifier results
        produce_config = {"target_columns": target_columns, "target_p_nodes": target_p_nodes,
                          "input_type": input_type, "wikifier_choice": wikifier_choice, "threshold": threshold}

        if use_cache:
            cache_key = CACHE_MANAGER.get_hash_key(inputs, json.dumps(produce_config))
            _logger.debug("Current wikification's key is " + cache_key)
            cache_result = CACHE_MANAGER.get_cache_results(cache_key)
            # get the cache for wikified results directly
            if cache_result is not None:
                _logger.info("Using caching results for wikifier")
                return cache_result
            # END cache part
            else:
                _logger.debug("Cache not hitted.")
        else:
            _logger.debug("Not use cache for this time's wikification.")

        column_to_p_node_dict = dict()

        if wikifier_choice is None:
            return_df, column_to_p_node_dict_all = produce_by_automatic(inputs, target_columns, target_p_nodes, threshold)
            column_to_p_node_dict.update(column_to_p_node_dict_all)
        elif target_columns is None:
            if wikifier_choice[0] == "identifier":
                return_df, column_to_p_node_dict_identifier = produce_for_pandas(inputs, target_columns, target_p_nodes,
                                                                                 threshold)
                column_to_p_node_dict.update(column_to_p_node_dict_identifier)
            elif wikifier_choice[0] == "new_wikifier":
                return_df, column_to_p_node_dict_new_wikifier = produce_by_new_wikifier(inputs, target_columns, target_p_nodes,
                                                                                        threshold)
                column_to_p_node_dict.update(column_to_p_node_dict_new_wikifier)
            else:
                return_df, column_to_p_node_dict_all = produce_by_automatic(inputs, target_columns, target_p_nodes, threshold)
                column_to_p_node_dict.update(column_to_p_node_dict_all)
        else:
            col_name = inputs.columns.tolist()
            col_identifier, col_new_wikifier, col_auto = [], [], []
            for i in range(len(wikifier_choice)):
                if wikifier_choice[i] == "identifier":
                    col_identifier.append(target_columns[i])
                elif wikifier_choice[i] == "new_wikifier":
                    col_new_wikifier.append(target_columns[i])
                else:
                    col_auto.append(target_columns[i])
            col_res = list(set([i for i in range(len(col_name))]).difference(set(col_new_wikifier + col_identifier + col_auto)))
            return_df = copy.deepcopy(inputs.iloc[:, col_res])

            if col_identifier:
                return_df_identifier, column_to_p_node_dict_identifier = produce_for_pandas(inputs.iloc[:, col_identifier],
                                                                                            [i for i in
                                                                                             range(len(col_identifier))],
                                                                                            target_p_nodes, threshold)
                column_to_p_node_dict.update(column_to_p_node_dict_identifier)
                # change the column name index
                col_tmp = return_df_identifier.columns.tolist()
                col_name.extend(list(set(col_tmp).difference(set(col_name).intersection(set(col_tmp)))))
                return_df = pd.concat([return_df, return_df_identifier], axis=1)
            if col_new_wikifier:
                return_df_new, column_to_p_node_dict_new_wikifier = produce_by_new_wikifier(inputs.iloc[:, col_new_wikifier],
                                                                                            [i for i in
                                                                                             range(len(col_new_wikifier))],
                                                                                            target_p_nodes, threshold)
                column_to_p_node_dict.update(column_to_p_node_dict_new_wikifier)
                col_tmp = return_df_new.columns.tolist()
                col_name.extend(list(set(col_tmp).difference(set(col_name).intersection(set(col_tmp)))))
                return_df = pd.concat([return_df, return_df_new], axis=1)
            if col_auto:
                return_df_auto, column_to_p_node_dict_all = produce_by_automatic(inputs.iloc[:, col_auto],
                                                                                 [i for i in range(len(col_auto))],
                                                                                 target_p_nodes, threshold)
                column_to_p_node_dict.update(column_to_p_node_dict_all)
                col_tmp = return_df_auto.columns.tolist()
                col_name.extend(list(set(col_tmp).difference(set(col_name).intersection(set(col_tmp)))))
                return_df = pd.concat([return_df, return_df_auto], axis=1)

            return_df = return_df[col_name]

        # push the full dataset specific relationship here
        if column_to_p_node_dict:
            MetadataCache.save_specific_wikifier_targets(inputs, column_to_p_node_dict)

        if use_cache:
            # push to cache system
            response = CACHE_MANAGER.add_to_memcache(supplied_dataframe=inputs,
                                                     search_result_serialized=json.dumps(produce_config),
                                                     augment_results=return_df,
                                                     hash_key=cache_key
                                                     )
            if not response:
                _logger.warning("Push wikifier results to results failed!")
            else:
                _logger.info("Push wikifier results to memcache success!")

        return return_df

    # elif input_type == "d3m_ds":
    #     return produce_for_d3m_dataset(input_ds=inputs, target_columns=target_columns)
    # elif input_type == "d3m_df":
    #     return produce_for_d3m_dataframe(input_df=inputs, target_columns=target_columns)
    else:
        raise ValueError("unknown type of input!")


def all_in_range_0_to_100(inputs):
    min_val = min(inputs)
    max_val = max(inputs)
    if 100 >= min_val >= 0 and 0 <= max_val <= 100:
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


def produce_for_pandas(input_df, target_columns: typing.List[int] = None, target_p_nodes: dict = None,
                       threshold_for_coverage=0.7) -> typing.Tuple[pd.DataFrame, dict]:
    """
    function used to produce for input type is pandas.dataFrame
    :param input_df: input pd.dataFrame
    :param target_columns: target columns to find with wikidata
    :param target_p_nodes: user-specified P node want to get, can be None if want automatic search
    :param threshold_for_coverage: minimum coverage ratio for a wikidata columns to be appended
    :return: a pd.dataFrame with updated columns from wikidata
    """
    _logger.info("Start to produce Q-nodes by identifier")
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
            for each in input_df.iloc[:, column].dropna():
                temp.add(float(each))
            if all_in_range_0_to_100(temp) or are_almost_continues_numbers(temp, threshold_for_coverage):
                _logger.debug("Column with all numerical values and useless detected, skipped")
                continue
        except:
            pass

        try:
            # for special case that if a column has only one character for each row, we should skip it
            temp = set(input_df.iloc[:, column].dropna())
            if one_character_alphabet(temp):
                _logger.debug("Column with only one letter in each line and useless detected, skipped")
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
            if coverage(curData) < threshold_for_coverage:
                _logger.debug("Coverage of data is " + str(coverage(curData)) + " which is less than threshold " + str(
                    threshold_for_coverage))
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
            if not target_p_node_to_send and coverage(new_col) < threshold_for_coverage:
                _logger.debug("[WARNING] Coverage of Q nodes is " + str(coverage(new_col)) +
                              " which is less than threshold " + str(threshold_for_coverage))
                continue
            column_to_p_node_dict[current_column_name] = res[0]
            col_name = current_column_name + '_wikidata'
            return_df[col_name] = new_col
            break

    if column_to_p_node_dict:
        MetadataCache.save_specific_wikifier_targets(input_df, column_to_p_node_dict)

    return return_df, column_to_p_node_dict


def produce_by_new_wikifier(input_df, target_columns=None, target_p_nodes: dict = None,
                            threshold_for_coverage=0.7) -> typing.Tuple[pd.DataFrame, dict]:
    """
    The function used to call new wikifier service
    :param input_df: a dataframe(both d3m or pandas are acceptable)
    :param target_columns: typing.List[int] indicates the column numbers of the columns need to be wikified
    :param target_p_nodes: user-speicified Q node want to get, can be None if want automatic search
    :param threshold_for_coverage: the minimum coverage of Q nodes for the column,
                                   if the appeared times are lower than threshold, we will not use it
    :return: a dataframe with wikifiered columns
    """
    _logger.info("Start running new wikifier")
    if target_columns is None:
        target_columns = list(range(input_df.shape[1]))
    col_names = []
    column_to_p_node_dict = dict()

    for column in target_columns:
        current_column_name = input_df.columns[column]
        # for special case that if a column has only one character for each row, we should skip it
        temp = set(input_df.iloc[:, column].dropna())
        if one_character_alphabet(temp):
            _logger.debug("Column with only one letter in each line and useless detected, skipped")
            continue
        else:
            col_names.append(current_column_name)

    if not col_names:
        return input_df, column_to_p_node_dict

    _logger.debug("Following {} columns will be send to new wikifier:".format(str(len(col_names))))
    _logger.debug(str(col_names))

    # transfer to bytes and send to new wikifier server
    data = io.BytesIO()
    data_bytes = input_df.to_csv(index=False).encode("utf-8")
    data.write(data_bytes)
    data.seek(0)

    files = {
        'file': ('wikifier.csv', data),
        'type': (None, 'text/csv'),
        'columns': (None, json.dumps({'names': col_names})),
        'wikifyPercentage': (None, str(threshold_for_coverage))
    }

    if target_p_nodes is not None:
        target_q_nodes_list = []
        for name in col_names:
            if name in target_p_nodes.keys():
                _logger.info("Specific Q node(class) given, will wikifier this column by this class.")
                target_q_nodes_list.append(target_p_nodes[name])
            else:
                target_q_nodes_list.append("")
        files['phase'] = (None, 'test')
        files['columnClass'] = (None, json.dumps({'names': target_q_nodes_list}))

    response = requests.post(NEW_WIKIFIER_SERVER, files=files)
    if response.status_code == 200:
        content = response.content.decode("utf-8")
        content = json.loads(content)
        data, column_to_p_node_dict = content['data'], content['class']
        data = list(csv.reader(data.splitlines(), delimiter=','))
        return_df = pd.DataFrame(data[1:], columns=data[0])
        col_name = return_df.columns.tolist()
        for cn in col_name:
            if "_WK" in cn and cn not in input_df.columns:
                new_name = cn[:-2] + "wikidata"
                return_df.rename(columns={cn: new_name}, inplace=True)
        _logger.debug("Get data from the new wikifier successfully.")
        if column_to_p_node_dict:
            _logger.info("For each column, the best matching class is:" + str(column_to_p_node_dict))
            MetadataCache.save_specific_wikifier_targets(input_df, column_to_p_node_dict)
    else:
        _logger.error('Something wrong in new wikifier server with response code: ' + response.text)
        _logger.debug("Wikifier_choice will change to identifier")
        return_df, column_to_p_node_dict_identifier = produce_for_pandas(input_df=input_df, target_columns=target_columns,
                                                                         threshold_for_coverage=0.7)
        column_to_p_node_dict.update(column_to_p_node_dict_identifier)
    return return_df, column_to_p_node_dict


def produce_by_automatic(input_df, target_columns=None, target_p_nodes=None, threshold_for_coverage=0.7) \
        -> typing.Tuple[pd.DataFrame, dict]:
    """
    The function used to call new wikifier service
    :param input_df: a dataframe(both d3m or pandas are acceptable)
    :param target_columns: typing.List[int] indicates the column numbers of the columns need to be wikified
    :param threshold_for_coverage: the minimum coverage of Q nodes for the column,
                                   if the appeared times are lower than threshold, we will not use it
    :param target_p_nodes: a dict which includes the specific p node for the given column
    :return: a dataframe with wikifiered columns
    """
    _logger.debug("Start running automatic wikifier")
    column_to_p_node_dict = dict()
    if target_columns is None:
        target_columns = list(range(input_df.shape[1]))
    col_new_wikifier, col_identifier = [], []
    for column in target_columns:
        current_column_name = input_df.columns[column]
        # _logger.debug('Current column: ' + current_column_name)
        if target_p_nodes is not None and current_column_name in target_p_nodes.keys():
            if "Q" in target_p_nodes[current_column_name]:
                col_new_wikifier.append(column)
                # _logger.debug(current_column_name + ' is text column, will choose new wikifier')
            elif "P" in target_p_nodes[current_column_name]:
                col_identifier.append(column)
                # _logger.debug(current_column_name + ' is numeric column, will choose identifier')
        else:
            try:
                if input_df.iloc[:, column].astype(float).dtypes == "float64" or input_df.iloc[:, column].astype(
                        int).dtypes == "int64":
                    # _logger.debug(current_column_name + ' is numeric column, will choose identifier')
                    col_identifier.append(column)
            except:
                # _logger.debug(current_column_name + ' is text column, will choose new wikifier')
                col_new_wikifier.append(column)

    if col_identifier:
        _logger.info("Following columns will be sent to identifier:")
        _logger.info(str(col_identifier))
    if col_new_wikifier:
        _logger.info("Following columns will be sent to new wikifier:")
        _logger.info(str(col_new_wikifier))
    if not col_new_wikifier and not col_identifier:
        _logger.info("No column will be wikified!")

    col_name = input_df.columns.tolist()
    return_df_identifier = input_df.iloc[:, col_identifier]
    return_df_new = input_df.iloc[:, col_new_wikifier]
    col_res = list(set([i for i in range(len(col_name))]).difference(set(col_new_wikifier + col_identifier)))
    return_df = copy.deepcopy(input_df.iloc[:, col_res])

    if col_identifier:
        return_df_identifier, column_to_p_node_dict_identifier = produce_for_pandas(return_df_identifier,
                                                                                    [i for i in range(len(col_identifier))],
                                                                                    target_p_nodes, threshold_for_coverage)
        column_to_p_node_dict.update(column_to_p_node_dict_identifier)
        col_tmp = return_df_identifier.columns.tolist()
        # change the column name index
        col_name.extend(list(set(col_tmp).difference(set(col_name).intersection(set(col_tmp)))))

    if col_new_wikifier:
        return_df_new, column_to_p_node_dict_new_wikifier = produce_by_new_wikifier(return_df_new,
                                                                                    [i for i in range(len(col_new_wikifier))],
                                                                                    target_p_nodes, threshold_for_coverage)
        column_to_p_node_dict.update(column_to_p_node_dict_new_wikifier)
        col_tmp = return_df_new.columns.tolist()
        col_name.extend(list(set(col_tmp).difference(set(col_name).intersection(set(col_tmp)))))
    return_df = pd.concat([return_df, return_df_identifier, return_df_new], axis=1)

    return return_df[col_name], column_to_p_node_dict


def coverage(column):
    count_stats = Counter(column)
    return (len(column) - count_stats['']) / len(column)
