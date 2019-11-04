import wikifier
import pandas as pd
import copy
import os
import typing
import re
import frozendict
import logging
import json
import hashlib
from d3m.base import utils as d3m_utils
from d3m.container import Dataset as d3m_Dataset
from d3m.container import DataFrame as d3m_DataFrame
from d3m.metadata.base import ALL_ELEMENTS
from datamart_isi.cache.metadata_cache import MetadataCache
from datamart_isi import config
from os import path
from pandas.util import hash_pandas_object

Q_NODE_SEMANTIC_TYPE = config.q_node_semantic_type
DEFAULT_TEMP_PATH = config.default_temp_path
_logger = logging.getLogger(__name__)


def run_wikifier(supplied_data: d3m_Dataset, use_cache=True):
    # the augmented dataframe should not run wikifier again to ensure the semantic type is correct
    # TODO: In this way, we will not search on augmented columns if run second time of wikifier
    exist_q_nodes, supplied_data = check_and_correct_q_nodes_semantic_type(supplied_data)
    if exist_q_nodes:
        _logger.warning("The input dataset already have Q nodes, will not run wikifier again!")
        return supplied_data

    try:
        output_ds = copy.copy(supplied_data)
        need_column_type = config.need_wikifier_column_type_list
        res_id, supplied_dataframe = d3m_utils.get_tabular_resource(dataset=supplied_data, resource_id=None)
        specific_p_nodes = MetadataCache.get_specific_p_nodes(supplied_dataframe)
        if use_cache and specific_p_nodes is not None:
            target_columns = list()
            _logger.info("Get specific column<->p_nodes relationship from previous TRAIN run. Will only wikifier those columns!")
            _logger.info(str(specific_p_nodes))
            for i, each_column_name in enumerate(supplied_dataframe.columns.tolist()):
                if each_column_name in specific_p_nodes:
                    # double check whether this column should be wikified
                    each_column_semantic_type = supplied_data.metadata.query((res_id, ALL_ELEMENTS, i))['semantic_types']
                    _logger.debug("column No." + str(i) + "'s semantic type is " + str(each_column_semantic_type))
                    skip_column_type = config.skip_wikifier_column_type_list
                    if set(each_column_semantic_type).intersection(skip_column_type):
                        _logger.warning("Detect the column semantic type should not be wikified on column No." + str(i) +
                                        ": " + each_column_name + "! Will remove this column from wikifier target columns")
                        continue
                    else:
                        target_columns.append(i)

        else:
            # if specific p nodes not given, try to find possible candidate p nodes columns
            target_columns = list(range(supplied_dataframe.shape[1]))
            temp = copy.deepcopy(target_columns)

            skip_column_type = set()
            # if we detect some special type of semantic type (like PrimaryKey here), it means some metadata is adapted
            # from exist dataset but not all auto-generated, so we can have more restricts
            for each in target_columns:
                each_column_semantic_type = supplied_data.metadata.query((res_id, ALL_ELEMENTS, each))['semantic_types']
                if "https://metadata.datadrivendiscovery.org/types/PrimaryKey" in each_column_semantic_type:
                    skip_column_type = config.skip_wikifier_column_type_list
                    break

            for each in target_columns:
                each_column_semantic_type = supplied_data.metadata.query((res_id, ALL_ELEMENTS, each))['semantic_types']
                # if the column type inside here found, this column should be wikified
                if set(each_column_semantic_type).intersection(need_column_type):
                    continue
                # if the column type inside here found, this column should not be wikified
                elif set(each_column_semantic_type).intersection(skip_column_type):
                    temp.remove(each)
                elif supplied_dataframe.columns[each] == "d3mIndex":
                    temp.remove(each)
            target_columns = temp

        if len(target_columns) == 0:
            _logger.info("No columns found need to be wikified!")
            return supplied_data

        _logger.debug("The target columns need to be wikified are: " + str(target_columns))
        # here because this function is called from augment part, so this part
        wikifier_res = wikifier.produce(inputs=pd.DataFrame(supplied_dataframe), target_columns=target_columns,
                                        target_p_nodes=specific_p_nodes, use_cache=use_cache)
        output_ds[res_id] = d3m_DataFrame(wikifier_res, generate_metadata=False)
        # update metadata on column length
        selector = (res_id, ALL_ELEMENTS)
        old_meta = dict(output_ds.metadata.query(selector))
        old_meta_dimension = dict(old_meta['dimension'])
        old_column_length = old_meta_dimension['length']
        old_meta_dimension['length'] = wikifier_res.shape[1]
        old_meta['dimension'] = frozendict.FrozenOrderedDict(old_meta_dimension)
        new_meta = frozendict.FrozenOrderedDict(old_meta)
        output_ds.metadata = output_ds.metadata.update(selector, new_meta)

        # update each column's metadata
        for i in range(old_column_length, wikifier_res.shape[1]):
            selector = (res_id, ALL_ELEMENTS, i)
            metadata = {"name": wikifier_res.columns[i],
                        "structural_type": str,
                        'semantic_types': (
                            "http://schema.org/Text",
                            "https://metadata.datadrivendiscovery.org/types/Attribute",
                            Q_NODE_SEMANTIC_TYPE
                        )}
            output_ds.metadata = output_ds.metadata.update(selector, metadata)
        return output_ds

    except Exception as e:
        _logger.error("Wikifier running failed!!!")
        _logger.debug(e, exc_info=True)
        return supplied_data


def check_and_correct_q_nodes_semantic_type(input):
    """
    Function used to detect whether a dataset or a dataframe already contains q nodes columns or not
    Usually, we should not run wikifier again if there already exist q nodes
    :param input:
    :return:
    """
    find_q_node_columns = False
    if type(input) is d3m_Dataset:
        input_type = "ds"
        res_id, input_dataframe = d3m_utils.get_tabular_resource(dataset=input, resource_id=None)
    elif type(input) is d3m_DataFrame:
        input_type = "df"
        input_dataframe = input
    else:
        _logger.error("Wrong type of input as :" + str(type(input)))
        return False

    for i in range(input_dataframe.shape[1]):
        if input_type == "ds":
            selector = (res_id, ALL_ELEMENTS, i)
        elif input_type == "df":
            selector = (ALL_ELEMENTS, i)

        each_metadata = input.metadata.query(selector)
        if Q_NODE_SEMANTIC_TYPE in each_metadata['semantic_types']:
            _logger.debug("Q nodes semantic type found in column No.{}, will not run wikifier.".format(str(i)))
            find_q_node_columns = True

        elif 'http://schema.org/Text' in each_metadata["semantic_types"]:
            # detect Q-nodes by content
            data = list(filter(None, input_dataframe.iloc[:, i].dropna().tolist()))
            if all(re.match(r'^Q\d+$', x) for x in data):
                input.metadata = input.metadata.update(selector=(res_id, ALL_ELEMENTS, i), metadata={
                    "semantic_types": ('http://schema.org/Text',
                                       'https://metadata.datadrivendiscovery.org/types/Attribute',
                                       Q_NODE_SEMANTIC_TYPE)
                })
                _logger.debug("Q nodes format data found in column No.{}, will not run wikifier.".format(str(i)))
                find_q_node_columns = True

    return find_q_node_columns, input


def save_wikifier_choice(input_dataframe: pd.DataFrame, choice: bool = None) -> bool:
    """
    Function used to check whether a given dataframe need to run wikifier or not, if check failed, default not to do wikifier
    :param choice: a optional param, if given, use user's setting, otherwise by checking the size of the input dataframe
    :param input_dataframe: the supplied dataframe that need to be wikified
    :return: a bool, True means it need to be wikifiered, False means not need
    """
    try:
        hash_input_data = str(hash_pandas_object(input_dataframe).sum())
        # if folder / file, create it
        storage_loc = os.path.join(config.cache_file_storage_base_loc, "other_cache")
        if not os.path.exists(storage_loc):
            os.mkdir(storage_loc)
        file_loc = os.path.join(storage_loc, "wikifier_choice.json")
        if os.path.exists(file_loc):
            with open(file_loc, 'r') as f:
                wikifier_choices = json.load(f)
        else:
            wikifier_choices = dict()

        if choice is None:
            input_size = input_dataframe.shape[0] * input_dataframe.shape[1]
            if input_size >= config.maximum_accept_wikifier_size:
                choice = False
            else:
                choice = True

        if hash_input_data in wikifier_choices.keys() and wikifier_choices[hash_input_data] != choice:
            _logger.warning("Exist wikifier choice and the old choice is different!")
            _logger.warning("Now change wikifier choice for dataset with hash tag " + hash_input_data + " to " + str(choice))

        wikifier_choices[hash_input_data] = choice

        with open(file_loc, 'w') as f:
            json.dump(wikifier_choices, f)
        return choice

    except Exception as e:
        _logger.error("Saving wikifier choice failed!")
        _logger.debug(e, exc_info=True)
        return False
