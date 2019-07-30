import wikifier
import pandas as pd
import copy
import os
import typing
import frozendict
import logging
import json
import hashlib
from d3m.base import utils as d3m_utils
from d3m.container import Dataset as d3m_Dataset
from d3m.container import DataFrame as d3m_DataFrame
from d3m.metadata.base import ALL_ELEMENTS
from datamart_isi import config
from datamart_isi.utilities.utils import Utils
from os import path

Q_NODE_SEMANTIC_TYPE = config.q_node_semantic_type
DEFAULT_TEMP_PATH = config.default_temp_path
_logger = logging.getLogger(__name__)


def run_wikifier(supplied_data: d3m_Dataset):
    try:
        output_ds = copy.copy(supplied_data)
        need_column_type = config.need_wikifier_column_type_list
        res_id, supplied_dataframe = d3m_utils.get_tabular_resource(dataset=supplied_data, resource_id=None)
        specific_p_nodes = get_specific_p_nodes(supplied_dataframe)
        if specific_p_nodes:
            _logger.info("Get specific column<->p_nodes relationship from previous TRAIN run.")
            _logger.info(str(specific_p_nodes))
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
            # if the column type inside here found, this coumn should be wikified
            if set(each_column_semantic_type).intersection(need_column_type):
                continue
            # if the column type inside here found, this column should not be wikified
            elif set(each_column_semantic_type).intersection(skip_column_type):
                temp.remove(each)
            elif supplied_dataframe.columns[each] == "d3mIndex":
                temp.remove(each)

        target_columns = temp
        _logger.debug("The target columns need to be wikified are: " + str(target_columns))
        wikifier_res = wikifier.produce(pd.DataFrame(supplied_dataframe), target_columns, specific_p_nodes)
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

        # the augmented dataframe should not run wikifier again to ensure the semantic type is correct
        # TODO: In this way, we will not search on augmented columns if run second time of wikifier
        wikifier.save_specific_p_nodes(original_dataframe=wikifier_res, column_to_p_node_dict=dict())
        Utils.save_metadata_from_dataset(output_ds)

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
        _logger.error("Wikifier running failed.")
        _logger.debug(e, exc_info=True)
        return supplied_data


def get_specific_p_nodes(supplied_dataframe) -> typing.Optional[list]:
    columns_list = supplied_dataframe.columns.tolist()
    columns_list.sort()
    hash_generator = hashlib.md5()
    hash_generator.update(str(columns_list).encode('utf-8'))
    hash_key = str(hash_generator.hexdigest())
    temp_path = os.getenv('D3MLOCALDIR', DEFAULT_TEMP_PATH)
    specific_q_nodes_file = os.path.join(temp_path, hash_key + "_column_to_P_nodes")
    if path.exists(specific_q_nodes_file):
        with open(specific_q_nodes_file, 'r') as f:
            res = json.load(f)
            return res
    else:
        return None


# def save_specific_p_nodes(original_dataframe, wikifiered_dataframe) -> bool:
#     try:
#         original_columns_list = set(original_dataframe.columns.tolist())
#         wikifiered_columns_list = set(wikifiered_dataframe.columns.tolist())
#         p_nodes_list = list(wikifiered_columns_list - original_columns_list)
#         p_nodes_list.sort()
#         p_nodes_str = ",".join(p_nodes_list)
#
#         hash_generator = hashlib.md5()
#         hash_generator.update(str(p_nodes_str).encode('utf-8'))
#         hash_key = str(hash_generator.hexdigest())
#         temp_path = os.getenv('D3MLOCALDIR', DEFAULT_TEMP_PATH)
#         specific_q_nodes_file = os.path.join(temp_path, hash_key)
#         if path.exists(specific_q_nodes_file):
#             _logger.warning("The specific p nodes file already exist! Will replace the old one!")
#
#         with open(specific_q_nodes_file, 'w') as f:
#              f.write(p_nodes_str)
#         return True
#
#     except Exception as e:
#         _logger.debug(e, exc_info=True)
#         return False
