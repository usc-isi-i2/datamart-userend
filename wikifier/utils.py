import json
import typing
import os
import copy
import logging
import typing
import frozendict
import random
import pandas as pd
from d3m.container import Dataset as d3m_Dataset
from d3m.metadata.base import ALL_ELEMENTS
from d3m.base import utils as d3m_utils
from collections import defaultdict
from datamart_isi import config
from pandas.util import hash_pandas_object
_logger = logging.getLogger(__name__)


ADDED_WOREDA_WIKIDATA_COLUMN_NAME = 'woreda_wikidata'
SAMPLE_AMOUNT = 100


def check_wikifier_choice(input_dataframe: pd.DataFrame) -> typing.Union[bool, None]:
    """
    Function used to find the recorded choice for a given dataset, if no record found, will return False
    :param input_dataframe: the input supplied dataframe
    :return: a bool, True means it need to be wikified, False means not need
    """
    try:
        hash_input_data = str(hash_pandas_object(input_dataframe).sum())
        file_loc = os.path.join(config.cache_file_storage_base_loc, "other_cache", "wikifier_choice.json")
        with open(file_loc, 'r') as f:
            wikifier_choices = json.load(f)

        # if we find exist choice, use this
        if hash_input_data in wikifier_choices.keys():
            wikifier_choice = wikifier_choices[hash_input_data]
            return wikifier_choice
        else:
            _logger.error("No choice record found for dataset " + hash_input_data)
            return None

    except Exception as e:
        _logger.error("Check wikifier choice failed!")
        _logger.debug(e, exc_info=True)
        return None

def wikifier_for_ethiopia_dataset(input_dataset: d3m_Dataset) -> typing.Tuple[bool, d3m_Dataset]:
    """
        wrapped version for d3m dataset
    """
    res_id, input_dataframe = d3m_utils.get_tabular_resource(dataset=input_dataset, resource_id=None)
    wikifiered_dataframe = wikifier_for_ethiopia(input_dataframe)
    if wikifiered_dataframe.shape == input_dataframe.shape:
        return False, input_dataset
    else:
        updated_dataset = copy.copy(input_dataset)
        updated_dataset[res_id] = wikifiered_dataframe
        updated_dataset.metadata = _update_metadata(updated_dataset.metadata, res_id)
        return True, updated_dataset

def _update_metadata(metadata, res_id):
    """
        update the metadata for wikifiered ethiopia dataset
    """
    column_selector = (res_id, ALL_ELEMENTS)
    column_metadata = dict(metadata.query(column_selector))
    updated_dimension_meta = dict(column_metadata['dimension'])
    updated_column_number = updated_dimension_meta['length']
    updated_dimension_meta['length'] += 1
    column_metadata['dimension'] = frozendict.FrozenOrderedDict(updated_dimension_meta)
    metadata = metadata.update(column_selector, column_metadata)
    updated_column_metadata = {
                                'name': ADDED_WOREDA_WIKIDATA_COLUMN_NAME, 
                                'structural_type': str, 
                                'semantic_types': (config.q_node_semantic_type, 
                                    config.augmented_column_semantic_type, 
                                    config.attribute_semantic_type,
                                    config.text_semantic_type,)
                              }
    metadata = metadata.update((res_id, ALL_ELEMENTS, updated_column_number), updated_column_metadata)
    return metadata

def wikifier_for_ethiopia(input_dataframe: pd.DataFrame, threshold=0.1) -> pd.DataFrame:
    """
        special wikifier that use human-generated dict to find correct woreda Q nodes from wikidata
        it will only care about woreda related columns
    """
    import wikifier
    output_dataframe = copy.deepcopy(input_dataframe)
    oromia_wikifier_file = os.path.join(wikifier.__path__[0], "oromia_woreda_wikifier.json")
    with open(oromia_wikifier_file, "r") as f:
        woreda_dict = json.load(f)

    # reference by woreda name or woreda id
    reverse_dict_name = defaultdict(dict)
    reverse_dict_id = defaultdict(dict)
    for k, v in woreda_dict.items():
        for each_label in v['labels']:
            reverse_dict_name[each_label.lower()][v['region'].lower()] = k
    for k, v in woreda_dict.items():
        for each_id in v['woreda_ids']:
            reverse_dict_id[each_id.lower()][v['region'].lower()] = k

    # woredas = []
    # qnodes = []
    # for k ,v in woreda_dict.items():
    #     for each_label in v['labels']:
    #         qnodes.append(k)
    #         woredas.append(each_label + "-" + v['region'])


    col_hit_count = defaultdict(int)
    random.seed(42)
    if len(output_dataframe) < SAMPLE_AMOUNT:
        chosen_rows = range(len(output_dataframe))
    else:
        chosen_rows = random.sample(range(len(output_dataframe)), k = SAMPLE_AMOUNT)

    for _, each_row in output_dataframe.iloc[chosen_rows, :].iterrows():
        for col_num, each_cell in enumerate(each_row):
            if str(each_cell).lower() in reverse_dict_name:
                col_hit_count[str(col_num) + "_as_name"] += 1
            try:
                if str(int(each_cell)) in reverse_dict_id:
                    col_hit_count[str(col_num) + "_as_id"] += 1
            except:
                pass

    max_hit_v = 0
    max_hit_k = None
    for k, v in col_hit_count.items():
        if v > max_hit_v:
            max_hit_k = k
            max_hit_v = v

    if threshold > (max_hit_v / SAMPLE_AMOUNT):
        _logger.warning("The coverge of found rows is {} which less than threshold, will not consider.".format(str(max_hit_v / 100)))
        return output_dataframe

    elif max_hit_k is None:
        _logger.warning("can't find any column that relate to woreda information!")
        return output_dataframe

    max_hit_column_num = int(max_hit_k.split("_as_")[0])
    _logger.info("Will wikifier base on column No.{} {}".format(str(max_hit_column_num), str(output_dataframe.columns[max_hit_column_num])))

    if max_hit_k.endswith("_as_name"):
        use_dict = reverse_dict_name
    elif max_hit_k.endswith("_as_id"):
        use_dict = reverse_dict_id

    wikifier_result_list = []
    found_q_node_count = 0
    for i, each_val in enumerate(output_dataframe.iloc[:, max_hit_column_num]):
        information = (use_dict.get(str(each_val).lower()))
        q_node = None
        if information is not None:
            if len(information) == 1:
                q_node = list(information.values())[0]
            else:
                # need to check other information
                each_row = output_dataframe.iloc[i, :]
                for each_val in each_row:
                    temp_key = str(each_val).lower()
                    if temp_key in information:
                        q_node = information[temp_key]
                        break
                if q_node is None:
                    q_node = list(information.values())[0]
                    _logger.warning("can't find information for row {}, will use the first one as {}".format(str(each_row.tolist()), str(q_node)))
        if q_node is not None:
            found_q_node_count += 1
        else:
            q_node == ""
        wikifier_result_list.append(q_node)

    _logger.info("Totally {} of {} woreda data found.".format(str(found_q_node_count), str(len(wikifier_result_list))))
    # add to dataframe
    output_dataframe[ADDED_WOREDA_WIKIDATA_COLUMN_NAME] = wikifier_result_list

    return output_dataframe