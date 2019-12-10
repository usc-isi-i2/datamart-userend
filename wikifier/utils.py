import json
import typing
import os
import logging
import pandas as pd
from collections import defaultdict
from datamart_isi import config
from pandas.util import hash_pandas_object
_logger = logging.getLogger(__name__)


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

def wikifier_for_ethiopia(input_dataframe: pd.DataFrame) -> pd.DataFrame:
    """
        special wikifier that use human-generated dict to find correct woreda Q nodes from wikidata
        it will only care about woreda related columns
    """
    with open("oromia_woreda_wikifier.json", "r") as f:
        woreda_dict = json.load(f)

    # reference by woreda name or woreda id
    reverse_dict_name = defaultdict(dict)
    reverse_dict_id = defaultdict(dict)
    for k, v in woreda_dict.items():
        for each_label in v['labels']:
            reverse_dict_name[each_label.lower()][v['region']] = k
    for k, v in woreda_dict.items():
        for each_id in v['woreda_ids']:
            reverse_dict_id[each_id.lower()][v['region']] = k

    col_hit_count = defaultdict(int)
    for i, each_row in input_dataframe.iterrows():
        if i >= 100:
            break
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

    if max_hit_k is None:
        _logger.warning("can't find any column that relate to woreda information!")
        return input_dataframe

    _logger.info("Will wikifier base on column No.{} {}".format(str(max_hit_k), str(input_dataframe.columns[max_hit_k])))

    if max_hit_k.endswith("_as_name"):
        use_dict = reverse_dict_name
        col_num = int(max_hit_k[:-8])
    elif max_hit_k.endswith("_as_id"):
        use_dict = reverse_dict_id
        col_num = int(max_hit_k[:-6])

    wikifier_result_list = []
    for each_val in input_dataframe.iloc[:, col_num]:
        wikifier_result_list.append(use_dict.get(str(each_val)))
    

