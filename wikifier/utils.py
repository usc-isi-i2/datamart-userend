import json
import typing
import os
import logging
import pandas as pd
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
