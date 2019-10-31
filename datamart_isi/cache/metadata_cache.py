import typing
import hashlib
import os
import logging
import json
import pandas as pd

from d3m.container.dataset import D3MDatasetLoader
from d3m.container import Dataset as d3m_Dataset
from d3m.metadata.base import ALL_ELEMENTS
from d3m.base import utils as d3m_utils
from datamart_isi.config import cache_file_storage_base_loc
from datamart_isi.config import default_temp_path

DEFAULT_TEMP_PATH = default_temp_path
_logger = logging.getLogger(__name__)
seed_dataset_store_location = os.path.join(cache_file_storage_base_loc, "datasets_cache")
wikifier_target_cache_exist_mark = "wikifier_target_cache_exist_mark"
if not os.path.exists(seed_dataset_store_location):
    print(f'Creating directory: {seed_dataset_store_location}')
    os.makedirs(seed_dataset_store_location, exist_ok=True)


class MetadataCache:
    @staticmethod
    def get_hash_key(input_data: pd.DataFrame) -> str:
        """
        Function used to get the hash key for dataset cache
        :param input_data:
        :return: the hash key of the input data
        """
        data_columns_list = input_data.columns.tolist()
        data_columns_list.sort()
        hash_generator = hashlib.md5()

        hash_generator.update(str(data_columns_list).encode('utf-8'))
        hash_key = str(hash_generator.hexdigest())
        _logger.debug("Current columns are: " + str(data_columns_list))
        _logger.debug("Current dataset's hash key is: " + hash_key)
        return hash_key

    @staticmethod
    def save_specific_wikifier_targets(current_dataframe, column_to_p_node_dict,
                                       cache_folder: str = seed_dataset_store_location) -> bool:
        hash_key = MetadataCache.get_hash_key(current_dataframe)
        # delete previous exist wikification target first
        MetadataCache.delete_specific_p_nodes_file(current_dataframe)
        file_loc = os.path.join(cache_folder, hash_key + "_metadata")
        if os.path.exists(file_loc):
            with open(file_loc, "r") as f:
                current_dataset_metadata_dict = json.load(f)
        else:
            current_dataset_metadata_dict = dict()
        try:
            # add wikifier targets to file
            current_dataset_metadata_dict[wikifier_target_cache_exist_mark] = True
            for i in range(current_dataframe.shape[1]):
                current_column_name = current_dataframe.columns[i]
                if current_column_name in column_to_p_node_dict:
                    current_dataset_metadata_dict[current_column_name + "_wikifier_target"] = column_to_p_node_dict[current_column_name]

            with open(file_loc, "w") as f:
                json.dump(current_dataset_metadata_dict, f)
            _logger.info("Saving wikifier targets to " + file_loc + " success!")
            return True

        except Exception as e:
            _logger.error("Saving dataset failed!")
            _logger.debug(e, exc_info=True)
            return False

    @staticmethod
    def check_and_get_dataset_real_metadata(input_dataset: d3m_Dataset, cache_folder: str = seed_dataset_store_location):
        """
        Function used to update the received dataset from datamart REST API with exist metadata if possible
        :param input_dataset: input dataset
        :param cache_folder:  the specific cache folder location, if not given, use default
        :return: The fixed dataset if possible, otherwise return original one
        """
        _logger.debug("current cache folder is: " + cache_folder)

        res_id, input_dataframe = d3m_utils.get_tabular_resource(dataset=input_dataset, resource_id=None)
        input_columns = input_dataframe.columns.tolist()
        input_columns.sort()
        hash_key = MetadataCache.get_hash_key(input_dataframe)

        try:
            file_loc = os.path.join(cache_folder, hash_key + "_metadata")
            if os.path.exists(file_loc):

                with open(file_loc, "r") as f:
                    metadata_info = json.load(f)
                    if "dataset_id" in metadata_info:
                        _logger.info("found exist metadata! Will use that")
                        _logger.info("The hit dataset id is: " + metadata_info["dataset_id"])
                        for i in range(len(input_columns)):
                            selector = (res_id, ALL_ELEMENTS, i)
                            current_column_name = input_dataframe.columns[i]
                            new_semantic_type = metadata_info[current_column_name]
                            input_dataset.metadata = input_dataset.metadata.update(selector, {"semantic_types": new_semantic_type})
                        return True, input_dataset
                    else:
                        _logger.info("Found file but the file do not contains metadata information for columns")
                        return False, input_dataset
            else:
                _logger.warning("No exist metadata from seed datasets found!")
                return False, input_dataset
        except Exception as e:
            _logger.warning("Trying to check whether the given dataset exist in seed augment failed, will skip.")
            _logger.debug(e, exc_info=True)
            return False, input_dataset

    @staticmethod
    def generate_real_metadata_files(dataset_paths: typing.List[str], cache_folder: str = seed_dataset_store_location):
        """
        Function that used to find the correct metadata from d3m datasets and save them into cache_folder, then those files can be
        accessed from datamart
        :param dataset_paths: a string format path to the dir that stored d3m datasets
        :param cache_folder: a string format path to the dir that store the metadata cache
        :return: None
        """
        loader = D3MDatasetLoader()
        for each_dataset_path in dataset_paths:
            if os.path.exists(each_dataset_path):
                current_dataset_paths = [os.path.join(each_dataset_path, o) for o in os.listdir(each_dataset_path)
                                         if os.path.isdir(os.path.join(each_dataset_path, o))]
                for current_dataset_path in current_dataset_paths:
                    dataset_doc_json_path = os.path.join(current_dataset_path,
                                                         current_dataset_path.split("/")[-1] + "_dataset",
                                                         "datasetDoc.json")
                    print(dataset_doc_json_path)
                    all_dataset_uri = 'file://{}'.format(dataset_doc_json_path)
                    current_dataset = loader.load(dataset_uri=all_dataset_uri)
                    response = MetadataCache.save_metadata_from_dataset(current_dataset, cache_folder)
                    if not response:
                        _logger.error("Saving dataset from " + current_dataset_path + " failed!")

            else:
                _logger.error("Path " + each_dataset_path + " do not exist!")

    @staticmethod
    def save_metadata_from_dataset(current_dataset: d3m_Dataset, cache_folder: str = seed_dataset_store_location) -> bool:
        """
        Function that store one dataset's metadata
        :param current_dataset: a d3m dataset
        :param cache_folder: a string format path to the dir that store the metadata cache
        :return: a Bool indicate saving success or not
        """
        try:
            res_id, current_dataframe = d3m_utils.get_tabular_resource(dataset=current_dataset, resource_id=None)
            hash_key = MetadataCache.get_hash_key(current_dataframe)
            file_loc = os.path.join(cache_folder, hash_key + "_metadata")
            if os.path.exists(file_loc):
                with open(file_loc, "r") as f:
                    current_dataset_metadata_dict = json.load(f)
            else:
                current_dataset_metadata_dict = dict()

            current_dataset_metadata_dict["dataset_id"] = current_dataset.metadata.query(())['id']
            for i in range(current_dataframe.shape[1]):
                each_metadata = current_dataset.metadata.query((res_id, ALL_ELEMENTS, i))
                current_dataset_metadata_dict[current_dataframe.columns[i]] = each_metadata['semantic_types']
            with open(file_loc, "w") as f:
                json.dump(current_dataset_metadata_dict, f)
            _logger.info("Saving " + current_dataset_metadata_dict["dataset_id"] + " to " + file_loc + " success!")
            return True

        except Exception as e:
            _logger.error("Saving dataset failed!")
            _logger.debug(e, exc_info=True)
            return False

    @staticmethod
    def generate_specific_meta_path(supplied_dataframe, cache_folder: str = seed_dataset_store_location):
        hash_key = MetadataCache.get_hash_key(supplied_dataframe)
        file_loc = os.path.join(cache_folder, hash_key + "_metadata")
        return file_loc

    @staticmethod
    def get_specific_p_nodes(supplied_dataframe) -> typing.Optional[dict]:
        specific_q_nodes_file = MetadataCache.generate_specific_meta_path(supplied_dataframe)
        if os.path.exists(specific_q_nodes_file):
            with open(specific_q_nodes_file, 'r') as f:
                loaded_metadata = json.load(f)
            specific_p_nodes_dict = dict()
            # if no mark exist, it means this dataset's wikifier cache not saved, so we should return None
            if wikifier_target_cache_exist_mark not in loaded_metadata:
                return None
            # otherwise, find corresponding wikifier targets
            # it is possible to return an empty dict to indicate that no columns can be wikified
            for key, value in loaded_metadata.items():
                if key.endswith("_wikifier_target"):
                    column_name = key[:-16]
                    specific_p_nodes_dict[column_name] = value
            return specific_p_nodes_dict
        else:
            return None

    @staticmethod
    def delete_specific_p_nodes_file(supplied_dataframe):
        specific_q_nodes_file = MetadataCache.generate_specific_meta_path(supplied_dataframe)
        if os.path.exists(specific_q_nodes_file):
            with open(specific_q_nodes_file, "r") as f:
                loaded_metadata = json.load(f)
            keys_need_to_remove = []
            for key in loaded_metadata.keys():
                if key.endswith("_wikifier_target") or key == wikifier_target_cache_exist_mark:
                    keys_need_to_remove.append(key)
            _logger.debug("Following specific wikifier targets will be removed:" + str(keys_need_to_remove))
            for each_key in keys_need_to_remove:
                loaded_metadata.pop(each_key)

            with open(specific_q_nodes_file, "w") as f:
                json.dump(loaded_metadata, f)
            _logger.info("Delete specific p node files on {} success!".format(specific_q_nodes_file))
