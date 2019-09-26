import typing
import hashlib
import os
import logging
import json

from d3m.container.dataset import D3MDatasetLoader
from d3m.container import Dataset as d3m_Dataset
from d3m.metadata.base import ALL_ELEMENTS
from d3m.base import utils as d3m_utils
from datamart_isi.config import cache_file_storage_base_loc

_logger = logging.getLogger(__name__)
seed_dataset_store_location = os.path.join(cache_file_storage_base_loc, "datasets_cache")
if not os.path.exists(seed_dataset_store_location):
    os.mkdir(seed_dataset_store_location)


class MetadataCache:
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
        hash_generator = hashlib.md5()
        hash_generator.update(str(input_columns).encode('utf-8'))
        hash_key = str(hash_generator.hexdigest())
        _logger.debug("Current columns are: " + str(input_columns))
        _logger.debug("Current dataset's hash key is: " + hash_key)
        try:
            file_loc = os.path.join(cache_folder, hash_key + "_metadata")
            if os.path.exists(file_loc):
                _logger.info("found exist metadata from seed datasets! Will use that")
                with open(file_loc, "r") as f:
                    metadata_info = json.load(f)
                    _logger.info("The hit dataset id is: " + metadata_info["dataset_id"])

                    for i in range(len(input_columns)):
                        selector = (res_id, ALL_ELEMENTS, i)
                        current_column_name = input_dataframe.columns[i]
                        new_semantic_type = metadata_info[current_column_name]
                        input_dataset.metadata = input_dataset.metadata.update(selector, {"semantic_types": new_semantic_type})
                    return True, input_dataset
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
            current_dataset_metadata_dict = dict()
            res_id, current_dataframe = d3m_utils.get_tabular_resource(dataset=current_dataset, resource_id=None)
            current_dataset_metadata_dict["dataset_id"] = current_dataset.metadata.query(())['id']
            for i in range(current_dataframe.shape[1]):
                each_metadata = current_dataset.metadata.query((res_id, ALL_ELEMENTS, i))
                current_dataset_metadata_dict[current_dataframe.columns[i]] = each_metadata['semantic_types']
            input_columns = current_dataframe.columns.tolist()
            input_columns.sort()
            hash_generator = hashlib.md5()
            hash_generator.update(str(input_columns).encode('utf-8'))
            hash_key = str(hash_generator.hexdigest())
            _logger.debug("Current columns are: " + str(input_columns))
            _logger.debug("Current dataset's hash key is: " + hash_key)
            file_loc = os.path.join(cache_folder, hash_key + "_metadata")
            with open(file_loc, "w") as f:
                json.dump(current_dataset_metadata_dict, f)
            _logger.info("Saving " + current_dataset_metadata_dict["dataset_id"] + " to " + file_loc + " success!")
            return True

        except Exception as e:
            _logger.error("Saving dataset failed!")
            _logger.debug(e, exc_info=True)
            return False
