import typing
import pandas as pd
import wikifier
import traceback
import hashlib
import os
import logging
import json
from SPARQLWrapper import SPARQLWrapper, JSON, POST, URLENCODED
from io import StringIO
from ast import literal_eval
from d3m.container import DataFrame as d3m_DataFrame
from d3m.container.dataset import D3MDatasetLoader
from d3m.container import Dataset as d3m_Dataset
from d3m.metadata.base import DataMetadata, ALL_ELEMENTS
from d3m.base import utils as d3m_utils
from datamart_isi.config import wikidata_server
from datamart_isi.config import cache_file_storage_base_loc

WIKIDATA_SERVER = wikidata_server
_logger = logging.getLogger(__name__)
seed_dataset_store_location = os.path.join(cache_file_storage_base_loc, "datasets_cache")


class Utils:
    DEFAULT_DESCRIPTION = {
        "materialization": {
            "python_path": "default_materializer"
        },
        "variables": []
    }

    @staticmethod
    def materialize(metadata) -> pd.DataFrame:
        # general type materializer
        if 'url' in metadata:
            dataset_url = metadata['url']['value']
            from dsbox.datapreprocessing.cleaner.data_profile import Profiler, Hyperparams as ProfilerHyperparams
            from dsbox.datapreprocessing.cleaner.cleaning_featurizer import CleaningFeaturizer, CleaningFeaturizerHyperparameter
            file_type = metadata.get("file_type") or ""
            if file_type == "":
                # no file type get, try to guess
                file_type = dataset_url.split(".")[-1]
            else:
                file_type = file_type['value']

            if file_type == "wikitable":
                extra_information = literal_eval(metadata['extra_information']['value'])
                loaded_data = Utils.materialize_for_wikitable(dataset_url, file_type, extra_information)
            else:
                loaded_data = Utils.materialize_for_general(dataset_url, file_type)

            # run dsbox's profiler and cleaner
            hyper1 = ProfilerHyperparams.defaults()
            profiler = Profiler(hyperparams=hyper1)
            profiled_df = profiler.produce(inputs=loaded_data).value
            hyper2 = CleaningFeaturizerHyperparameter.defaults()
            clean_f = CleaningFeaturizer(hyperparams=hyper2)
            clean_f.set_training_data(inputs=profiled_df)
            clean_f.fit()
            cleaned_df = pd.DataFrame(clean_f.produce(inputs=profiled_df).value)
            wikifier_res = wikifier.produce(cleaned_df)

            return wikifier_res

        elif "p_nodes_needed" in metadata:
            # wikidata materializer
            label_part = "  ?itemLabel \n"
            where_part = ""
            for i, each_p_node in enumerate(metadata["p_nodes_needed"]):
                label_part += "  ?value" + str(i) + "Label\n"
                where_part += "  ?item wdt:" + each_p_node + " ?value" + str(i) + ".\n"
            try:
                sparql_query = """PREFIX wikibase: <http://wikiba.se/ontology#>
                                  PREFIX wd: <http://www.wikidata.org/entity/>
                                  prefix bd: <http://www.bigdata.com/rdf#>
                                  PREFIX wdt: <http://www.wikidata.org/prop/direct/>
                                  SELECT \n""" + label_part + "WHERE \n {\n" + where_part \
                               + """  SERVICE wikibase:label { bd:serviceParam wikibase:language "[AUTO_LANGUAGE],en". }\n}\n""" \
                               + "LIMIT 100"

                sparql = SPARQLWrapper(WIKIDATA_SERVER)
                sparql.setQuery(sparql_query)
                sparql.setReturnFormat(JSON)
                # sparql.setMethod(POST)
                # sparql.setRequestMethod(URLENCODED)
            except:
                print("[ERROR] Wikidata query failed!")
                traceback.print_exc()

            results = sparql.query().convert()["results"]["bindings"]
            all_res = {}
            for i, result in enumerate(results):
                each_res = {}
                for each_key in result.keys():
                    each_res[each_key] = result[each_key]['value']
                all_res[i] = each_res
            df_res = pd.DataFrame.from_dict(all_res, "index")
            column_names = df_res.columns.tolist()
            column_names = column_names[1:]
            column_names_replaced = dict()
            for each in zip(column_names, metadata["p_nodes_needed"]):
                column_names_replaced[each[0]] = Utils.get_node_name(each[1])
            df_res.rename(columns=column_names_replaced, inplace=True)

            df_res = d3m_DataFrame(df_res, generate_metadata=True)
            return df_res
        else:
            raise ValueError("Unknown type for materialize!")

    @staticmethod
    def get_node_name(node_code) -> str:
        """
        Function used to get the properties(P nodes) names with given P node
        :param node_code: a str indicate the P node (e.g. "P123")
        :return: a str indicate the P node label (e.g. "inception")
        """
        sparql_query = "SELECT DISTINCT ?x WHERE \n { \n" + \
                       "wd:" + node_code + " rdfs:label ?x .\n FILTER(LANG(?x) = 'en') \n} "
        try:
            sparql = SPARQLWrapper(WIKIDATA_SERVER)
            sparql.setQuery(sparql_query)
            sparql.setReturnFormat(JSON)
            sparql.setMethod(POST)
            sparql.setRequestMethod(URLENCODED)
            results = sparql.query().convert()
            return results['results']['bindings'][0]['x']['value']
        except:
            return node_code

    @staticmethod
    def materialize_for_wikitable(dataset_url: str, file_type: str, extra_information: str) -> pd.DataFrame:
        from datamart_isi.materializers.wikitables_materializer import WikitablesMaterializer
        materializer = WikitablesMaterializer()
        loaded_data = materializer.get_one(dataset_url, extra_information['xpath'])
        return loaded_data

    @staticmethod
    def materialize_for_general(dataset_url: str, file_type: str) -> pd.DataFrame:
        from datamart_isi.materializers.general_materializer import GeneralMaterializer
        general_materializer = GeneralMaterializer()
        file_metadata = {
            "materialization": {
                "arguments": {
                    "url": dataset_url,
                    # one example here: "url": "http://insight.dev.schoolwires.com/HelpAssets/C2Assets/C2Files/C2ImportFamRelSample.csv",
                    "file_type": file_type
                }
            }
        }

        try:
            result = general_materializer.get(metadata=file_metadata).to_csv(index=False)
            # remove last \n so that we will not get an extra useless row
            if result[-1] == "\n":
                result = result[:-1]

            loaded_data = StringIO(result)
            loaded_data = pd.read_csv(loaded_data, dtype="str")
            return loaded_data
        except:
            traceback.print_exc()
            raise ValueError("Materializing from " + dataset_url + " failed!")

    @staticmethod
    def calculate_dsbox_features(data: pd.DataFrame, metadata: typing.Union[dict, None],
                                 selected_columns: typing.Set[int] = None) -> dict:
        """Calculate dsbox features, add to metadata dictionary

         Args:
             data: dataset as a pandas dataframe
             metadata: metadata dict

         Returns:
              updated metadata dict
         """

        from datamart_isi.profilers.dsbox_profiler import DSboxProfiler
        if not metadata:
            return metadata
        return DSboxProfiler().profile(inputs=data, metadata=metadata, selected_columns=selected_columns)

    @classmethod
    def generate_metadata_from_dataframe(cls, data: pd.DataFrame, original_meta: dict = None) -> dict:
        """Generate a default metadata just from the data, without the dataset schema

         Args:
             data: pandas DataFrame

         Returns:
              metadata dict
         """
        from datamart_isi.profilers.basic_profiler import BasicProfiler, VariableMetadata, GlobalMetadata

        global_metadata = GlobalMetadata.construct_global(description=cls.DEFAULT_DESCRIPTION)
        for col_offset in range(data.shape[1]):
            variable_metadata = BasicProfiler.basic_profiling_column(
                description={},
                variable_metadata=VariableMetadata.construct_variable(description={}),
                column=data.iloc[:, col_offset]
            )
            global_metadata.add_variable_metadata(variable_metadata)
        global_metadata = BasicProfiler.basic_profiling_entire(global_metadata=global_metadata,
                                                               data=data)
        if original_meta:
            global_metadata.value.update(original_meta)
        return global_metadata.value

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
                    response = Utils.save_metadata_from_dataset(current_dataset, cache_folder)
                    if not response:
                        _logger.error("Saving dataset from " + current_dataset_path + " failed!")

            else:
                _logger.error("Path " + each_dataset_path + " do not exist!")

    @staticmethod
    def save_metadata_from_dataset(current_dataset: d3m_Dataset, cache_folder: str = seed_dataset_store_location) -> bool:
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
            file_loc = os.path.join(cache_folder, hash_key + "_metadata")
            with open(file_loc, "w") as f:
                json.dump(current_dataset_metadata_dict, f)
            _logger.info("Saving " + current_dataset_metadata_dict["dataset_id"] + " to " + file_loc + " success!")
            return True

        except Exception as e:
            _logger.error("Saving dataset failed!")
            _logger.debug(e, exc_info=True)
            return False
