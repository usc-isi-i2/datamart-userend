import logging
import pandas as pd
import typing
import collections
import frozendict
import traceback
import re

from ast import literal_eval
from d3m.metadata.base import DataMetadata, ALL_ELEMENTS
from d3m.base import utils as d3m_utils
from d3m.container import DataFrame as d3m_DataFrame
from d3m.container import Dataset as d3m_Dataset
from datamart_isi import config
from datamart_isi.utilities.download_manager import DownloadManager
from datamart_isi.utilities.d3m_wikifier import check_and_correct_q_nodes_semantic_type
from datamart_isi.config import q_node_semantic_type

AUGMENT_RESOURCE_ID = config.augmented_resource_id
AUGMENTED_COLUMN_SEMANTIC_TYPE = config.augmented_column_semantic_type
Q_NODE_SEMANTIC_TYPE = config.q_node_semantic_type
CONTAINER_SCHEMA_VERSION = config.d3m_container_version


class MetadataGenerator:
    def __init__(self, supplied_data, search_result, search_type, connection_url, wikidata_cache_manager):
        self._logger = logging.getLogger(__name__)
        self.supplied_data = supplied_data
        self.search_result = search_result
        self.search_type = search_type
        self.connection_url = connection_url
        self.res_id = None
        self.d3m_metadata = None
        if type(supplied_data) is d3m_Dataset:
            self.res_id, self.supplied_dataframe = d3m_utils.get_tabular_resource(dataset=supplied_data,
                                                                                  resource_id=None)
            self.selector_base_type = "ds"
        elif type(supplied_data) is d3m_DataFrame:
            self.supplied_dataframe = supplied_data
            self.selector_base_type = "df"
        else:
            self.supplied_dataframe = None
            
        if self.search_type == "general":
            self.id = self.search_result['datasetLabel']['value']
            self.score = float(self.search_result['score']['value'])
        elif self.search_type == "wikidata":
            self.id = "wikidata search on " + str(self.search_result['p_nodes_needed']) + " with column " + \
                       self.search_result['target_q_node_column_name']
            self.id = self.id.replace(" ", "_")
            self.id = self.id.replace("[", "_")
            self.id = self.id.replace("]", "_")
            self.id = self.id.replace("'", "_")
            self.id = self.id.replace(",", "")
            self.score = 1
        elif self.search_type == "vector":
            self.id = "vector search on Q nodes with column " + \
                       self.search_result['target_q_node_column_name']
            self.id = self.id.replace(" ", "_")
            self.score = 1
        elif self.search_type == "wikifier":
            self.id = "wikifier_for_dataset" + self.res_id if self.res_id else None
            self.score = 1
        else:
            raise ValueError("Unknown search type for this search result as " + str(search_type))
        self.wikidata_cache_manager = wikidata_cache_manager

    def set_supplied_data(self, supplied_data):
        self.supplied_data = supplied_data
        if type(supplied_data) is d3m_Dataset:
            self.res_id, self.supplied_dataframe = d3m_utils.get_tabular_resource(dataset=supplied_data,
                                                                                  resource_id=None)
            self.selector_base_type = "ds"
        elif type(supplied_data) is d3m_DataFrame:
            self.supplied_dataframe = supplied_data
            self.selector_base_type = "df"

    def generate_d3m_metadata_for_search_result(self) -> DataMetadata:
        """
        function used to generate the d3m format metadata
        """
        self._logger.debug("Start getting d3m metadata...")
        if self.search_type == "wikidata":
            metadata = self.generate_metadata_for_wikidata_search()
        elif self.search_type == "general":
            metadata = self.generate_metadata_for_general_search()
        elif self.search_type == "wikifier":
            self._logger.warning("No metadata can provide for wikifier augment")
            metadata = DataMetadata()
        elif self.search_type == "vector":
            metadata = self.generate_metadata_for_vector_search()
        else:
            self._logger.error("Unknown search type as " + str(self.search_type))
            metadata = DataMetadata()
        self._logger.debug("Getting d3m metadata finished.")


        self.d3m_metadata = metadata

        return metadata

    def generate_metadata_for_wikidata_search(self, selector_base=tuple()) -> DataMetadata:
        """
        function used to generate the d3m format metadata - specified for wikidata search result
        because search results don't have value type of each P node, we have to query one sample to find
        """
        return_metadata = DataMetadata()
        if self.supplied_dataframe is not None:
            data_length = self.supplied_dataframe.shape[0]
        elif self.supplied_data is not None:
            res_id, self.supplied_dataframe = d3m_utils.get_tabular_resource(dataset=self.supplied_data,
                                                                             resource_id=None)
            data_length = self.supplied_dataframe.shape[0]
        else:
            self._logger.warning("Can't calculate the row length for wikidata search results without supplied data")
            data_length = None

        metadata_all = {"structural_type": d3m_DataFrame,
                        "semantic_types": ["https://metadata.datadrivendiscovery.org/types/Table"],
                        "dimension": {
                            "name": "rows",
                            "semantic_types": ["https://metadata.datadrivendiscovery.org/types/TabularRow"],
                            "length": data_length,
                        },
                        "schema": "https://metadata.datadrivendiscovery.org/schemas/v0/container.json"
                        }
        return_metadata = return_metadata.update(selector=selector_base + (), metadata=metadata_all)
        metadata_all_elements = {
            "dimension": {
                "name": "columns",
                "semantic_types": ["https://metadata.datadrivendiscovery.org/types/TabularColumn"],
                "length": len(self.search_result['p_nodes_needed']),
            }
        }
        return_metadata = return_metadata.update(selector=selector_base + (ALL_ELEMENTS,), metadata=metadata_all_elements)

        for i, each_p_node in enumerate(self.search_result['p_nodes_needed']):
            target_q_node_column_name = self.search_result['target_q_node_column_name']
            # try to get the semantic type of p nodes, if failed, set it to be Text
            try:
                q_node_column_number = self.supplied_dataframe.columns.tolist().index(target_q_node_column_name)
                sample_row_number = 0
                q_node_sample = self.supplied_dataframe.iloc[sample_row_number, q_node_column_number]
                semantic_types = self._get_wikidata_column_semantic_types(q_node_sample, each_p_node)
                # if we failed with first test, repeat until we get success one
                while not semantic_types[0]:
                    sample_row_number += 1
                    q_node_sample = self.supplied_dataframe.iloc[sample_row_number, q_node_column_number]
                    semantic_types = self._get_wikidata_column_semantic_types(q_node_sample, each_p_node)
            except:
                semantic_types = (
                    "http://schema.org/Text",
                    'https://metadata.datadrivendiscovery.org/types/Attribute',
                    AUGMENTED_COLUMN_SEMANTIC_TYPE
                )
            each_metadata = {
                "name": self.get_node_name(
                    self.search_result['p_nodes_needed'][i]) + "_for_" + target_q_node_column_name,
                "P_node": self.search_result['p_nodes_needed'][i],
                "structural_type": str,
                "semantic_types": semantic_types,
            }
            return_metadata = return_metadata.update(selector=selector_base + (ALL_ELEMENTS, i), metadata=each_metadata)

            each_metadata = {
                "name": "q_node",
                "structural_type": str,
                "semantic_types": (
                    "https://metadata.datadrivendiscovery.org/types/CategoricalData",
                    'https://metadata.datadrivendiscovery.org/types/Attribute',
                    Q_NODE_SEMANTIC_TYPE,
                    AUGMENTED_COLUMN_SEMANTIC_TYPE
                ),
            }
            return_metadata = return_metadata.update(selector=selector_base + (ALL_ELEMENTS, i + 1), metadata=each_metadata)

            each_metadata = {
                "name": "joining_pairs",
                "structural_type": list,
                "semantic_types": (
                    'http://schema.org/Integer',
                    'https://metadata.datadrivendiscovery.org/types/Attribute',
                    AUGMENTED_COLUMN_SEMANTIC_TYPE
                ),
            }
            return_metadata = return_metadata.update(selector=selector_base + (ALL_ELEMENTS, i + 2), metadata=each_metadata)

        return return_metadata

    def _get_wikidata_column_semantic_types(self, q_node_sample, p_node_target) -> tuple:
        """
        Inner function used to get the semantic types for given wikidata column
        :return: a tuple, tuple[0] indicate success get semantic type or not, tuple[1] indicate the found semantic types(tuple)
        """
        q_nodes_query = '(wd:' + q_node_sample + ') \n'
        p_nodes_query_part = ' ?' + p_node_target + '\n'
        p_nodes_optional_part = "  OPTIONAL { ?q wdt:" + p_node_target + " ?" + p_node_target + "}\n"
        sparql_query = "SELECT DISTINCT ?q " + p_nodes_query_part + \
                       "WHERE \n{\n  VALUES (?q) { \n " + q_nodes_query + "}\n" + \
                       p_nodes_optional_part + "}\n"

        results = self.wikidata_cache_manager.get_result(sparql_query)
        try:
            # try to get the results if success (sometimes may failed if no Q nodes corresponded found)
            p_val = results[0][p_node_target]
            if "datatype" in p_val.keys():
                semantic_types = (
                    self.transfer_semantic_type(p_val["datatype"]),
                    'https://metadata.datadrivendiscovery.org/types/Attribute', AUGMENTED_COLUMN_SEMANTIC_TYPE)
            else:
                semantic_types = (
                    "http://schema.org/Text",
                    'https://metadata.datadrivendiscovery.org/types/Attribute',
                    AUGMENTED_COLUMN_SEMANTIC_TYPE)

            return True, semantic_types
        except:
            return False, None
        
    def generate_metadata_for_general_search(self, selector_base=tuple()) -> DataMetadata:
        """
        function used to generate the d3m format metadata - specified for general search result
        """
        return_metadata = DataMetadata()
        metadata_dict = literal_eval(self.search_result['extra_information']['value'])
        data_metadata = metadata_dict.pop('data_metadata')
        metadata_all = {"structural_type": d3m_DataFrame,
                        "semantic_types": ["https://metadata.datadrivendiscovery.org/types/Table"],
                        "dimension": {
                            "name": "rows",
                            "semantic_types": ["https://metadata.datadrivendiscovery.org/types/TabularRow"],
                            "length": int(data_metadata['shape_0']),
                        },
                        "schema": "https://metadata.datadrivendiscovery.org/schemas/v0/container.json"
                        }
        return_metadata = return_metadata.update(selector=selector_base + (), metadata=metadata_all)
        metadata_all_elements = {
            "dimension": {
                "name": "columns",
                "semantic_types": ["https://metadata.datadrivendiscovery.org/types/TabularColumn"],
                "length": int(data_metadata['shape_1']),
            }
        }
        return_metadata = return_metadata.update(selector=selector_base + (ALL_ELEMENTS,), metadata=metadata_all_elements)

        for each_key, each_value in metadata_dict.items():
            if each_key[:12] == 'column_meta_':
                each_metadata = {
                    "name": each_value['name'],
                    "structural_type": str,
                    "semantic_types": each_value['semantic_type'],
                }
                i = int(each_key.split("_")[-1])
                return_metadata = return_metadata.update(selector=selector_base + (ALL_ELEMENTS, i), metadata=each_metadata)

        return return_metadata
    
    def generate_metadata_for_vector_search(self, selector_base=tuple()) -> DataMetadata:
        """
        function used to generate the d3m format metadata - specified for vector search result
        """
        return_metadata = DataMetadata()
        if self.supplied_dataframe is not None:
            data_length = self.supplied_dataframe.shape[0]
        elif self.supplied_data is not None:
            res_id, self.supplied_dataframe = d3m_utils.get_tabular_resource(dataset=self.supplied_data,
                                                                             resource_id=None)
            data_length = self.supplied_dataframe.shape[0]
        else:
            self._logger.warning("Can't calculate the row length for vector search results without supplied data")
            data_length = None

        metadata_all = {"structural_type": d3m_DataFrame,
                        "semantic_types": ["https://metadata.datadrivendiscovery.org/types/Table"],
                        "dimension": {
                            "name": "rows",
                            "semantic_types": ["https://metadata.datadrivendiscovery.org/types/TabularRow"],
                            "length": data_length,
                        },
                        "schema": "https://metadata.datadrivendiscovery.org/schemas/v0/container.json"
                        }
        return_metadata = return_metadata.update(selector=selector_base + (), metadata=metadata_all)
        # fetch the num of columns
        return_df = DownloadManager.fetch_fb_embeddings([self.search_result['q_nodes_list'][0]],
                                                       self.search_result["target_q_node_column_name"])
        length = len(return_df) - 1
        metadata_all_elements = {
            "dimension": {
                "name": "columns",
                "semantic_types": ["https://metadata.datadrivendiscovery.org/types/TabularColumn"],
                "length": length,
            }
        }
        return_metadata = return_metadata.update(selector=selector_base + (ALL_ELEMENTS,), metadata=metadata_all_elements)

        for i in range(length):
            target_q_node_column_name = self.search_result['target_q_node_column_name']
            semantic_types = (
                "http://schema.org/Float",
                'https://metadata.datadrivendiscovery.org/types/Attribute',
                AUGMENTED_COLUMN_SEMANTIC_TYPE
            )
            if i < 10:
                s = '00' + str(i)
            elif i < 100:
                s = '0' + str(i)
            else:
                s = str(i)

            each_metadata = {
                "name": "vector_" + s + "_of_qnode_with_" + target_q_node_column_name,
                "structural_type": float,
                "semantic_types": semantic_types,
            }
            return_metadata = return_metadata.update(selector=selector_base + (ALL_ELEMENTS, i), metadata=each_metadata)

            each_metadata = {
                "name": "q_node",
                "structural_type": str,
                "semantic_types": (
                    "https://metadata.datadrivendiscovery.org/types/CategoricalData",
                    'https://metadata.datadrivendiscovery.org/types/Attribute',
                    Q_NODE_SEMANTIC_TYPE,
                    AUGMENTED_COLUMN_SEMANTIC_TYPE
                ),
            }
            return_metadata = return_metadata.update(selector=selector_base + (ALL_ELEMENTS, i + 1), metadata=each_metadata)

            each_metadata = {
                "name": "joining_pairs",
                "structural_type": list,
                "semantic_types": (
                    'http://schema.org/Integer',
                    'https://metadata.datadrivendiscovery.org/types/Attribute',
                    AUGMENTED_COLUMN_SEMANTIC_TYPE
                ),
            }
            return_metadata = return_metadata.update(selector=selector_base + (ALL_ELEMENTS, i + 2), metadata=each_metadata)

        return return_metadata

    def _generate_metadata_shape_part(self, value, selector, supplied_data=None) -> dict:
        """
        recursively generate all metadata for shape part, return a dict
        :param value: the input data
        :param selector: a tuple which indicate the selector
        :return: a dict with key as the selector, value as the metadata
        """
        if supplied_data is None:
            supplied_data = self.supplied_data
        generated_metadata = dict()
        generated_metadata['schema'] = CONTAINER_SCHEMA_VERSION
        if isinstance(value, d3m_Dataset):  # type: ignore
            generated_metadata['id'] = supplied_data.metadata.query(())['id']
            generated_metadata['name'] = supplied_data.metadata.query(())['name']
            generated_metadata['location_uris'] = supplied_data.metadata.query(())['location_uris']
            generated_metadata['digest'] = supplied_data.metadata.query(())['digest']
            generated_metadata['description'] = supplied_data.metadata.query(())['description']
            generated_metadata['source'] = supplied_data.metadata.query(())['source']
            generated_metadata['version'] = supplied_data.metadata.query(())['version']
            generated_metadata['structural_type'] = supplied_data.metadata.query(())['structural_type']
            generated_metadata['dimension'] = {
                'name': 'resources',
                'semantic_types': ['https://metadata.datadrivendiscovery.org/types/DatasetResource'],
                'length': len(value),
            }

            metadata_dict = collections.OrderedDict([(selector, generated_metadata)])

            for k, v in value.items():
                metadata_dict.update(self._generate_metadata_shape_part(v, selector + (k,)))

            # It is unlikely that metadata is equal across dataset resources, so we do not try to compact metadata here.

            return metadata_dict

        if isinstance(value, d3m_DataFrame):  # type: ignore
            generated_metadata['semantic_types'] = ['https://metadata.datadrivendiscovery.org/types/Table']
            generated_metadata['structural_type'] = d3m_DataFrame
            generated_metadata['dimension'] = {
                'name': 'rows',
                'semantic_types': ['https://metadata.datadrivendiscovery.org/types/TabularRow'],
                'length': value.shape[0],
            }

            metadata_dict = collections.OrderedDict([(selector, generated_metadata)])

            # Reusing the variable for next dimension.
            generated_metadata = {
                'dimension': {
                    'name': 'columns',
                    'semantic_types': ['https://metadata.datadrivendiscovery.org/types/TabularColumn'],
                    'length': value.shape[1],
                },
            }

            selector_all_rows = selector + (ALL_ELEMENTS,)
            metadata_dict[selector_all_rows] = generated_metadata
            return metadata_dict

    def _generate_metadata_column_part_for_general(self, data, metadata_return, return_format,
                                                   augment_resource_id) -> DataMetadata:
        """
        Inner function used to generate metadata for general search
        """
        # part for adding each column's metadata
        for i, each_column in enumerate(data[augment_resource_id]):
            if return_format == "ds":
                metadata_selector = (augment_resource_id, ALL_ELEMENTS, i)
            elif return_format == "df":
                metadata_selector = (ALL_ELEMENTS, i)
            structural_type = data[augment_resource_id][each_column].dtype.name
            if "int" in structural_type:
                structural_type = int
            elif "float" in structural_type:
                structural_type = float
            else:
                structural_type = str
            metadata_each_column = {"name": each_column,
                                    "structural_type": structural_type,
                                    'semantic_types': ("http://schema.org/Text",
                                                       'https://metadata.datadrivendiscovery.org/types/Attribute',
                                                       )
                                    }
            metadata_return = metadata_return.update(metadata=metadata_each_column, selector=metadata_selector)

        if return_format == "ds":
            metadata_selector = (augment_resource_id, ALL_ELEMENTS, i + 1)
        elif return_format == "df":
            metadata_selector = (ALL_ELEMENTS, i + 1)

        metadata_joining_pairs = {"name": "joining_pairs",
                                  "structural_type": typing.List[int],
                                  'semantic_types': ("http://schema.org/Integer",)
                                  }
        metadata_return = metadata_return.update(metadata=metadata_joining_pairs, selector=metadata_selector)

        return metadata_return
    
    def get_node_name(self, node_code) -> str:
        """
        Function used to get the properties(P nodes) names with given P node
        :param node_code: a str indicate the P node (e.g. "P123")
        :return: a str indicate the P node label (e.g. "inception")
        """
        sparql_query = "SELECT DISTINCT ?x WHERE \n { \n" + \
                       "wd:" + node_code + " rdfs:label ?x .\n FILTER(LANG(?x) = 'en') \n} "

        results = self.wikidata_cache_manager.get_result(sparql_query)
        if results:
            return results[0]['x']['value']
        else:
            self._logger.error("Getting name of node " + node_code + " failed!")
            return node_code

    def get_simple_view(self):
        """
        function used to see what found inside this search result class in a human vision
        contains information for search result's title, columns and join hints
        :return: a pandas DataFrame
        """
        if self.search_type == "wikidata":
            column_names = []
            for each in self.search_result["p_nodes_needed"]:
                each_name = self.get_node_name(each)
                column_names.append(each_name)
            column_names = ", ".join(column_names)
            required_variable = list()
            required_variable.append(self.search_result["target_q_node_column_name"])
            result = pd.DataFrame({"title": "wikidata search result for "
                                            + self.search_result["target_q_node_column_name"],
                                   "columns": column_names, "join columns": required_variable, "score": self.score},
                                  index=[0])

        elif self.search_type == "general":
            title = self.search_result['title']['value']
            column_names = []
            current_column_number = 0
            temp_selector = (ALL_ELEMENTS, current_column_number)
            temp_meta = self.d3m_metadata.query(selector=temp_selector)
            while len(temp_meta) != 0:
                column_names.append(temp_meta['name'])
                current_column_number += 1
                temp_selector = (ALL_ELEMENTS, current_column_number)
                temp_meta = self.d3m_metadata.query(selector=temp_selector)

            column_names = ", ".join(column_names)
            join_columns = self.search_result['variableName']['value'] if "variableName" in self.search_result else ""

            result = pd.DataFrame(
                {"title": title, "columns": column_names, "join columns": join_columns, "score": self.score},
                index=[0])

        elif self.search_type == "wikifier":
            title = "wikifier"
            result = pd.DataFrame({"title": title, "columns": "", "join columns": "", "score": self.score},
                                  index=[0])

        elif self.search_type == "vector":
            # fetch the num of columns
            column_names = []
            # res_dict = DownloadManager.fetch_fb_embeddings([self.search_result['q_nodes_list'][0]])
            # length = len(list(res_dict.values())[0].split(','))
            length = self.d3m_metadata.query((ALL_ELEMENTS, ))['dimension']['length']
            for i in range(length):
                if i < 10:
                    s = '00' + str(i)
                elif i < 100:
                    s = '0' + str(i)
                else:
                    s = str(i)
                each_name = "vector_" + s + "_of_qnode_with_" + self.search_result["target_q_node_column_name"]
                column_names.append(each_name)
            column_names = ", ".join(column_names)
            required_variable = list()
            required_variable.append(self.search_result["target_q_node_column_name"])
            result = pd.DataFrame({"title": "vector search result for "
                                            + self.search_result["target_q_node_column_name"],
                                   "columns": column_names, "join columns": required_variable,
                                   "score": self.score, "number_of_vectors": self.search_result["number_of_vectors"]},
                                  index=[0])
        else:
            raise ValueError("Unknown search type with " + self.search_type)
        return result

    def generate_metadata_for_download_result(self, return_result, supplied_data) -> DataMetadata:
        self.set_supplied_data(supplied_data)
        if type(return_result) is d3m_DataFrame:
            return_format = "df"
            augment_resource_ids = None
            selector_base = ()
        elif type(return_result) is d3m_Dataset:
            return_format = "ds"
            augment_resource_ids = list(return_result.keys())
            selector_base = tuple([augment_resource_ids[0]])
            if len(augment_resource_ids) > 1:
                self._logger.error("Generating metadata for multi resource dataset is not implemented yet! "
                                   "Will only generate metadata for first one.")
        else:
            raise ValueError("Unknown download data format as " + str(type(return_result)))

        if self.search_type == "wikidata":
            return_result.metadata = self.generate_metadata_for_wikidata_search(selector_base=selector_base)
        elif self.search_type == "vector":
            return_result.metadata = self.generate_metadata_for_vector_search(selector_base=selector_base)
        elif self.search_type == "general":
            return_result.metadata = self.generate_metadata_for_general_search(selector_base=selector_base)
        else:
            raise ValueError("Unknown search type as " + str(self.search_type))
        # update dataset level metadata here
        metadata_shape_part_dict = self._generate_metadata_shape_part(value=return_result, selector=(),
                                                                      supplied_data=supplied_data)

        for each_selector, each_metadata in metadata_shape_part_dict.items():
            return_result.metadata = return_result.metadata.update(selector=each_selector,
                                                                   metadata=each_metadata)
            # return_result.metadata = self._generate_metadata_column_part_for_general(return_result,
            #                                                                          return_result.metadata,
            #                                                                          return_format,
            #                                                                          augment_resource_id=augment_resource_ids[0])

        return return_result

    def generate_metadata_for_augment_result(self, df_joined, return_format, supplied_data, augment_resource_id):
        # put d3mIndex at first column and in the descend order in that column
        self.set_supplied_data(supplied_data)
        columns_all = list(df_joined.columns)
        if 'd3mIndex' in df_joined.columns:
            if df_joined.columns[0] != "d3mIndex":
                old_index = columns_all.index('d3mIndex')
                columns_all.insert(0, columns_all.pop(old_index))
        else:
            self._logger.warning("No d3mIndex column found after datamart augment!!!")
        df_joined = df_joined[columns_all]

        metadata_dict_left = {}
        metadata_dict_right = {}
        # if self.search_type == "general":
        #     # if the search type is general, we need to generate the metadata dict here
        #     for i, each in enumerate(df_joined):
        #         # description = each['description']
        #         dtype = df_joined[each].dtype.name
        #         if "float" in dtype:
        #             semantic_types = (
        #                 "http://schema.org/Float",
        #                 "https://metadata.datadrivendiscovery.org/types/Attribute",
        #                 AUGMENTED_COLUMN_SEMANTIC_TYPE
        #             )
        #         elif "int" in dtype:
        #             semantic_types = (
        #                 "http://schema.org/Integer",
        #                 "https://metadata.datadrivendiscovery.org/types/Attribute",
        #                 AUGMENTED_COLUMN_SEMANTIC_TYPE
        #             )
        #         else:
        #             semantic_types = (
        #                 "http://schema.org/Text",
        #                 "https://metadata.datadrivendiscovery.org/types/Attribute",
        #                 AUGMENTED_COLUMN_SEMANTIC_TYPE
        #             )
        #
        #         each_meta = {
        #             "name": each,
        #             "structural_type": str,
        #             "semantic_types": semantic_types,
        #             # "description": description
        #         }
        #         metadata_dict_right[each] = frozendict.FrozenOrderedDict(each_meta)
        # else:
        #     # if from wikidata, we should have already generated it
        # metadata_dict_right = self.d3m_metadata
        i = 0

        while len(self.d3m_metadata.query((ALL_ELEMENTS, i))) != 0:
            each_column_meta = self.d3m_metadata.query((ALL_ELEMENTS, i))
            metadata_dict_right[each_column_meta["name"]] = each_column_meta
            i += 1

        if return_format == "df":
            try:
                left_df_column_length = supplied_data.metadata.query((ALL_ELEMENTS,))['dimension']['length']
            except Exception:
                traceback.print_exc()
                raise ValueError("Getting left metadata information failed!")
        elif return_format == "ds":
            left_df_column_length = supplied_data.metadata.query((self.res_id, ALL_ELEMENTS,))['dimension'][
                'length']
        else:
            raise ValueError("Unknown return format as " + str(return_format))

        # add the original metadata
        for i in range(left_df_column_length):
            if return_format == "df":
                each_selector = (ALL_ELEMENTS, i)
            elif return_format == "ds":
                each_selector = (self.res_id, ALL_ELEMENTS, i)
            each_column_meta = supplied_data.metadata.query(each_selector)
            metadata_dict_left[each_column_meta['name']] = each_column_meta

        metadata_new = DataMetadata()
        new_column_names_list = list(df_joined.columns)

        # update each column's metadata
        for i in range(len(new_column_names_list)):
            current_column_name = new_column_names_list[i]
            if return_format == "df":
                each_selector = (ALL_ELEMENTS, i)
            elif return_format == "ds":
                each_selector = (augment_resource_id, ALL_ELEMENTS, i)

            if current_column_name in metadata_dict_left:
                new_metadata_i = metadata_dict_left[current_column_name]
            elif current_column_name in metadata_dict_right:
                new_metadata_i = metadata_dict_right[current_column_name]
            else:
                new_metadata_i = {
                    "name": current_column_name,
                    "structural_type": str,
                    "semantic_types": ("http://schema.org/Text",
                                       "https://metadata.datadrivendiscovery.org/types/Attribute",),
                }

                if current_column_name.endswith("_wikidata"):
                    # add vector semantic type here
                    if current_column_name.startswith("vector_"):
                        new_metadata_i["semantic_types"] = ("http://schema.org/Float",
                                                            "https://metadata.datadrivendiscovery.org/types/Attribute",
                                                            )
                    else:
                        data = list(filter(None, df_joined.iloc[:, i].astype(str).dropna()))
                        if all(re.match(r'^Q\d+$', x) for x in data):
                            new_metadata_i["semantic_types"] = ("http://schema.org/Text",
                                                                "https://metadata.datadrivendiscovery.org/types/Attribute",
                                                                q_node_semantic_type
                                                                )
                else:
                    self._logger.warning("Please check!")
                    self._logger.warning("No metadata found for column No." + str(i) + "with name " + current_column_name)

            metadata_new = metadata_new.update(each_selector, new_metadata_i)
        return_result = None

        # start adding shape metadata for dataset
        if return_format == "ds":
            return_df = d3m_DataFrame(df_joined, generate_metadata=False)
            resources = {augment_resource_id: return_df}
            return_result = d3m_Dataset(resources=resources, generate_metadata=False)
            return_result.metadata = metadata_new
            metadata_shape_part_dict = self._generate_metadata_shape_part(value=return_result,
                                                                          selector=(),
                                                                          supplied_data=supplied_data)
            for each_selector, each_metadata in metadata_shape_part_dict.items():
                return_result.metadata = return_result.metadata.update(selector=each_selector,
                                                                       metadata=each_metadata)
        elif return_format == "df":
            return_result = d3m_DataFrame(df_joined, generate_metadata=False)
            return_result.metadata = metadata_new
            metadata_shape_part_dict = self._generate_metadata_shape_part(value=return_result,
                                                                          selector=(),
                                                                          supplied_data=supplied_data)
            for each_selector, each_metadata in metadata_shape_part_dict.items():
                return_result.metadata = return_result.metadata.update(selector=each_selector,
                                                                       metadata=each_metadata)

        return_result = check_and_correct_q_nodes_semantic_type(return_result)

        return return_result[1]

    # def generate_metadata_for_wikidata_download_result(self):
    #     metadata_new = DataMetadata()
    #     self.metadata = dict()
    #     # add remained attributes metadata
    #
    #     for each_column in range(0, return_df.shape[1] - 1):
    #         current_column_name = p_name_dict[return_df.columns[each_column]]
    #         if return_format == "df":
    #             each_selector = (ALL_ELEMENTS, each_column)
    #         elif return_format == "ds":
    #             each_selector = (AUGMENT_RESOURCE_ID, ALL_ELEMENTS, each_column)
    #         # here we do not modify the original data, we just add an extra "expected_semantic_types" to metadata
    #         metadata_each_column = {"name": current_column_name, "structural_type": str,
    #                                 'semantic_types': semantic_types_dict[return_df.columns[each_column]]}
    #         self.metadata[current_column_name] = metadata_each_column
    #         if generate_metadata:
    #             metadata_new = metadata_new.update(metadata=metadata_each_column, selector=each_selector)
    #
    #     # special for joining_pairs column
    #     if return_format == "df":
    #         each_selector = (ALL_ELEMENTS, return_df.shape[1] - 1)
    #     elif return_format == "ds":
    #         each_selector = (AUGMENT_RESOURCE_ID, ALL_ELEMENTS, return_df.shape[1] - 1)
    #     metadata_joining_pairs = {"name": "joining_pairs", "structural_type": typing.List[int],
    #                               'semantic_types': ("http://schema.org/Integer",)}
    #     if generate_metadata:
    #         metadata_new = metadata_new.update(metadata=metadata_joining_pairs, selector=each_selector)
    #
    #     # start adding shape metadata for dataset
    #     if return_format == "ds":
    #         return_df = d3m_DataFrame(return_df, generate_metadata=False)
    #         return_df = return_df.rename(columns=p_name_dict)
    #         resources = {AUGMENT_RESOURCE_ID: return_df}
    #         return_result = d3m_Dataset(resources=resources, generate_metadata=False)
    #         if generate_metadata:
    #             return_result.metadata = metadata_new
    #             metadata_shape_part_dict = self._generate_metadata_shape_part(value=return_result, selector=(),
    #                                                                           supplied_data=self.supplied_data)
    #             for each_selector, each_metadata in metadata_shape_part_dict.items():
    #                 return_result.metadata = return_result.metadata.update(selector=each_selector,
    #                                                                        metadata=each_metadata)
    #         # update column names to be property names instead of number
    #
    #     elif return_format == "df":
    #         return_result = d3m_DataFrame(return_df, generate_metadata=False)
    #         return_result = return_result.rename(columns=p_name_dict)
    #         if generate_metadata:
    #             return_result.metadata = metadata_new
    #             metadata_shape_part_dict = self._generate_metadata_shape_part(value=return_result, selector=(),
    #                                                                           supplied_data=self.supplied_data)
    #             for each_selector, each_metadata in metadata_shape_part_dict.items():
    #                 return_result.metadata = return_result.metadata.update(selector=each_selector,
    #                                                                        metadata=each_metadata)
    #     else:
    #         raise ValueError("Invalid return format was given as " + str(return_format))

    # def generate_metadata_for_vector(self):
    #
    #     metadata_new = DataMetadata()
    #     self.metadata = dict()
    #     # add remained attributes metadata
    #
    #     for each_column in range(0, return_df.shape[1] - 1):
    #         current_column_name = return_df.columns[each_column]
    #         if return_format == "df":
    #             each_selector = (ALL_ELEMENTS, each_column)
    #         elif return_format == "ds":
    #             each_selector = (AUGMENT_RESOURCE_ID, ALL_ELEMENTS, each_column)
    #         # here we do not modify the original data, we just add an extra "expected_semantic_types" to metadata
    #         metadata_each_column = {"name": current_column_name, "structural_type": float,
    #                                 'semantic_types': semantic_types_dict[current_column_name]}
    #         self.metadata[current_column_name] = metadata_each_column
    #         if generate_metadata:
    #             metadata_new = metadata_new.update(metadata=metadata_each_column, selector=each_selector)
    #
    #     # special for joining_pairs column
    #     if return_format == "df":
    #         each_selector = (ALL_ELEMENTS, each_column + 1)
    #     elif return_format == "ds":
    #         each_selector = (AUGMENT_RESOURCE_ID, ALL_ELEMENTS, each_column + 1)
    #     metadata_joining_pairs = {"name": "joining_pairs", "structural_type": typing.List[int],
    #                               'semantic_types': ("http://schema.org/Integer",)}
    #     if generate_metadata:
    #         metadata_new = metadata_new.update(metadata=metadata_joining_pairs, selector=each_selector)
    #
    #     # start adding shape metadata for dataset
    #     if return_format == "ds":
    #         return_df = d3m_DataFrame(return_df, generate_metadata=False)
    #         resources = {AUGMENT_RESOURCE_ID: return_df}
    #         return_result = d3m_Dataset(resources=resources, generate_metadata=False)
    #         if generate_metadata:
    #             return_result.metadata = metadata_new
    #             metadata_shape_part_dict = self._generate_metadata_shape_part(value=return_result, selector=(),
    #                                                                           supplied_data=self.supplied_data)
    #             for each_selector, each_metadata in metadata_shape_part_dict.items():
    #                 return_result.metadata = return_result.metadata.update(selector=each_selector,
    #                                                                        metadata=each_metadata)
    #         # update column names to be property names instead of number
    #
    #     elif return_format == "df":
    #         return_result = d3m_DataFrame(return_df, generate_metadata=False)
    #         if generate_metadata:
    #             return_result.metadata = metadata_new
    #             metadata_shape_part_dict = self._generate_metadata_shape_part(value=return_result, selector=(),
    #                                                                           supplied_data=self.supplied_data)
    #             for each_selector, each_metadata in metadata_shape_part_dict.items():
    #                 return_result.metadata = return_result.metadata.update(selector=each_selector,
    #                                                                        metadata=each_metadata)

    def transfer_semantic_type(self, datatype: str):
        """
        Inner function used to transfer the wikidata semantic type to D3M semantic type
        :param datatype: a str indicate the semantic type adapted from wikidata
        :return: a str indicate the semantic type for D3M
        """
        special_type_dict = {"http://www.w3.org/2001/XMLSchema#dateTime": "http://schema.org/DateTime",
                             "http://www.w3.org/2001/XMLSchema#decimal": "http://schema.org/Float",
                             "http://www.opengis.net/ont/geosparql#wktLiteral":
                                 "https://metadata.datadrivendiscovery.org/types/Location"
                             }
        default_type = "http://schema.org/Text"
        if datatype in special_type_dict:
            return special_type_dict[datatype]
        else:
            self._logger.warning("Not seen data type: ", datatype)
            self._logger.warning("Please check this new type!!!")
            return default_type
