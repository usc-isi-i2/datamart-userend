import typing
import pandas as pd
import os
import logging
from d3m.metadata.base import ALL_ELEMENTS
from datamart_isi.config import cache_file_storage_base_loc
from datamart_isi.utilities import connection
from datamart_isi.cache.wikidata_cache import QueryCache
from dsbox.datapreprocessing.cleaner.data_profile import Profiler, Hyperparams as ProfilerHyperparams
from dsbox.datapreprocessing.cleaner.cleaning_featurizer import CleaningFeaturizer, CleaningFeaturizerHyperparameter


_logger = logging.getLogger(__name__)
seed_dataset_store_location = os.path.join(cache_file_storage_base_loc, "datasets_cache")
WIKIDATA_CACHE_MANAGER = QueryCache()
WIKIDATA_SERVER = connection.get_wikidata_server_url()


class Utils:
    DEFAULT_DESCRIPTION = {
        "materialization": {
            "python_path": "default_materializer"
        },
        "variables": []
    }

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
            results = WIKIDATA_CACHE_MANAGER.get_result(sparql_query)
            return results[0]['x']['value']
        except:
            return node_code

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
        from datamart_isi.profilers.basic_profiler import BasicProfiler, GlobalMetadata

        global_metadata = GlobalMetadata.construct_global(description=cls.DEFAULT_DESCRIPTION)
        global_metadata = BasicProfiler.basic_profiling_entire(global_metadata=global_metadata, data=data)
        metadata_dict = global_metadata.value

        # for col_offset in range(data.shape[1]):
        #     variable_metadata = BasicProfiler.basic_profiling_column(
        #         description={},
        #         variable_metadata=VariableMetadata.construct_variable(description={}),
        #         column=data.iloc[:, col_offset]
        #     )
        #     global_metadata.add_variable_metadata(variable_metadata)
        hyper1 = ProfilerHyperparams.defaults()
        hyper2 = CleaningFeaturizerHyperparameter.defaults()
        clean_f = CleaningFeaturizer(hyperparams=hyper2)
        profiler = Profiler(hyperparams=hyper1)
        profiled_df = profiler.produce(inputs=data).value
        clean_f.set_training_data(inputs=profiled_df)
        clean_f.fit()
        cleaned_df = clean_f.produce(inputs=profiled_df).value
        cleaned_df_metadata = cleaned_df.metadata

        for i in range(data.shape[1]):
            each_column_metadata = cleaned_df_metadata.query((ALL_ELEMENTS, i))
            column_name = data.columns[i]
            if "datetime" in data.iloc[:, i].dtype.name:
                semantic_type = ("http://schema.org/DateTime", 'https://metadata.datadrivendiscovery.org/types/Attribute')
            else:
                semantic_type = each_column_metadata['semantic_types']
            variable_metadata = {'datamart_id': None,
                                 'semantic_type': semantic_type,
                                 'name': column_name,
                                 'description': 'column name: {}, dtype: {}'.format(column_name, cleaned_df.iloc[:, i].dtype.name)
                                 }
            metadata_dict['variables'].append(variable_metadata)

        if original_meta:
            metadata_dict.update(original_meta)
        return metadata_dict
