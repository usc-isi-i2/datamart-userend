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

    @staticmethod
    def get_time_granularity(time_column: pd.DataFrame) -> str:
        if "datetime" not in time_column.dtype.name:
            try:
                time_column = pd.to_datetime(time_column)
            except:
                raise ValueError("Can't parse given time column!")
        if any(time_column.dt.second != 0):
            time_granularity = 'second'
        elif any(time_column.dt.minute != 0):
            time_granularity = 'minute'
        elif any(time_column.dt.hour != 0):
            time_granularity = 'hour'
        elif any(time_column.dt.day != 0):
            # it is also possible weekly data
            is_weekly_data = True
            time_column_sorted = time_column.sort_values()
            temp1 = time_column_sorted.iloc[0]
            for i in range(1, len(time_column_sorted)):
                temp2 = time_column_sorted.iloc[i]
                if (temp2 - temp1).days != 7:
                    is_weekly_data = False
                    break
            if is_weekly_data:
                time_granularity = 'week'
            else:
                time_granularity = 'day'
        elif any(time_column.dt.month != 0):
            time_granularity = 'month'
        elif any(time_column.dt.year != 0):
            time_granularity = 'year'
        else:
            raise ValueError("Can't guess the time granularity for this dataset!")
        return time_granularity

    @staticmethod
    def map_granularity_to_d3m_format(granularity: str):
        """
        d3m allowed following granularities:
        "timeGranularity":{"type":"dict", "required":false, "schema":{
            "value":{"type":"number", "required":true},
            "units":{"type":"string", "required":true, "allowed":[
                "seconds",
                "minutes",
                "days",
                "weeks",
                "years",
                "unspecified"
            ]
        }
        :param granularity:
        :return: a list follow d3m format
        """
        if "second" in granularity:
            return [('value', 1), ('unit', 'seconds')]
        elif "minute" in granularity:
            return [('value', 1), ('unit', 'minutes')]
        elif "hour" in granularity:
            return [('value', 1), ('unit', 'hours')]
        elif "day" in granularity:
            return [('value', 1), ('unit', 'days')]
        elif "week" in granularity:
            return [('value', 1), ('unit', 'weeks')]
        # how about months??
        elif "month" in granularity:
            return [('value', 1), ('unit', 'months')]
        elif "year" in granularity:
            return [('value', 1), ('unit', 'years')]
        else:
            raise ValueError("Unrecognized granularity")

    @staticmethod
    def map_granularity_to_value(granularity_str: str) -> int:
        TemporalGranularity = {
            'second': 14,
            'minute': 13,
            'hour': 12,
            'day': 11,
            'month': 10,
            'year': 9
        }
        if granularity_str.lower() in TemporalGranularity:
            return TemporalGranularity[granularity_str.lower()]
        else:
            raise ValueError("Can't find corresponding granularity value.")

    @staticmethod
    def map_d3m_granularity_to_value(granularity_str: str) -> int:
        TemporalGranularity = {
            'unspecified': 15,
            'seconds': 14,
            'minutes': 13,
            'hours': 12,
            'days': 11,
            'months': 10,
            'years': 9

        }
        if granularity_str.lower() in TemporalGranularity:
            return TemporalGranularity[granularity_str.lower()]
        else:
            raise ValueError("Can't find corresponding granularity value.")

    @staticmethod
    def overlap(first_inter, second_inter) -> bool:
        """
        function used to check whether two time intervals has overlap
        :param first_inter: [start_time, end_time]
        :param second_inter: [start_time, end_time]
        :return: a bool value indicate has overlap or not
        """
        for f, s in ((first_inter, second_inter), (second_inter, first_inter)):
            # will check both ways
            for time in (f[0], f[1]):
                if s[0] < time < s[1]:
                    return True
        else:
            return False
