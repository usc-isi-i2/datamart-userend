import typing
import pandas as pd
import os
import requests
import json
import sys
import argparse
import logging
import copy
from functools import wraps
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

        if len(time_column.unique()) == 1:
            allow_duplicate_amount = 0
        else:
            allow_duplicate_amount = 1

        time_granularity = 'second'
        if any(time_column.dt.minute != 0) and len(time_column.dt.minute.unique()) > allow_duplicate_amount:
            time_granularity = 'minute'
        elif any(time_column.dt.hour != 0) and len(time_column.dt.hour.unique()) > allow_duplicate_amount:
            time_granularity = 'hour'
        elif any(time_column.dt.day != 0) and len(time_column.dt.day.unique()) > allow_duplicate_amount:
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
        elif any(time_column.dt.month != 0) and len(time_column.dt.month.unique()) > allow_duplicate_amount:
            time_granularity = 'month'
        elif any(time_column.dt.year != 0) and len(time_column.dt.year.unique()) > allow_duplicate_amount:
            time_granularity = 'year'
        else:
            _logger.error("Can't guess the time granularity for this dataset! Will use as second")
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
        """
        a simple dict map which map time granulairty string to wikidata int format
        :param granularity_str:
        :return:
        """
        TemporalGranularity = {
            'seconds': 14,
            'minutes': 13,
            'hours': 12,
            'days': 11,
            'weeks': 11,  # now also use week as days
            'months': 10,
            'years': 9,
            'unspecified': 8,

        }
        if granularity_str.lower() in TemporalGranularity:
            return TemporalGranularity[granularity_str.lower()]
        else:
            raise ValueError("Can't find corresponding granularity value.")

    @staticmethod
    def time_granularity_value_to_stringfy_time_format(granularity_int: int) -> str:
        try:
            granularity_int = int(granularity_int)
        except ValueError:
            raise ValueError("The given granulairty is not int format!")

        granularity_dict = {
            14: "%Y-%m-%d %H:%M:%S",
            13: "%Y-%m-%d %H:%M",
            12: "%Y-%m-%d %H",
            11: "%Y-%m-%d",
            10: "%Y-%m",
            9: "%Y"

        }
        if granularity_int in granularity_dict:
            return granularity_dict[granularity_int]
        else:
            _logger.warning("Unknown time granularity value as {}! Will use second level.".format(str(granularity_int)))
            return granularity_dict[14]

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
                if s[0] <= time <= s[1]:
                    return True
        else:
            return False

    @staticmethod
    def keywords_augmentation(keywords: typing.List[str], server_address: str = None) -> typing.List[str]:
        """
        function that use fuzzy search to get more related keywords
        :param server_address: the request server address
        :param keywords: a list of keywords
        :return:
        """
        if not server_address:
            server_address = connection.get_keywords_augmentation_server_url()
        url = server_address + "/" + ",".join(keywords)
        resp = requests.get(url)
        if resp.status_code // 100 == 2:
            new_keywords = json.loads(resp.text)['message'].split(",")
            _logger.info("Get augmented keywords as {}".format(str(new_keywords)))
        else:
            new_keywords = keywords
            _logger.warning("Failed on augmenting keywords! Please check the service condition!")
        return new_keywords

    @staticmethod
    def qgram_tokenizer(x, _q):
        if len(x) < _q:
            return [x]
        return [x[i:i + _q] + "*" for i in range(len(x) - _q + 1)]

    @staticmethod
    def trigram_tokenizer(x):
        return Utils.qgram_tokenizer(x, 3)

    @staticmethod
    def join_datasets_by_files(files: typing.List[typing.Union[str, pd.DataFrame]], how: str = "left") -> pd.DataFrame:
        """
        :param how: the method to join the dataframe, {‘left’, ‘right’, ‘outer’, ‘inner’}, default ‘left’
            How to handle the operation of the two objects.
            left: use calling frame’s index (or column if on is specified)
            right: use other’s index.
            outer: form union of calling frame’s index (or column if on is specified) with other’s index, and sort it. lexicographically.
            inner: form intersection of calling frame’s index (or column if on is specified) with other’s index, preserving the order of the calling’s one.
        :param files: either a path to csv or a DataFrame directly
        :return: a joined DataFrame object
        """
        if not isinstance(files, list):
            raise ValueError("Input must be a list of files")
        if len(files) < 2:
            raise ValueError("Input files amount must be larger than 2")
        _logger.info("Totally {} files.".format(str(len(files))))

        necessary_column_names = {"region_wikidata", "precision", "time"}
        ignore_column_names = {"region_wikidata", "precision", "time", "variable_name", "variable", "region_Label", "calendar",
                               "productLabel", "qualityLabel"}
        loaded_dataframes = []
        for i, each in enumerate(files):
            if isinstance(each, str):
                try:
                    temp_loaded_df = pd.read_csv(each)
                except Exception as e:
                    _logger.warning("Failed on loading dataframe No.{}".format(str(i)))
                    _logger.error(str(e))
                    continue
            elif isinstance(each, pd.DataFrame):
                temp_loaded_df = each
            else:
                _logger.warning("Unsupported format '{}' on No.{} input, will ignore.".format(str(type(each)), str(i)))
                continue

            temp_loaded_df = temp_loaded_df.dropna(subset=['region_wikidata'], inplace=False)
            if len(set(temp_loaded_df.columns.tolist()).intersection(necessary_column_names)) != len(necessary_column_names):
                _logger.error("Following columns {} are necessary to be exists".format(str(necessary_column_names)))
                raise ValueError("Not all columns found on given No.{} datasets.")
            loaded_dataframes.append(temp_loaded_df)

        # use first input df as base df
        output_df = copy.deepcopy(loaded_dataframes[0])
        source_precision = output_df['precision'].iloc[0]
        # transfer the datetime format to ensure format match
        time_stringfy_format = Utils.time_granularity_value_to_stringfy_time_format(source_precision)
        output_df['time'] = pd.to_datetime(output_df['time']).dt.strftime(
            time_stringfy_format)

        for i, each_loaded_df in enumerate(loaded_dataframes[1:]):
            current_precision = each_loaded_df['precision'].iloc[0]
            if source_precision != current_precision:
                left_join_columns = ["region_wikidata"]
                right_join_columns = ["region_wikidata"]
            else:
                left_join_columns = ["region_wikidata", "time"]
                right_join_columns = ["region_wikidata", "time"]
                each_loaded_df['time'] = pd.to_datetime(each_loaded_df['time']).dt.strftime(time_stringfy_format)
            possible_name = []
            for each_col_name in each_loaded_df.columns:
                if each_col_name not in ignore_column_names and "label" not in each_col_name.lower():
                    possible_name.append(each_col_name)
            if len(possible_name) != 1:
                _logger.error("get multiple possible name???")
                _logger.error(str(each_loaded_df.columns))
                _logger.error("???")
                # import pdb
                # pdb.set_trace()
            right_needed_columns = right_join_columns + [possible_name[0]]
            print(str(right_needed_columns))
            right_join_df = each_loaded_df[right_needed_columns]
            output_df = pd.merge(left=output_df, right=right_join_df,
                                 left_on=left_join_columns, right_on=right_join_columns,
                                 how=how)
            if len(output_df) == 0:
                _logger.error("Get 0 rows after join with No.{} DataFrame".format(str(i + 1)))
        return output_df


def main(argv: typing.Sequence) -> None:
    parser = argparse.ArgumentParser(prog='datamart_isi', description="Run ISI datamart utils command.")
    subparsers = parser.add_subparsers(dest='commands', title='commands')
    # define join parser
    join_parser = subparsers.add_parser(
        'join', help="join ethiopia related datasets directly by commands",
        description="Join datasets",
    )
    join_parser.add_argument(
        '-d', '--dataset_dirs', action='store', dest='datasets_dirs',
        help="paths to the datasets",
    )
    join_parser.add_argument(
        '-o', '--output_dir', action='store', dest='output_dir',
        help="paths to the output file",
    )
    arguments = parser.parse_args(argv[1:])

    _logger.debug("given arguments are {}".format(str(arguments)))
    _logger.info("Running {} function.".format(arguments.commands))
    if arguments.commands == 'join':
        processed_dirs = arguments.datasets_dirs.split(",")
        _logger.info("Join from following files")
        for i, each_dir in enumerate(processed_dirs):
            if each_dir[0] == " ":
                each_dir = each_dir[1:]
                processed_dirs[i] = each_dir
            _logger.info(each_dir)
        res = Utils.join_datasets_by_files(files=processed_dirs)
        res.to_csv(arguments.output_dir, index=False)


if __name__ == '__main__':
    main(sys.argv)
