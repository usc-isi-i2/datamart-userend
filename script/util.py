'''
Utility scripts
'''
import argparse
import copy
import logging
import sys
import typing

import pandas as pd

_logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)


def time_granularity_value_to_stringfy_time_format(granularity_int: int) -> str:
    try:
        granularity_int = int(granularity_int)
    except ValueError:
        raise ValueError("The given granularity is not int format!")

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
    loaded_filenames = []
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

        if len(set(temp_loaded_df.columns.tolist()).intersection(necessary_column_names)) != len(necessary_column_names):
            _logger.error("Following columns {} are necessary to be exists".format(str(necessary_column_names)))
            raise ValueError("Not all columns found on given No.{} datasets {}.".format(str(i), each))
        loaded_dataframes.append(temp_loaded_df)
        loaded_filenames.append(each)

    # use first input df as base df
    output_df = copy.deepcopy(loaded_dataframes[0])
    # drop_columns = []
    # for col_name in ["productLabel", "qualityLabel"]:
    #     if col_name in output_df:
    #         drop_columns.append(col_name)
    # if drop_columns:
    #     output_df = output_df.drop(drop_columns, axis=1)
    possible_name = []
    for each_col_name in output_df.columns:
        if each_col_name not in ignore_column_names and "label" not in each_col_name.lower():
            possible_name.append(each_col_name)
    if len(possible_name) != 1:
        _logger.error("get multiple possible name???")
        _logger.error(str(output_df.columns))

    source_precision = output_df['precision'].iloc[0]
    output_df = output_df[["region_wikidata", "time", possible_name[0]]]
    output_df = output_df.dropna()
    output_df = output_df.drop_duplicates()
    

    # transfer the datetime format to ensure format match
    time_stringfy_format = time_granularity_value_to_stringfy_time_format(source_precision)
    output_df['time'] = pd.to_datetime(output_df['time']).dt.strftime(
        time_stringfy_format)

    for i, (each_loaded_df, filename) in enumerate(zip(loaded_dataframes[1:], loaded_filenames[1:])):
        _logger.debug('Joining %d of %d: %s', i, len(loaded_dataframes)-1, filename)
        each_loaded_df = each_loaded_df.dropna()
        each_loaded_df = each_loaded_df.drop_duplicates()        
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
        # print(str(right_needed_columns))
        right_join_df = each_loaded_df[right_needed_columns]
        _logger.debug('left shape: %s', output_df.shape)
        _logger.debug('right shape: %s', right_join_df.shape)

        output_df = pd.merge(left=output_df, right=right_join_df,
                             left_on=left_join_columns, right_on=right_join_columns,
                             how=how)
        output_df = output_df.drop_duplicates()
        # output_df.to_csv(f'partial_{i}.csv', index=False)
        if len(output_df) == 0:
            _logger.error("Get 0 rows after join with No.{} DataFrame".format(str(i + 1)))
    return output_df


def join(files: typing.List[str]):
    result = join_datasets_by_files(files)
    try:
        print(result.to_csv(index=False))
    except BrokenPipeError:
        pass

    
def main() -> None:
    parser = argparse.ArgumentParser(prog='UTIL',
                                     description="Run ISI datamart utility command.")
    subparsers = parser.add_subparsers(dest='command', title='command')
    # define join parser
    join_parser = subparsers.add_parser(
        'join',
        help="join ethiopia related CSV datasets directly",
        description="Join CSV datasets",
    )
    join_parser.add_argument(
        'csv_files', nargs='+', help="paths to the csv datasets",
    )

    args = parser.parse_args()
    _logger.info("Running {} function.".format(args.command))
    _logger.debug("given args are {}".format(str(args)))


    if args.command == 'join':
        join(args.csv_files)
    
if __name__ == '__main__':
    main()
