
import datetime
import typing
import pandas as pd
import copy
import random
import frozendict
import collections
import typing
from d3m.container import DataFrame as d3m_DataFrame
from d3m.container import Dataset as d3m_Dataset
import d3m.metadata.base as metadata_base
# from datamart.dataset import Dataset
from datamart.utilities.utils import PRODUCTION_ES_INDEX, SEARCH_URL
from datamart.es_managers.json_query_manager import JSONQueryManager
from datamart.new_query.augment import Augment
# old Augment 
# from datamart.augment import Augment
# from datamart.data_loader import DataLoader
from d3m.base import utils as d3m_utils
from datamart.utilities.utils import Utils
# from datamart.joiners.join_result import JoinResult
# from datamart.joiners.joiner_base import JoinerType
# from itertools import chain
from datamart.joiners.rltk_joiner import RLTKJoiner
from SPARQLWrapper import SPARQLWrapper, JSON, POST, URLENCODED
from d3m.metadata.base import DataMetadata, ALL_ELEMENTS
from datamart.joiners.rltk_joiner import RLTKJoiner_new
from wikifier import config
# import requests
import traceback
import logging
import datetime
import enum

from d3m import container
import d3m.metadata.base as metadata_base
from d3m import utils

__all__ = ('DatamartQueryCursor', 'Datamart', 'DatasetColumn', 'DatamartSearchResult', 'AugmentSpec',
           'TabularJoinSpec', 'UnionSpec', 'TemporalGranularity', 'GeospatialGranularity', 'ColumnRelationship', 'DatamartQuery',
           'VariableConstraint', 'NamedEntityVariable', 'TemporalVariable', 'GeospatialVariable', 'TabularVariable')


class DatamartQueryCursor(object):
    """
    Cursor to iterate through Datamarts search results.
    """

    @abc.abstractmethod
    def get_next_page(self, *, limit: typing.Optional[int] = 20, timeout: int = None) -> typing.Optional[typing.Sequence['DatamartSearchResult']]:
        """
        Return the next page of results. The call will block until the results are ready.

        Note that the results are not ordered; the first page of results can be returned first simply because it was
        found faster, but the next page might contain better results. The caller should make sure to check
        `DatamartSearchResult.score()`.

        Parameters
        ----------
        limit : int or None
            Maximum number of search results to return. None means no limit.
        timeout : int
            Maximum number of seconds before returning results. An empty list might be returned if it is reached.

        Returns
        -------
        Sequence[DatamartSearchResult] or None
            A list of `DatamartSearchResult's, or None if there are no more results.
        """
        pass


class Datamart(object):
    """
    All Datamarts must implement this abstract class.
    """

    def __init__(self, connection_url: str) -> None:
        self.url = connection_url
        self._logger = logging.getLogger(__name__)
        query_server = config.endpoint_query_main
        self.augmenter = Augment(endpoint=query_server)

    def set_test_mode(self) -> None:
        query_server = config.endpoint_query_test
        self.augmenter = Augment(endpoint=query_server)

    def search(self, query: 'DatamartQuery') -> DatamartQueryCursor:
        """This entry point supports search using a query specification.

        The query specification supports querying datasets by keywords, named entities, temporal ranges, and geospatial ranges.

        Datamart implementations should return a DatamartQueryCursor immediately.

        Parameters
        ----------
        query : DatamartQuery
            Query specification.

        Returns
        -------
        DatamartQueryCursor
            A cursor pointing to search results.
        """
        print("Not implemented yet")
        pass

    def search_with_data(self, query: 'DatamartQuery', supplied_data: container.Dataset) -> DatamartQueryCursor:
        """
        Search using on a query and a supplied dataset.

        This method is a "smart" search, which leaves the Datamart to determine how to evaluate the relevance of search
        result with regard to the supplied data. For example, a Datamart may try to identify named entities and date
        ranges in the supplied data and search for companion datasets which overlap.

        To manually specify query constraints using columns of the supplied data, use the `search_with_data_columns()`
        method and `TabularVariable` constraints.

        Datamart implementations should return a DatamartQueryCursor immediately.

        Parameters
        ------_---
        query : DatamartQuery
            Query specification
        supplied_data : container.Dataset
            The data you are trying to augment.

        Returns
        -------
        DatamartQueryCursor
            A cursor pointing to search results containing possible companion datasets for the supplied data.
        """

        # first take a search on wikidata
        wikidata_search_results = self.search_wiki_data(query_json, supplied_data)
        search_results.extend(wikidata_search_results)
        limit_remained = limit - len(wikidata_search_results)

        if query is None:
            # if not query given, try to find the Text columns from given dataframe and use it to find some candidates
            can_query_columns = []
            for each in range(len(supplied_dataframe.columns)):
                if type(supplied_data) is d3m_Dataset:
                    selector = (res_id, ALL_ELEMENTS, each)
                else:
                    selector = (ALL_ELEMENTS, each)
                each_column_meta = supplied_data.metadata.query(selector)
                if 'http://schema.org/Text' in each_column_meta["semantic_types"]:
                    # or "https://metadata.datadrivendiscovery.org/types/CategoricalData" in each_column_meta["semantic_types"]:
                    can_query_columns.append(each)

            # import pdb
            # pdb.set_trace()

            if len(can_query_columns) == 0:
                self._logger.warning("No columns can be augment!")
                return search_results

            results_no_query = []
            for each_column in can_query_columns:
                tabular_variable = TabularVariable(columns=[each_column], relationship=None)
                results_no_query.extend(self.search_with_data_columns(query=None, supplied_data=supplied_data, data_constraints=tabular_variable))

            # we will return the results of each searching query one by one
            # for example: [res_q1_1,res_q1_2,res_q1_3], [res_q2_1,res_q2_2,res_q2_3] , [res_q3_1,res_q3_2,res_q3_3]
            # will return as: [res_q1_1, res_q2_1, res_q3_1, res_q1_2, res_q2_2, res_q3_3...]
            results_rescheduled = []
            has_remained = True
            while has_remained:
                has_remained = False
                for each in results_no_query:
                    if len(each) > 0:
                        has_remained = True
                        results_rescheduled.append(each.pop(0))
            # append together to wikidata search results
            search_results.extend(results_rescheduled)

            return search_results

    def search_with_data_columns(self, query: 'DatamartQuery', supplied_data: container.Dataset,
                                 data_constraints: typing.List['TabularVariable']) -> DatamartQueryCursor:
        """
        Search using a query which can include constraints on supplied data columns (TabularVariable).

        This search is similar to the "smart" search provided by `search_with_data()`, but caller must manually specify
        constraints using columns from the supplied data; Datamart will not automatically analyze it to determine
        relevance or joinability.

        Use of the query spec enables callers to compose their own "smart search" implementations.

        Datamart implementations should return a DatamartQueryCursor immediately.

        Parameters
        ------_---
        query : DatamartQuery
            Query specification
        supplied_data : container.Dataset
            The data you are trying to augment.
        data_constraints : list
            List of `TabularVariable` constraints referencing the supplied data.

        Returns
        -------
        DatamartQueryCursor
            A cursor pointing to search results containing possible companion datasets for the supplied data.
        """

        # put the enetities of all given columns from "data_constraints" into the query's variable part and run the query
        all_query_variables = ""
        for each_column in data_constraints.columns:
            column_values = supplied_dataframe.iloc[:, each_column]
            query_column_entities = list(set(column_values.tolist()))

            if len(query_column_entities) > MAX_ENTITIES_LENGTH:
                query_column_entities = random.sample(query_column_entities, MAX_ENTITIES_LENGTH)

            for i in range(len(query_column_entities)):
                query_column_entities[i] = str(query_column_entities[i])

            query_column_entities = " ".join(query_column_entities)

        all_query_variables += query_column_entities + " "
        search_query = DatamartQuery(variables=query_column_entities)
        query_json = search_query.to_json()

        # sort to put best results at top
        search_results = self.augmenter.query_by_sparql(json_query=query_json)
        search_results.sort(key=lambda x: x.score, reverse=True)
        return search_results

class DatasetColumn:
    """
    Specify a column of a dataframe in a D3MDataset
    """

    def __init__(self, resource_id: str, column_index: int) -> None:
        self.resource_id = resource_id
        self.column_index = column_index


class DatamartSearchResult(abc.ABC):
    """
    This class represents the search results of a Datamart search.
    Different Datamarts will provide different implementations of this class.

    All implementations of DatamartSearchResult should be picklable.
    """

    @abc.abstractmethod
    def score(self) -> float:
        """
        Returns a non-negative score of the search result.
        Larger scores indicate better matches. Scores across Datamart implementations are not comparable.
        """
        pass

    @abc.abstractmethod
    def download(self, supplied_data: typing.Optional[container.Dataset]) -> container.Dataset:
        """
        Produces a D3M dataset (data plus metadata) corresponding to the search result.
        Every time the download method is called on a search result, it will produce the exact same columns
        (as specified in the metadata -- get_metadata), but the set of rows may depend on the supplied_data.
        Datamart is encouraged to return a dataset that joins well with the supplied data, e.g., has rows that match
        the entities in the supplied data. Datamarts may ignore the supplied_data and return the same data regardless.

        If the supplied_data is None, Datamarts may return None or a default dataset, based on the search query.

        Parameters
        ---------
        supplied_data : container.Dataset
            A D3M dataset containing the dataset that is the target for augmentation. Datamart will try to download data
            that augments the supplied data well.
        """
        pass

    @abc.abstractmethod
    def augment(self, supplied_data: container.Dataset, augment_columns: typing.Optional[typing.List[DatasetColumn]] = None) -> container.Dataset:
        """
        Produces a D3M dataset that augments the supplied data with data that can be retrieved from this search result.
        The augment methods is a baseline implementation of download plus augment.

        Callers who want to control over the augmentation process should use the download method and use their own
        augmentation algorithm.

        Parameters
        ---------
        supplied_data : container.Dataset
            A D3M dataset containing the dataset that is the target for augmentation.
        augment_columns : typing.List[DatasetColumn]
            If provided, only the specified columns from the Datamart dataset that will be added to the supplied dataset.
        """
        pass

    @abc.abstractmethod
    def get_metadata(self) -> metadata_base.DataMetadata:
        """
        Access the metadata of the dataset.

        Returns
        -------
        DataMetadata
            The Datamart metadata of the dataset.
        """
        pass

    @abc.abstractmethod
    def get_augment_hint(self) -> 'AugmentSpec':
        """
        Returns specification for augmenting supplied data with the data that can be downloaded using this search result.
        """
        pass


class AugmentSpec:
    """
    Abstract class for D3M augmentation specifications
    """


class TabularJoinSpec(AugmentSpec):
    """
    A join spec specifies a possible way to join a left dataset with a right dataset. The spec assumes that it may
    be necessary to use several columns in each datasets to produce a key or fingerprint that is useful for joining
    datasets. The spec consists of two lists of column identifiers or names (left_columns, left_column_names and
    right_columns, right_column_names).

    In the simplest case, both left and right are singleton lists, and the expectation is that an appropriate
    matching function exists to adequately join the datasets. In some cases equality may be an appropriate matching
    function, and in some cases fuzz matching is required. The join spec does not specify the matching function.

    In more complex cases, one or both left and right lists contain several elements. For example, the left list
    may contain columns for "city", "state" and "country" and the right dataset contains an "address" column. The join
    spec pairs up ["city", "state", "country"] with ["address"], but does not specify how the matching should be done
    e.g., combine the city/state/country columns into a single column, or split the address into several columns.
    """
    def __init__(self, left_resource_id: str, right_resource_id: str, left_columns: typing.List[typing.List[DatasetColumn]],
                 right_columns: typing.List[typing.List[DatasetColumn]]) -> None:
        self.left_resource_id = left_resource_id
        self.right_resource_id = right_resource_id
        self.left_columns = left_columns
        self.right_columns = right_columns
        # we can have list of the joining column pairs
        # each list inside left_columns/right_columns is a candidate joining column for that dataFrame
        # each candidate joining column can also have multiple columns


class UnionSpec(AugmentSpec):
    """
    A union spec specifies how to combine rows of a dataframe in the left dataset with a dataframe in the right dataset.
    The dataframe after union should have the same columns as the left dataframe.

    Implementation: TBD
    """
    pass


class TemporalGranularity(utils.Enum):
    YEAR = 1
    MONTH = 2
    DAY = 3
    HOUR = 4
    SECOND = 5


class GeospatialGranularity(utils.Enum):
    COUNTRY = 1
    STATE = 2
    COUNTY = 3
    CITY = 4
    POSTAL_CODE = 5


class ColumnRelationship(utils.Enum):
    CONTAINS = 1
    SIMILAR = 2
    CORRELATED = 3
    ANTI_CORRELATED = 4
    MUTUALLY_INFORMATIVE = 5
    MUTUALLY_UNINFORMATIVE = 6


class DatamartQuery:
    """
    A Datamart query consists of two parts:

    * A list of keywords.

    * A list of required variables. A required variable specifies that a matching dataset must contain a variable
      satisfying the constraints provided in the query. When multiple required variables are given, the matching
      dataset should contain variables that match each of the variable constraints.

    The matching is fuzzy. For example, when a user specifies a required variable spec using named entities, the
    expectation is that a matching dataset contains information about the given named entities. However, due to name,
    spelling, and other differences it is possible that the matching dataset does not contain information about all
    the specified entities.

    In general, Datamart will do a best effort to satisfy the constraints, but may return datasets that only partially
    satisfy the constraints.
    """
    def __init__(self, keywords: typing.List[str], variables: typing.List['VariableConstraint']) -> None:
        self.keywords = keywords
        self.variables = variables


class VariableConstraint(abc.ABC):
    """
    Abstract class for all variable constraints.
    """


class NamedEntityVariable(VariableConstraint):
    """
    Specifies that a matching dataset must contain a variable including the specified set of named entities.

    For example, if the entities are city names, the expectation is that a matching dataset must contain a variable
    (column) with the given city names. Due to spelling differences and incompleteness of datasets, the returned
    results may not contain all the specified entities.

    Parameters
    ----------
    entities : List[str]
        List of strings that should be contained in the matched dataset column.
    """
    def __init__(self, entities: typing.List[str]) -> None:
        self.entities = entities


class TemporalVariable(VariableConstraint):
    """
    Specifies that a matching dataset should contain a variable with temporal information (e.g., dates) satisfying
    the given constraint.

    The goal is to return a dataset that covers the requested temporal interval and includes
    data at a requested level of granularity.

    Datamart will return best effort results, including datasets that may not fully cover the specified temporal
    interval or whose granularity is finer or coarser than the requested granularity.

    Parameters
    ----------
    start : datetime
        A matching dataset should contain a variable with temporal information that starts earlier than the given start.
    end : datetime
        A matching dataset should contain a variable with temporal information that ends after the given end.
    granularity : TemporalGranularity
        A matching dataset should provide temporal information at the requested level of granularity.
    """
    def __init__(self, start: datetime.datetime, end: datetime.datetime, granularity: TemporalGranularity = None) -> None:
        self.start = start
        self.end = end
        self.granularity = granularity


class GeospatialVariable(VariableConstraint):
    """
    Specifies that a matching dataset should contain a variable with geospatial information that covers the given
    bounding box.

    A matching dataset may contain variables with latitude and longitude information (in one or two columns) that
    cover the given bounding box.

    Alternatively, a matching dataset may contain a variable with named entities of the given granularity that provide
    some coverage of the given bounding box. For example, if the bounding box covers a 100 mile square in Southern
    California, and the granularity is City, the result should contain Los Angeles, and other cities in Southern
    California that intersect with the bounding box (e.g., Hawthorne, Torrance, Oxnard).

    Parameters
    ----------
    latitude1 : float
        The latitude of the first point
    longitude1 : float
        The longitude of the first point
    latitude2 : float
        The latitude of the second point
    longitude2 : float
        The longitude of the second point
    granularity : GeospatialGranularity
        Requested geospatial values are well matched with the requested granularity
    """
    def __init__(self, latitude1: float, longitude1: float, latitude2: float, longitude2: float, granularity: GeospatialGranularity = None) -> None:
        self.latitude1 = latitude1
        self.longitude1 = longitude1
        self.latitude2 = latitude2
        self.longitude2 = longitude2
        self.granularity = granularity


class TabularVariable(object):
    """
    Specifies that a matching dataset should contain variables related to given columns in the supplied_dataset.

    The relation ColumnRelationship.CONTAINS specifies that string values in the columns overlap using the string
    equality comparator. If supplied_dataset columns consists of temporal or spatial values, then
    ColumnRelationship.CONTAINS specifies overlap in temporal range or geospatial bounding box, respectively.

    The relation ColumnRelationship.SIMILAR specifies that string values in the columns overlap using fuzzy string matching.

    The relations ColumnRelationship.CORRELATED and ColumnRelationship.ANTI_CORRELATED specify the columns are
    correlated and anti-correlated, respectively.

    The relations ColumnRelationship.MUTUALLY_INFORMATIVE and ColumnRelationship.MUTUALLY_UNINFORMATIVE specify the columns
    are mutually and anti-correlated, respectively.

    Parameters:
    -----------
    columns : typing.List[int]
        Specify columns in the dataframes of the supplied_dataset
    relationship : ColumnRelationship
        Specifies how the the columns in the supplied_dataset are related to the variables in the matching dataset.
    """
    def __init__(self, columns: typing.List[DatasetColumn], relationship: ColumnRelationship) -> None:
        self.columns = columns
        self.relationship = relationship
