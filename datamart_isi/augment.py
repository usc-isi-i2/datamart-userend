import pandas as pd
import typing
import traceback
import logging
from d3m.base import utils as d3m_utils
from datamart_isi.utilities.utils import Utils
from d3m.container import Dataset as d3m_Dataset
from datamart_isi.utilities import connection
from SPARQLWrapper import SPARQLWrapper, JSON, POST, URLENCODED
from wikifier.utils import remove_punctuation
from datamart_isi.utilities.geospatial_related import GeospatialRelated
from datamart_isi.utilities.singleton import singleton
from datamart_isi.cache.wikidata_cache import QueryCache
from datamart_isi import config


@singleton
class Augment(object):
    def __init__(self) -> None:
        self.qm = SPARQLWrapper(connection.get_general_search_server_url())
        self.qm.setReturnFormat(JSON)
        self.qm.setMethod(POST)
        self.qm.setRequestMethod(URLENCODED)
        self.logger = logging.getLogger(__name__)
        self.wikidata_cache_manager = QueryCache()

    def query_by_sparql(self, query: dict, dataset: d3m_Dataset = None, **kwargs) -> typing.Optional[typing.List[dict]]:
        """
        Args:
            query: a dictnary format query
            dataset: a d3m format dataset for reference
            **kwargs: some extra parameters may get

        Returns:

        """
        if query:
            query_body = self.parse_sparql_query(query, dataset, **kwargs)
            self.logger.debug("Query sent to datamart blazegraph is:")
            self.logger.debug(query_body)
            try:
                self.qm.setQuery(query_body)
                results = self.qm.query().convert()['results']['bindings']
            except Exception as e:
                self.logger.error(e, exc_info=True)
                traceback.print_exc()
                return []
            return results
        else:
            self.logger.error("No query given, query failed!")
            return []

    def parse_sparql_query(self, json_query, dataset, **kwargs) -> str:
        """
        parse the json query to a spaqrl query format
        :param json_query:
        :param dataset: supplied dataset
        :return: a string indicate the sparql query
        """
        # example of query variables: Chaves Los Angeles Sacramento
        PREFIX = '''
            prefix ps: <http://www.wikidata.org/prop/statement/>
            prefix pq: <http://www.wikidata.org/prop/qualifier/>
            prefix p: <http://www.wikidata.org/prop/>
        '''
        SELECTION = '''
            SELECT ?dataset ?datasetLabel ?variableName ?variable ?score ?rank ?url ?file_type ?title ?start_time ?end_time ?time_granularity ?keywords ?extra_information
        '''
        STRUCTURE = '''
            WHERE {
                ?dataset rdfs:label ?datasetLabel.
                ?dataset p:P2699/ps:P2699 ?url.
                ?dataset p:P2701/ps:P2701 ?file_type.
                ?dataset p:C2010/ps:C2010 ?extra_information.
                ?dataset p:C2005 ?variable.
                ?variable ps:C2005 ?variableName.
                ?dataset p:P1476 ?title_url.
                ?title_url ps:P1476 ?title .
                ?dataset p:C2004 ?keywords_url.
                ?keywords_url ps:C2004 ?keywords.
        '''
        spaqrl_query = PREFIX + SELECTION + STRUCTURE
        need_keywords_search = "keywords_search" in json_query.keys() and json_query["keywords_search"] != []
        need_variables_search = "variables" in json_query.keys() and json_query["variables"] != {}
        need_temporal_search = "variables_search" in json_query.keys() and \
                               "temporal_variable" in json_query["variables_search"].keys()
        need_geospatial_search = "variables_search" in json_query.keys() and \
                                 "geospatial_variable" in json_query["variables_search"].keys()
        need_consider_wikifier_columns_only = kwargs.get("consider_wikifier_columns_only", False)
        need_augment_with_time = kwargs.get("augment_with_time", False)
        limit_amount = kwargs.get("limit_amount", config.default_search_limit)
        bind = ""
        ORDER = "ORDER BY DESC(?score) ?title"
        LIMIT = "LIMIT " + str(limit_amount)

        if need_variables_search:
            query_variables = json_query['variables']
            query_part = " ".join(query_variables.values())
            spaqrl_query += '''
                ?variable pq:C2006 [
                            bds:search """''' + query_part + '''""" ;
                            bds:relevance ?score_var ;
                          ].
                '''
            bind = "?score_var" if bind == "" else bind + "+ ?score_var"

        if need_keywords_search:
            query_keywords = json_query["keywords_search"]
            # update v2019.12.13: use keywords augmentation
            query_keywords = Utils.keywords_augmentation(query_keywords)

            # updated v2019.11.1, for search_without_data, we should remove duplicates
            if dataset is None:
                SELECTION = '''
                            SELECT DISTINCT ?dataset ?datasetLabel ?score ?rank ?url ?file_type ?title ?keywords ?extra_information
                            '''
                spaqrl_query = PREFIX + SELECTION + STRUCTURE
            else:
                # updated v2019.11.1, now use fuzzy search
                _, supplied_dataframe = d3m_utils.get_tabular_resource(dataset=dataset, resource_id=None)
                # column_names_list = supplied_dataframe.columns.tolist()
                # for each_col_name in supplied_dataframe.columns:
                #     query_keywords.extend(remove_punctuation(each_col_name, "list"))

            # trigram_keywords = []
            # for each_keyword in query_keywords:
                # trigram_keywords.extend(Utils.trigram_tokenizer(each_keyword))

            query_keywords_filtered = set()
            for each_keyword in query_keywords:
                query_keywords_filtered.add(each_keyword.lower())

            query_part = " ".join(list(query_keywords_filtered))

            spaqrl_query += '''
                optional {
                ?keywords_url ps:C2004 [
                                bds:search """''' + query_part + '''""" ;
                                bds:relevance ?score_key1 ;
                              ].
                }
                
                optional {
                ?title_url ps:P1476 [
                                bds:search """''' + query_part + '''""" ;
                                bds:relevance ?score_key2 ;
                              ].
                }
                '''
            if bind == "":
                bind = "IF(BOUND(?score_key1), ?score_key1, 0) + IF(BOUND(?score_key2), ?score_key2, 0)"
            else:
                bind += "+ IF(BOUND(?score_key1), ?score_key1, 0) + IF(BOUND(?score_key2), ?score_key2, 0)"

        if need_temporal_search:
            tv = json_query["variables_search"]["temporal_variable"]
            start_date = pd.to_datetime(tv["start"]).isoformat()
            end_date = pd.to_datetime(tv["end"]).isoformat()
            granularity = Utils.map_granularity_to_value(tv["granularity"])
            spaqrl_query += '''
                ?variable pq:C2013 ?time_granularity .
                ?variable pq:C2011 ?start_time .
                ?variable pq:C2012 ?end_time .
                FILTER(?time_granularity >= ''' + str(granularity) + ''')
                FILTER(!((?start_time > "''' + end_date + '''"^^xsd:dateTime) || (?end_time < "''' + \
                            start_date + '''"^^xsd:dateTime)))
                '''

        if need_geospatial_search:
            geo_variable = json_query["variables_search"]["geospatial_variable"]
            qnodes = self.parse_geospatial_query(geo_variable)
            if qnodes:
                # find similar dataset from datamart
                query_part = " ".join(qnodes)
                # query_part = "q1494 q1400 q759 q1649 q1522 q1387 q16551" # COMMENT: for testing
                spaqrl_query += '''
                                    ?variable pq:C2006 [
                                        bds:search """''' + query_part + '''""" ;
                                        bds:relevance ?score_geo ;
                                    ].
                                 '''
                bind = "?score_geo" if bind == "" else bind + "+ ?score_geo"

        # if "title_search" in json_query.keys() and json_query["title_search"] != '':
        #     query_title = json_query["title_search"]
        #     spaqrl_query += '''
        #         ?title_url ps:P1476 [
        #                   bds:search """''' + query_title + '''""" ;
        #                   bds:relevance ?score_title ;
        #                 ].
        #     '''
        #     bind = "?score_title" if bind == "" else bind + "+ ?score_title"
        if bind:
            spaqrl_query += "\n BIND((" + bind + ") AS ?score) "

        if need_keywords_search:
            spaqrl_query += """
                BIND((IF(BOUND(?score_key1), ?score_key1, 0) + IF(BOUND(?score_key2), ?score_key2, 0)) AS ?score_keywords)
                filter (?score_keywords != 0)
                """

        if need_augment_with_time:
            time_info = json_query["variables_time"]
            spaqrl_query += '''
                ?dataset p:C2005 ?variable2.
                ?variable2 pq:C2011 ?start_time.
                ?variable2 pq:C2012 ?end_time.
                ?variable2 pq:C2013 ?time_granularity.
                FILTER(?time_granularity >= ''' + str(time_info['granularity']) + ''')
                FILTER(!((?start_time > "''' + \
                            time_info['start'] + \
                            '''"^^xsd:dateTime) || (?end_time < "''' + \
                            time_info['end'] + '''"^^xsd:dateTime)))
            '''

        spaqrl_query += "\n }" + "\n" + ORDER + "\n" + LIMIT

        return spaqrl_query

    def parse_geospatial_query(self, geo_variable):
        geo_gra_dict = {'country': 'Q6256', 'state': 'Q7275', 'city': 'Q515', 'county': 'Q28575',
                        'postal_code': 'Q37447'}
        qnodes = set()

        # located inside a bounding box
        if "latitude1" in geo_variable.keys() and "latitude2" in geo_variable.keys():
            geo1_related = GeospatialRelated(float(geo_variable["latitude1"]), float(geo_variable["longitude1"]))
            geo1_related.coordinate_transform()  # axis transformation
            geo2_related = GeospatialRelated(float(geo_variable["latitude2"]), float(geo_variable["longitude2"]))
            geo2_related.coordinate_transform()
            # find top left point and bottom right point
            top_left_point, botm_right_point = geo1_related.distinguish_two_points(geo2_related)
            granularity = geo_gra_dict[geo_variable["granularity"]]

            if top_left_point and botm_right_point:
                # get Q nodes located inside a geospatial bounding box from wikidata query
                sparql_query = "select distinct ?place where \n{\n  ?place wdt:P31/wdt:P279* wd:" + granularity + " .\n" \
                               + "SERVICE wikibase:box {\n ?place wdt:P625 ?location .\n" \
                               + "bd:serviceParam wikibase:cornerWest " + "\"Point(" + str(
                    top_left_point[0]) + " " + str(top_left_point[1]) + ")\"^^geo:wktLiteral .\n" \
                               + "bd:serviceParam wikibase:cornerEast " + "\"Point(" + str(
                    botm_right_point[0]) + " " + str(botm_right_point[1]) + ")\"^^geo:wktLiteral .\n}\n" \
                               + "SERVICE wikibase:label { bd:serviceParam wikibase:language \"en\" }\n}\n"
                results = self.wikidata_cache_manager.get_result(sparql_query)
                if results:
                    for each in results:
                        value = each["place"]["value"]
                        value = value.split('/')[-1]
                        qnodes.add(value)

        return qnodes

    def get_dataset_time_information(self, dataset_id: str):

        query_body = """
        prefix ps: <http://www.wikidata.org/prop/statement/>
        prefix pq: <http://www.wikidata.org/prop/qualifier/>
        prefix p: <http://www.wikidata.org/prop/>
        
        SELECT ?dataset ?score ?title ?start_time ?end_time ?time_granularity
        
        WHERE {
            values ?dataset { <http://www.wikidata.org/entity/""" + dataset_id + """> }
        
          ?dataset p:C2005 ?variable.
          ?variable pq:C2013 ?time_granularity .
          ?variable pq:C2011 ?start_time .
          ?variable pq:C2012 ?end_time .
        }"""
        try:
            self.qm.setQuery(query_body)
            results = self.qm.query().convert()['results']['bindings']
        except Exception as e:
            self.logger.error(e, exc_info=True)
            traceback.print_exc()
            return []
        return results
