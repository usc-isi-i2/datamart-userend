from SPARQLWrapper import SPARQLWrapper, JSON, POST, URLENCODED
import memcache
import logging
import hashlib
import pickle
import datetime
import typing
from datamart_isi import config
from datamart_isi.utilities import connection
from datamart_isi.utilities.singleton import singleton
MEMCAHCE_MAX_VALUE_SIZE = config.memcache_max_value_size


@singleton
class QueryCache(object):
    def __init__(self, *, memcache_max_value_size=MEMCAHCE_MAX_VALUE_SIZE):
        self._logger = logging.getLogger(__name__)
        self.memcache_server = connection.get_memcache_server_url()
        self.wikidata_server = connection.get_wikidata_server_url()
        self._logger.debug("Current memcache server url is: " + self.memcache_server)
        self._logger.debug("Current wikidata server url is: " + self.wikidata_server)
        try:
            self.mc = memcache.Client([self.memcache_server], debug=True, server_max_value_length=memcache_max_value_size)
            self._logger.info("Start memcache connection to " + self.memcache_server + " success!")
        except Exception as e:
            self.mc = None
            self._logger.error("Start memcache connection to " + self.memcache_server + " failed!")
            self._logger.debug(e, exc_info=True)

        self.qm = SPARQLWrapper(self.wikidata_server)
        self.qm.setReturnFormat(JSON)
        self.qm.setMethod(POST)
        self.qm.setRequestMethod(URLENCODED)

    def get_result(self, query: str) -> typing.Optional[typing.List]:
        """
        The main function used to get a result either from cache or run query from wikidata server
        :param query: a sparql query in str format
        :return: query results in list format if success get the result, otherwise None
        """
        hash_key = self.get_hash_key(query)
        results = self.get_cache_result(hash_key)
        cache_hit = False
        if results is not None:
            try:
                results = pickle.loads(results)
                cache_hit = True
                self._logger.info("Cache hit! will use this results.")
                return results
            except Exception as e:
                self._logger.warning("Hit results are broken! Need to rerun the query!")
                self._logger.debug(e, exc_info=True)
        else:
            self._logger.info("Cache not hit, will run general query.")

        if not cache_hit:
            results = self.run_sparql_query(query, hash_key)
            if results is not None:
                response = self.add_to_memcache(query, hash_key, results)
                if not response:
                    self._logger.warning("Pushing some of the query failed! Please check!")
            else:
                self._logger.warning("No query result returned, will skip adding to memcache.")
            return results

    def run_sparql_query(self, query, hash_tag) -> typing.Optional[typing.List]:
        """
        Function used to call query manager and run the query
        :param query: the sparql query
        :param hash_tag: the hash tag of the sparql query
        :return: the query results returned from wikidata if success, otherwise None
        """
        self._logger.debug("Start running wikidata query on " + self.wikidata_server)
        try:
            self.qm.setQuery(query)
            results = self.qm.query().convert()['results']['bindings']
            self._logger.debug("Running wikidata query success!")
        except Exception as e:
            self._logger.error("Query for " + hash_tag + " failed!")
            self._logger.debug(e, exc_info=True)
            results = None
        return results

    def get_cache_result(self, hash_key) -> typing.Optional[bytes]:
        """
        Function used to check whether this query hash tag exist in memcache system or not
        :param hash_key: the hash key of the query
        :return: the cached query in pickled format if the hash key hit, otherwise None
        """
        if self.mc is not None:
            # check whether we have cache or not
            results = self.mc.get("results_" + hash_key)
            return results
        else:
            self._logger.info("No memcache server connected, skip cache searching.")
            return None

    def remove_from_memcache(self, hash_key):
        """
        Function used to remove corresponding query
        :param hash_key:
        :return:
        """
        fail_count = 0
        for each_key in ["query_", "timestamp_", "results_"]:
            memcache_key = each_key + hash_key
            response = self.mc.delete(memcache_key)
            if response == 0:
                self._logger.warning("Delete key " + memcache_key + " from memcache server failed!")
                fail_count += 1
        self._logger.info("Delete key " + hash_key + " from memcache server finished! Totally " + str(fail_count) + " failed.")

    def add_to_memcache(self, query, hash_key, query_result) -> bool:
        """
        Function used to add the query results to memcache server
        :param query: the sparql query
        :param hash_key: hash key of the sparql query
        :param query_result: the query results from wikidata
        :return: a bool represent the pushing is success or not
        """
        try:
            self._logger.debug("Start pushing the search result of wikidata to " + self.memcache_server)
            # add query results
            response_code1 = self.mc.set("results_" + hash_key, pickle.dumps(query_result))
            if not response_code1:
                self._logger.warning("Pushing wikidata search result failed! Maybe the size too big?")

            # add timestamp to let the system know when to update
            response_code2 = self.mc.set("timestamp_" + hash_key, str(datetime.datetime.now().timestamp()))
            if not response_code2:
                self._logger.warning("Pushing timestamp failed! What happened???")

            # add original query for further updating
            response_code3 = self.mc.set("query_" + hash_key, query)
            if not response_code3:
                self._logger.warning("Pushing query failed! Maybe the size too big?")

            # only return True if all success
            if response_code1 and response_code2 and response_code3:
                self._logger.info("Pushing search result success!")
                return True
            else:
                return False
        except Exception as e:
            self._logger.error("Pushing results of query hash key" + hash_key + " failed!")
            self._logger.debug(e, exc_info=True)
            return False

    def get_hash_key(self, query: str) -> str:
        """
        Function used to get the hash key of the query string, each query is stored with the hash key to prevent over length
        :param query: a str indicate the sparql query
        :return: a str of the hash to the query
        """
        hash_generator = hashlib.md5()
        hash_generator.update(query.encode('utf-8'))
        hash_key = hash_generator.hexdigest()
        self._logger.debug("Current query's hash tag is " + hash_key)
        return hash_key
