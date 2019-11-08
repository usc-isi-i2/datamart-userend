from copy import deepcopy
from d3m import container
import d3m.metadata.base as metadata_base
import io
import json
import logging
import os
import queue
import requests
import shutil
import pathlib
import tempfile
import threading
import time
import typing
from urllib.parse import urlparse, urlunparse
import websocket
import zipfile

import datamart


logger = logging.getLogger(__name__)


DEFAULT_URL = 'https://dsbox02.isi.edu:9000'


class WsQueryCursor(datamart.DatamartQueryCursor):
    """
    Query cursor implemented using a websocket.
    """
    def __init__(self, ws: websocket.WebSocket) -> None:
        self._ws = ws
        self._closed = False
        self._queue = queue.PriorityQueue()
        self._recv_thread = threading.Thread(target=self._recv)
        self._recv_thread.setDaemon(True)

    def _recv(self) -> None:
        while True:
            data = self._ws.recv_data()
            if not data:
                logger.info("Websocket closed by server")
                self._queue.put((float('nan'), None))
                break
            result = RESTSearchResult(json.loads(data))
            self._queue.put((result.score(), result))

    def get_next_page(
            self,
            *, limit: typing.Optional[int] = 20, timeout: int = None,
            ) -> typing.Optional[typing.Sequence['RESTSearchResult']]:
        if self._closed:
            return None

        logger.info("Getting results, limit=%r, timeout=%r, qsize=%r",
                    limit, timeout, self._queue.qsize())
        start = time.perf_counter()
        initial_timeout = timeout
        results = []
        while len(results) < limit and (timeout is None or timeout >= 0):
            try:
                result = self._queue.get(timeout=timeout)
            except queue.Empty:
                # Timeout reached, return results, possibly empty list
                logger.info("Timeout reached")
                break
            if result is None:
                # Special case: cursor reached the end
                if not results:
                    self._closed = True
                    logger.info("End of search results")
                    return None
            results.append(result)
            if timeout is not None:
                timeout = initial_timeout - time.perf_counter() + start
        return results


class RESTQueryCursor(datamart.DatamartQueryCursor):
    """
    Query cursor implemented over an HTTP query.
    """
    def __init__(
            self,
            url: str,
            query: dict,
            data: bytes = None,
            data_id: str = None,
            data_resource_id: str = None
        ) -> None:
        self._args = url, query, data
        self._mutex = threading.RLock()
        self._sent = False
        self._done = threading.Condition(self._mutex)
        self._results = None
        self._supplied_data_id = data_id
        self._supplied_resource_id = data_resource_id

    def _query(self, timeout: int = None) -> None:
        url, query, data = self._args
        self._args = None

        # Add timeout requirement to query
        query['timeout'] = timeout

        logger.info("Sending REST query to %s...", url)
        if data:
            res = requests.post(
                url,
                files={'data': data, 'query': json.dumps(query)}
            )
        else:
            res = requests.post(
                url,
                files={'query': json.dumps(query)}
            )
        if res.status_code != 200:
            msg = "Error from DataMart: %s %s" % (
                res.status_code, res.reason
            )
            logger.warning(msg)
            raise Exception(msg)

        # with self._mutex:
        self._results = []
        for result in res.json()['results']:
            # TODO: handle union results
            if result['augmentation']['type'] == 'union':
                continue
            result['supplied_id'] = self._supplied_data_id
            result['supplied_resource_id'] = self._supplied_resource_id
            self._results.append(
                RESTSearchResult(result)
            )
        self._results.sort(key=lambda r: r.score(), reverse=True)
        logger.info("REST query complete, %d results", len(self._results))
            # self._done.notify_all()

    def get_next_page(
            self,
            *, limit: typing.Optional[int] = 20, timeout: int = None,
            ) -> typing.Optional[typing.Sequence['RESTSearchResult']]:
        logger.info("Getting results, limit=%r, timeout=%r",
                    limit, timeout)
        with self._mutex:
            # If we haven't yet sent the query, send it
            if self._results is None:
                if not self._sent:
                    self._query(timeout)
                    # thread = threading.Thread(target=self._query,
                    #                           args=[timeout])
                    # thread.setDaemon(True)
                    # thread.start()
                    self._sent = True

            # If results are not available, wait
            # if not self._done.wait(timeout):
            #     logger.info("Timeout reached")
            #     return []

            # Return results
            if not self._results:
                logger.info("End of search results")
                return None
            else:
                res = self._results[:limit]
                del self._results[:limit]
                return res


class RESTDatamart(datamart.Datamart):
    def __init__(self, connection_url: str = None) -> None:
        connection_url = connection_url or DEFAULT_URL

        super(RESTDatamart, self).__init__(connection_url)

        if not (connection_url.startswith('http://') or
                connection_url.startswith('https://')):
            raise ValueError("Invalid URL: should be HTTP or HTTPS URL")
        if connection_url[-1] == '/':
            connection_url = connection_url[:-1]
        self.url = connection_url

    def search(self, query: datamart.DatamartQuery,
               ) -> datamart.DatamartQueryCursor:
        return self._search_stream(query, None, None)

    def search_with_data(self, query: datamart.DatamartQuery,
                         supplied_data: container.Dataset,
                         ) -> datamart.DatamartQueryCursor:
        return self._search_stream(query, supplied_data, None)

    def search_with_data_columns(
            self, query: datamart.DatamartQuery,
            supplied_data: container.Dataset,
            data_constraints: typing.List[datamart.TabularVariable],
    ) -> datamart.DatamartQueryCursor:
        return self._search_stream(query, supplied_data, data_constraints)

    def _search_rest(
            self, query: datamart.DatamartQuery,
            supplied_data: typing.Optional[container.Dataset],
            data_constraints: typing.Optional[
                typing.List[datamart.TabularVariable]
            ]) -> datamart.DatamartQueryCursor:
        return RESTQueryCursor(
            self.url + '/search',
            query_to_json(query, data_constraints),
            get_dataset_bytes(supplied_data),
            get_dataset_id(supplied_data),
            get_resource_id(supplied_data)
        )

    def _search_stream(
            self, query: datamart.DatamartQuery,
            supplied_data: typing.Optional[container.Dataset],
            data_constraints: typing.Optional[
                typing.List[datamart.TabularVariable]
            ]) -> datamart.DatamartQueryCursor:
        # TODO: handle datasets with more than 1 resource
        if supplied_data and len(supplied_data.keys()) > 1:
            raise Exception("Datasets with multiple resources are"
                            " not supported at this moment.")

        # Construct ws:// or wss:// URL
        url = urlparse(self.url)
        ws_scheme = {'https': 'wss', 'http': 'ws'}
        ws_url = urlunparse([ws_scheme[url.scheme], url.netloc,
                             url.path + '/search_stream',
                             url.params, url.query, ''])

        # Connect
        try:
            ws = websocket.create_connection(ws_url)
        except (IOError, websocket.WebSocketException):
            # Use HTTP
            return self._search_rest(query, supplied_data, data_constraints)

        # Send query
        logger.info("Connected to %s, sending query...", ws_url)
        ws.send(json.dumps(query_to_json(query, data_constraints)))

        # Send data
        if supplied_data:
            logger.info("Sending data...")
            # ws.send_binary(TODO)  # TODO: Send data
        else:
            ws.send(json.dumps({'data': None}))

        # Return cursor
        return WsQueryCursor(ws)


def get_dataset_bytes(data: typing.Optional[container.Dataset]):
    # updated v2019.10.24: change to send d3m dataset
    if data:
        result_id = data.metadata.query(())['id']
        with tempfile.TemporaryDirectory() as tmpdir:
                save_dir = os.path.join(str(tmpdir), result_id)
                absolute_path_part_length = len(str(save_dir))
                # print(save_dir)
                # sys.stdout.flush()
                data.save("file://" + save_dir + "/datasetDoc.json")
                # zip and send to client
                base_path = pathlib.Path(save_dir + '/')
                data_buf = io.BytesIO()
                filePaths = retrieve_file_paths(save_dir)

                zip_file = zipfile.ZipFile(data_buf, 'w')
                with zip_file:
                    # write each file seperately
                    for fileName in filePaths:
                        shorter_path = fileName[absolute_path_part_length:]
                        zip_file.write(fileName, shorter_path)
                data_buf.seek(0)
        # data_buf = io.StringIO()
        # data[get_resource_id(data)].to_csv(data_buf, index=False)
        # data_buf.seek(0)
        return data_buf
    return None


def retrieve_file_paths(dirName):
    # setup file paths variable
    filePaths = []
    # Read all directory, subdirectories and file lists
    for root, directories, files in os.walk(dirName):
        for filename in files:
            # Create the full filepath by using os module.
            filePath = os.path.join(root, filename)
            filePaths.append(filePath)
    # return all paths
    return filePaths


def get_dataset_id(data: typing.Optional[container.Dataset]) -> typing.Optional[str]:
    if data:
        return data.metadata.query(())['id']
    return None


def get_resource_id(data: typing.Optional[container.Dataset]) -> typing.Optional[str]:
    if data:
        return list(data.keys())[0]
    return None


def query_to_json(query: datamart.DatamartQuery,
                  data_constraints: typing.Optional[
                      typing.List[datamart.TabularVariable]
                  ]) -> dict:
    json_obj = dict()
    if query:
        if query.keywords:
            json_obj['keywords'] = query.keywords
        if query.variables:
            json_obj['variables'] = []
            for variable in query.variables:
                if isinstance(variable, datamart.NamedEntityVariable):
                    json_obj['variables'].append(
                        dict(
                            type='named_entity_variable',
                            entities=variable.entities
                        )
                    )
                elif isinstance(variable, datamart.TemporalVariable):
                    temporal_variable = dict(type='temporal_variable')
                    if variable.start:
                        temporal_variable['start'] =\
                            variable.start.strftime('%d-%b-%Y %H:%M:%S.%f')
                    if variable.end:
                        temporal_variable['end'] =\
                            variable.end.strftime('%d-%b-%Y %H:%M:%S.%f')
                    if variable.granularity:
                        temporal_variable['granularity'] = variable.granularity.value
                    json_obj['variables'].append(temporal_variable)
                elif isinstance(variable, datamart.GeospatialVariable):
                    geospatial_variable = dict(
                        type='geospatial_variable',
                        latitude1=variable.latitude1,
                        longitude1=variable.longitude1,
                        latitude2=variable.latitude2,
                        longitude2=variable.longitude2
                    )
                    if variable.granularity:
                        geospatial_variable['granularity'] = variable.granularity.value
                    json_obj['variables'].append(geospatial_variable)
                elif isinstance(variable, datamart.TabularVariable):
                    json_obj['variables'].append(
                        dict(
                            type='tabular_variable',
                            columns=[column.column_index for column in variable.columns],
                            relationship=variable.relationship.value
                        )
                    )
                else:
                    pass
    if data_constraints:
        if 'variables' not in json_obj:
            json_obj['variables'] = []
        for variable in data_constraints:
            json_obj['variables'].append(
                dict(
                    type='tabular_variable',
                    columns=[column.column_index for column in variable.columns],
                    relationship=variable.relationship.value
                )
            )
    return json_obj


class RESTSearchResult(datamart.DatamartSearchResult):
    def __init__(self, json_obj) -> None:
        self._json_obj = json_obj
        self._id = json_obj['id']
        self._supplied_data_id = json_obj['supplied_id']
        self._score = json_obj['score']
        self._metadata = json_to_metadata(json_obj['metadata'])
        self._augment_hint = json_to_augment_spec(
            json_obj['augmentation'],
            self._supplied_data_id,
            json_obj['supplied_resource_id'],
            self._id
        )

    def serialize(self) -> str:
        return json.dumps(self._json_obj, sort_keys=True)

    @classmethod
    def deserialize(cls, string: str) -> 'RESTSearchResult':
        return RESTSearchResult(json.loads(string))

    def id(self) -> str:
        return self._id

    def score(self) -> float:
        return self._score

    def get_augment_hint(self) -> datamart.AugmentSpec:
        return self._augment_hint

    def get_metadata(self) -> metadata_base.DataMetadata:
        logger.warning('The method "get_metadata" has not been '
                       'implemented yet! Sorry!')
        return self._metadata

    def get_json_metadata(self) -> dict:
        return self._json_obj

    def download(self, supplied_data: typing.Optional[container.Dataset],
                 *, connection_url: str = None,
                 ) -> container.Dataset:

        connection_url = connection_url or DEFAULT_URL

        files = dict(
            task=json.dumps(self._json_obj),
            format='d3m'.encode('utf-8')
        )
        if supplied_data:
            files['data'] = get_dataset_bytes(supplied_data)

        res = requests.post(
            connection_url + '/download',
            files=files,
            allow_redirects=True,
            stream=True
        )
        if res.status_code != 200:
            msg = "Error from DataMart: %s %s" % (
                res.status_code, res.reason
            )
            logger.warning(msg)
            raise Exception(msg)

        dataset = download_dataset(res)
        if dataset:
            return dataset
        raise Exception("Error while reading the HTTP response.")

    def augment(self, supplied_data: container.Dataset,
                augment_columns: typing.Optional[
                    typing.List[datamart.DatasetColumn]
                ] = None,
                *, connection_url: str = None,
                ) -> container.Dataset:

        connection_url = connection_url or DEFAULT_URL

        task = deepcopy(self._json_obj)
        supplied_data_id = get_dataset_id(supplied_data)
        if self._supplied_data_id != supplied_data_id:
            logger.warning("The requested supplied data (%s) "
                           "is different than the original supplied data (%s)."
                           % (supplied_data_id, self._supplied_data_id))
            task['augmentation'] = {
                'type': 'none',
                'left_columns': [],
                'right_columns': []
            }

        columns = []
        if augment_columns:
            columns = [column.column_index for column in augment_columns]

        res = requests.post(
            connection_url + '/augment',
            files={
                'data': get_dataset_bytes(supplied_data),
                'task': json.dumps(self._json_obj),
                'columns': json.dumps(columns)
            },
            allow_redirects=True,
            stream=True
        )

        if res.status_code != 200:
            msg = "Error from DataMart: %s %s" % (
                res.status_code, res.reason
            )
            logger.warning(msg)
            raise Exception(msg)

        dataset = download_dataset(res)
        if dataset:
            dataset = fix_metadata(dataset, supplied_data)
            return dataset
        raise Exception("Error while reading the HTTP response.")


def json_to_augment_spec(
        json_obj: dict,
        supplied_dataset_id: str,
        supplied_resource_id: str,
        datamart_dataset_id: str
        ) -> typing.Optional[datamart.AugmentSpec]:

    if not supplied_dataset_id:
        return None

    left_columns = []
    right_columns = []
    for column_group in json_obj['left_columns']:
        left_columns.append([])
        for column in column_group:
            left_columns[-1].append(
                datamart.DatasetColumn(supplied_resource_id, column)
            )
    # TODO: which resource id to use for a DataMart dataset?
    for column_group in json_obj['right_columns']:
        right_columns.append([])
        for column in column_group:
            right_columns[-1].append(
                datamart.DatasetColumn('0', column)
            )
    return datamart.TabularJoinSpec(
        supplied_dataset_id,
        datamart_dataset_id,
        left_columns,
        right_columns
    )


def json_to_metadata(json_obj: dict) -> metadata_base.DataMetadata:
    # TODO: implement -- not sure how to create a DataMetadata obj?
    return metadata_base.DataMetadata()


def download_dataset(
        response: requests.Response) -> container.Dataset:

    # Download D3M ZIP to temporary file
    fd, tmpfile = tempfile.mkstemp(prefix='datamart_download_',
                                   suffix='.d3m.zip')
    destination = None
    try:
        with open(tmpfile, 'wb') as f:
            for chunk in response.iter_content(chunk_size=4096):
                if chunk:  # filter out keep-alive chunks
                    f.write(chunk)
        # Unzip
        destination = tempfile.mkdtemp(prefix='datamart_download_')
        zip = zipfile.ZipFile(tmpfile)
        zip.extractall(destination)

        dataset = container.Dataset.load(
            'file://' + destination + '/datasetDoc.json'
        )
    except Exception as e:
        logger.warning("Error: %s" % str(e))
        dataset = None
    finally:
        os.close(fd)
        os.remove(tmpfile)
        if destination:
            shutil.rmtree(destination)

    return dataset


def fix_metadata(
        dataset: container.Dataset,
        supplied_data: container.Dataset
        ) -> container.Dataset:

    dataset_resource_id = get_resource_id(dataset)
    supplied_data_resource_id = get_resource_id(supplied_data)

    new_metadata = metadata_base.DataMetadata()
    new_metadata = new_metadata.update(
        selector=(),
        metadata=dataset.metadata.query(())
    )
    new_metadata = new_metadata.update(
        selector=([dataset_resource_id, ]),
        metadata=dataset.metadata.query((dataset_resource_id,))
    )
    new_metadata = new_metadata.update(
        selector=([dataset_resource_id, metadata_base.ALL_ELEMENTS]),
        metadata=dataset.metadata.query(
            (dataset_resource_id, metadata_base.ALL_ELEMENTS)
        )
    )

    # keeping metadata about previously existing columns
    number_cols = supplied_data.metadata.query_field(
        (supplied_data_resource_id, metadata_base.ALL_ELEMENTS),
        'dimension'
    )['length']
    for i in range(number_cols):
        new_metadata = new_metadata.update(
            selector=([dataset_resource_id, metadata_base.ALL_ELEMENTS, i]),
            metadata=supplied_data.metadata.query(
                (supplied_data_resource_id, metadata_base.ALL_ELEMENTS, i)
            )
        )

    # now bringing back metadata from new columns
    number_augment_cols = dataset.metadata.query_field(
        (dataset_resource_id, metadata_base.ALL_ELEMENTS),
        'dimension'
    )['length']
    for i in range(number_cols, number_augment_cols):
        new_metadata = new_metadata.update(
            selector=([dataset_resource_id, metadata_base.ALL_ELEMENTS, i]),
            metadata=dataset.metadata.query(
                (dataset_resource_id, metadata_base.ALL_ELEMENTS, i)
            )
        )

    # loading new metadata
    dataset.metadata = new_metadata

    return dataset
