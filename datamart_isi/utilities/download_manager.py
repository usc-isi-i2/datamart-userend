import requests
from datamart_isi import config
import pandas

WIKIDATA_URI_TEMPLATE = config.wikidata_uri_template
EM_ES_URL = config.em_es_url
EM_ES_INDEX = config.em_es_index
EM_ES_TYPE = config.em_es_type


class DownloadManager:
    @staticmethod
    def fetch_fb_embeddings(q_nodes_list, target_q_node_column_name):
        # add vectors columns in wikifier_res
        qnodes = list(filter(None, q_nodes_list))
        qnode_uris = [WIKIDATA_URI_TEMPLATE.format(qnode) for qnode in qnodes]
        # do elastic search
        num_of_try = int(len(qnode_uris)/1024) + 1 if len(qnode_uris) % 1024 != 0 else int(len(qnode_uris)/1024)
        res = dict()
        for i in range(num_of_try):
            query = {
                'query': {
                    'terms': {
                        'key.keyword': qnode_uris[1024*i:1024*i+1024]
                    }
                },
                "size": len(qnode_uris[1024*i:1024*i+1024])
            }
            url = '{}/{}/{}/_search'.format(EM_ES_URL, EM_ES_INDEX, EM_ES_TYPE)
            resp = requests.get(url, json=query)
            if resp.status_code == 200:
                result = resp.json()
                hits = result['hits']['hits']
                for hit in hits:
                    source = hit['_source']
                    _qnode = source['key'].split('/')[-1][:-1]
                    res[_qnode] = ",".join(source['value'])

        # change to dataframe
        return_df = pandas.DataFrame()
        for key, val in res.items():
            each_result = dict()
            each_result["q_node"] = key
            vectors = val.split(',')
            for i in range(len(vectors)):
                if i < 10:
                    s = '00' + str(i)
                elif i < 100:
                    s = '0' + str(i)
                else:
                    s = str(i)
                v_name = "vector_" + s + "_of_qnode_with_" + target_q_node_column_name
                each_result[v_name] = float(vectors[i])
            return_df = return_df.append(each_result, ignore_index=True)

        return return_df

