import wikifier
import pandas as pd
import copy
import frozendict
import logging
from d3m.base import utils as d3m_utils
from d3m.container import Dataset as d3m_Dataset
from d3m.container import DataFrame as d3m_DataFrame
from d3m.metadata.base import ALL_ELEMENTS
from datamart_isi import config
Q_NODE_SEMANTIC_TYPE = config.q_node_semantic_type


def run_wikifier(supplied_data: d3m_Dataset, skip_column_type: set):
    _logger = logging.getLogger(__name__)
    try:
        output_ds = copy.copy(supplied_data)
        specific_q_nodes = None
        res_id, supplied_dataframe = d3m_utils.get_tabular_resource(dataset=supplied_data, resource_id=None)
        target_columns = list(range(supplied_dataframe.shape[1]))
        temp = copy.deepcopy(target_columns)

        for each in target_columns:
            each_column_semantic_type = supplied_data.metadata.query((res_id, ALL_ELEMENTS, each))['semantic_types']
            if set(each_column_semantic_type).intersection(skip_column_type):
                temp.remove(each)

        target_columns = temp

        wikifier_res = wikifier.produce(pd.DataFrame(supplied_dataframe), target_columns, specific_q_nodes)
        output_ds[res_id] = d3m_DataFrame(wikifier_res, generate_metadata=False)
        # update metadata on column length
        selector = (res_id, ALL_ELEMENTS)
        old_meta = dict(output_ds.metadata.query(selector))
        old_meta_dimension = dict(old_meta['dimension'])
        old_column_length = old_meta_dimension['length']
        old_meta_dimension['length'] = wikifier_res.shape[1]
        old_meta['dimension'] = frozendict.FrozenOrderedDict(old_meta_dimension)
        new_meta = frozendict.FrozenOrderedDict(old_meta)
        output_ds.metadata = output_ds.metadata.update(selector, new_meta)

        # update each column's metadata
        for i in range(old_column_length, wikifier_res.shape[1]):
            selector = (res_id, ALL_ELEMENTS, i)
            metadata = {"name": wikifier_res.columns[i],
                        "structural_type": str,
                        'semantic_types': (
                            "http://schema.org/Text",
                            "https://metadata.datadrivendiscovery.org/types/Attribute",
                            Q_NODE_SEMANTIC_TYPE
                        )}
            output_ds.metadata = output_ds.metadata.update(selector, metadata)
        return output_ds

    except Exception as e:
        _logger.error("Wikifier running failed.")
        _logger.debug(e, exc_info=True)
        return supplied_data
