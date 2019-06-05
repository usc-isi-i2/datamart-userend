import datamart
from d3m import container
from d3m import utils as d3m_utils
import d3m.metadata.base as metadata_base
from d3m.metadata import hyperparams
from d3m.primitive_interfaces.base import CallResult
from d3m.primitive_interfaces.transformer import TransformerPrimitiveBase
from datamart import DatamartSearchResult, DatasetColumn
import os
import typing
import pickle

__all__ = ('DataMartAugmentPrimitive',)

Inputs = container.Dataset
Outputs = container.Dataset

class Hyperparams(hyperparams.Hyperparams):
    search_result = hyperparams.Hyperparameter[bytes](
        default=b'',
        semantic_types=[
            'https://metadata.datadrivendiscovery.org/types/ControlParameter',
        ],
        description="Pickled search result provided by Datamart",
    )
    augment_columns = hyperparams.Hyperparameter[list](
        default=container.List(),
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
        description="Optional list of columns from the Datamart dataset that will be added"
    )


class DataMartAugmentPrimitive(TransformerPrimitiveBase[Inputs, Outputs, Hyperparams]):
    metadata = metadata_base.PrimitiveMetadata({
        'id': 'fe0f1ac8-1d39-463a-b344-7bd498a31b91',
        'version': '0.1',
        'name': "Perform augmentation using Datamart",
        'python_path': 'd3m.primitives.data_augmentation.datamart_augment.Common',
        'source': {
            'name': "Datamart Program",
            'contact': 'mailto:remi.rampin@nyu.edu',
            'uris': [
                'https://gitlab.com/datadrivendiscovery/common-primitives/blob/master/common_primitives/datamart_augment.py',
                'https://gitlab.com/datadrivendiscovery/common-primitives.git',
            ],
        },
        'installation': [{
            'type': metadata_base.PrimitiveInstallationType.PIP,
            'package_uri': 'git+https://gitlab.com/datadrivendiscovery/common-primitives.git@{git_commit}#egg=common_primitives'.format(
                git_commit=d3m_utils.current_git_commit(
                    os.path.dirname(__file__)),
            ),
        }],
        'algorithm_types': [
            metadata_base.PrimitiveAlgorithmType.DATA_RETRIEVAL,
        ],
        'primitive_family': metadata_base.PrimitiveFamily.DATA_AUGMENTATION,
    })

    def produce(self, *, inputs: Inputs, timeout: float = None, iterations: int = None) -> CallResult[Outputs]:
        search_result = self.hyperparams['search_result']
        augment_columns = self.hyperparams['augment_columns']
        search_result_loaded = pickle.loads(search_result)
        output = search_result_loaded.augment(supplied_data=inputs, augment_columns=augment_columns)
        return CallResult(output)
