import datamart
from d3m import container
from d3m import utils as d3m_utils
import d3m.metadata.base as metadata_base
from d3m.metadata import hyperparams
from d3m.primitive_interfaces.base import CallResult
from d3m.primitive_interfaces.transformer import TransformerPrimitiveBase
import os
import typing
import pickle


__all__ = ('DataMartDownloadPrimitive',)


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


class DataMartDownloadPrimitive(TransformerPrimitiveBase[Inputs, Outputs, Hyperparams]):
    metadata = metadata_base.PrimitiveMetadata({
        'id': '9e2077eb-3e38-4df1-99a5-5e647d21331f',
        'version': '0.1',
        'name': "Download a dataset from Datamart",
        'python_path': 'd3m.primitives.data_augmentation.datamart_download.Common',
        'source': {
            'name': "Datamart Program",
            'contact': 'mailto:remi.rampin@nyu.edu',
            'uris': [
                'https://gitlab.com/datadrivendiscovery/common-primitives/blob/master/common_primitives/datamart_download.py',
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
        search_result_loaded = pickle.loads(search_result)
        output = search_result_loaded.download(supplied_data=inputs)
        return CallResult(output)

    # def can_accept(cls, *, method_name: str, arguments: typing.Dict[str, typing.Union[metadata_base.Metadata, type]],
    #                hyperparams: Hyperparams):
    #     import pdb
    #     pdb.set_trace()
    #     output_metadata = super().can_accept(method_name=method_name, arguments=arguments, hyperparams=hyperparams)

    #     if output_metadata is None:
    #         return None

    #     if method_name != 'produce':
    #         return output_metadata

    #     return hyperparams['search_result'].get_metadata()
