# initialize
from datamart.entries import Datamart
from d3m.container.dataset import Dataset, D3MDatasetLoader
from common_primitives.denormalize import Hyperparams as hyper_denormalize, DenormalizePrimitive
from d3m.base import utils as d3m_utils
import os
import pandas as pd

# load the ISI datamart, currently the url is here, may change in the future
isi_datamart_url = "http://dsbox02.isi.edu:9999/blazegraph/namespace/datamart3/sparql"
a = Datamart(connection_url=isi_datamart_url)
# load the D3M dataset,here we use "DA_poverty_estimation" as exmaple ,please change to your dataset path
loader = D3MDatasetLoader()
path = "/Users/minazuki/Desktop/studies/master/2018Summer/data/datasets/seed_datasets_data_augmentation/DA_poverty_estimation/TRAIN/dataset_TRAIN/datasetDoc.json"
json_file = os.path.abspath(path)
all_dataset_uri = 'file://{}'.format(json_file)
all_dataset = loader.load(dataset_uri=all_dataset_uri)
# run denormlaize primitive
denormalize_hyperparams = hyper_denormalize.defaults()
denormalize_primitive = DenormalizePrimitive(hyperparams = denormalize_hyperparams)
all_dataset = denormalize_primitive.produce(inputs = all_dataset).value


"""
start search, run search with data function.
Here because the dataset do not have any "Text" semantic type columns,
the system will said that no columns can be augment
"""
search_res = a.search_with_data(query=None, supplied_data=all_dataset)

"""
run get next page, we will get real search results, it will only have 2 wikidata search results
Explain:
here we do not find any "Qnodes" semantic type columns, so we will try to run wikifier before searching in wikidata database
Then, We will generate 2 Q nodes columns for FIPS and State. 
These 2 columns can be used to search in wikidata database
Because searching on wikidata with large amount of Q nodes, it will take about 3 minutes or more to finish
"""
s1 = search_res.get_next_page()