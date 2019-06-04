from datamart.entries import Datamart
from d3m.container.dataset import Dataset, D3MDatasetLoader
from common_primitives.denormalize import Hyperparams as hyper_denormalize, DenormalizePrimitive
import os
a = Datamart(connection_url="http://dsbox02.isi.edu:9999/blazegraph/namespace/datamart3/sparql")

loader = D3MDatasetLoader()
# path = "/Users/minazuki/Desktop/studies/master/2018Summer/data/datasets/seed_datasets_data_augmentation/DA_poverty_estimation/TRAIN/dataset_TRAIN/datasetDoc.json"
path = "/Users/minazuki/Desktop/studies/master/2018Summer/data/datasets/seed_datasets_current/LL1_336_MS_Geolife_transport_mode_prediction/TRAIN/dataset_TRAIN/datasetDoc.json"
json_file = os.path.abspath(path)
all_dataset_uri = 'file://{}'.format(json_file)
all_dataset = loader.load(dataset_uri=all_dataset_uri)

denormalize_hyperparams = hyper_denormalize.defaults()
denormalize_primitive = DenormalizePrimitive(hyperparams = denormalize_hyperparams)
all_dataset = denormalize_primitive.produce(inputs = all_dataset).value

search_res = a.search_with_data(query=None, supplied_data=all_dataset)
# this should have only 2 wikidata search results
s1 = search_res.get_next_page()
# this should be none
s2 = search_res.get_next_page()
# augment with first search result
aug1 = s1[0].augment(supplied_data=search_res.supplied_data)
# augment second time
aug2 = s1[1].augment(supplied_data=aug1)

# search second time and skip searching with wikidata part
search_res2 = a.search_with_data(query=None, supplied_data=aug2, skip_wikidata=True)

s3 = search_res2.get_next_page()
s4 = search_res2.get_next_page()

download_res = s3[0].download()

aug_res = s3[0].augment(supplied_data=search_res.supplied_data)