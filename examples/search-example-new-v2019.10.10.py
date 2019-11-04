import datamart_nyu
import os
import datamart
import json
from d3m.container.dataset import Dataset, D3MDatasetLoader

# the first time of the search on a dataset maybe slow, it will be much quicker afterwards

# load poverty dataset for a example
loader = D3MDatasetLoader()
path = "/Users/minazuki/Desktop/studies/master/2018Summer/data/datasets/seed_datasets_data_augmentation/DA_poverty_estimation/TRAIN/dataset_TRAIN/datasetDoc.json"
json_file = os.path.abspath(path)
all_dataset_uri = 'file://{}'.format(json_file)
input_dataset = loader.load(dataset_uri=all_dataset_uri)

# currently query_search only support setting up the specific columns that need to be wikified
# this can be useful when find that some columns should not be wikified (e.g.: some index columns)
# For example:
query_example = {'%^$#@wikifier@%^$#':{'FIPS': 'P882', 'State': 'Q35657'}}
meta_to_str = json.dumps(query_example)
query_search = datamart.DatamartQuery(keywords=[meta_to_str], variables=None)
datamart_nyu_url = "http://dsbox01.isi.edu:9000"
datamart_unit = datamart_nyu.RESTDatamart(connection_url=datamart_nyu_url)
search_unit = datamart_unit.search_with_data(query=query_search, supplied_data=input_dataset)
all_results1 = search_unit.get_next_page()


if all_results1 is None:
    print("No search result returned!")
# print the brief information of the search results
else:
    for i, each_search_result in enumerate(all_results1):
        each_search_res_json = each_search_result.get_json_metadata()
        print("------------ Search result No.{} ------------".format(str(i)))
        print(each_search_res_json['augmentation'])
        summary = each_search_res_json['summary'].copy()
        if "Columns" in summary:
            summary.pop("Columns")
        print(summary)
        print("-"*100)

"""
# each search result in all_result1 can be treated as one augment candidate, 
just need to run search_result.serialize() and use the output as the hyperparam as shown below

{
    "primitive": "d3m.primitives.data_augmentation.datamart_augmentation.Common",
    "hyperparameters":
    {
        'system_identifier':["NYU"],
        'search_result':[each_search_result.serialize()],
    }
}
"""