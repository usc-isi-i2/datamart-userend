from datamart_isi import rest
import os
import datamart
import json
from d3m.container.dataset import Dataset, D3MDatasetLoader

# the first time of the search on a dataset maybe slow, it will be much quicker afterwards

# load your dataset here
loader = D3MDatasetLoader()
path = "/Users/minazuki/Desktop/studies/master/2018Summer/data/datasets/seed_datasets_data_augmentation/DA_poverty_estimation/TRAIN/dataset_TRAIN/datasetDoc.json"
json_file = os.path.abspath(path)
all_dataset_uri = 'file://{}'.format(json_file)
input_dataset = loader.load(dataset_uri=all_dataset_uri)

# special keywords setting for specify the wikifier columns, in most condition, it is not needed
query_example = {'%^$#@wikifier@%^$#':{'FIPS': 'P882', 'State': 'Q35657'}}
meta_to_str = json.dumps(query_example)

keywords = ["poverty"]
query_search = datamart.DatamartQuery(keywords=keywords + [meta_to_str], variables=None)

datamart_url = "https://dsbox02.isi.edu:9000"
# if not given this is the default datamart url
datamart_unit = rest.RESTDatamart(connection_url=datamart_url)

"""
# current rest search support 4 types of control parameters
1. run_wikifier: bool, default is True
    if set to true, the system will find possible columns that can be wikifiered to get corresponding Q nodes in wikidata and 
    then a new columns will be added. This Q node column can be used for further augment. If set to false, the search speed
     will be quicker.
2. consider_wikifier_columns_only: bool, default is False
    if set to true, the system will only consider the Q node columns found from wikifier as join columns.
3. augment_with_time: bool, default is False
    if set to true, the system will auto generate join pairs base on 2 columns like (time_column, content_column). 
    This will return candidate datasets with both time and the contents are matched. If the supplied data do not contains any 
    time columns, the returned results will be empty. It would help when augmenting LL1_PHEM dataset.
4.  consider_time: bool, default is True
    Similar to augment_with_time, if set to true, the system will match the time ONLY. This is different from augment_with_time 
    which requires extra content column matches. If augment_with_time is set to true, this option will be useless.
    It would help when augmenting NY_TAXI dataset cause there is only a time column. 
"""
search_unit = datamart_unit.search_with_data(query=query_search, 
                                             supplied_data=input_dataset,
                                             run_wikifier=True,
                                             consider_wikifier_columns_only=True,
                                             augment_with_time=False,
                                             consider_time=False,)
all_results1 = search_unit.get_next_page()


if all_results1 is None:
    print("No search result returned!")
# print the brief information of the search results
else:
    rest.pretty_print_search_results(all_results1)

"""
# each search result in all_result1 can be treated as one augment candidate, 
just need to run search_result.serialize() and use the output as the hyperparam as shown below

{
    "primitive": "d3m.primitives.data_augmentation.datamart_augmentation.Common",
    "hyperparameters":
    {
        'system_identifier':["ISI"],
        'search_result':[each_search_result.serialize()],
    }
}
for details, please refer to sample-augment-pipeline.json

Before running the pipeline using runtime, please run following command:

export DATAMART_URL_ISI="https://dsbox02.isi.edu:9000"

"""

