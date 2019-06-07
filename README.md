# ISI Datamart
This project provides an implementation of the [D3M's Datamart API](https://gitlab.com/datadrivendiscovery/datamart-api).

## Using ISI Datamart

Here is a Jupyter notebook that show show to search the datamart, [here](https://github.com/usc-isi-i2/datamart-userend/blob/d3m/examples/search_primitive_example.ipynb).

## Sample Pipelines

## Uploading Dataset

To upload datasets into the ISI Datamart, use the python class `datamart_isi.upload.Datamart_isi_upload`, [here](https://github.com/usc-isi-i2/datamart-upload).

[Here](https://github.com/usc-isi-i2/datamart-upload/blob/master/examples/upload_example.ipynb) is a sample Jupyter notebook that shows how to upload datasets into the ISI Datamart.

First, create an uploader instance, and call its `load_and_preprocess` method with a URL pointing to the CSV file. The `load_and_preprocess` method returns two lists: a list of dataframes and a list of metadata describing those dataframes. In this case, the length of each list is one since the input URL references a single CSV file.
```python
uploader = Datamart_isi_upload()
url_to_csv_file = 'https://raw.githubusercontent.com/usc-isi-i2/datamart-userend/master/example_datasets/List_of_United_States_counties_by_per_capita_income.csv'
dataframes, metadata = uploader.load_and_preprocess(input_dir=url_to_csv_file, file_type='online_csv')
```

The metadata contains information deduced by the data profiler. To add additional metadata information to the dataset do:
```python
metadata[0]['title'] = "County Income"
metadata[0]['description'] = "Rank of counties by income"
metadata[0]['keywords'] = ["Per capita income", "Median household income", "Median family income"]
```

Finally, upload the dataset with the updated metadata into the datmart:
```python
uploader.model_data(dataframes, metadata, 0)
uploader.upload()
```
