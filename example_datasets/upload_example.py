from store import Datamart_dataset
# this sample will save the following online csv datasets into datamart in blaze graph
a = Datamart_dataset()
all_dir = ["https://raw.githubusercontent.com/usc-isi-i2/datamart-userend/master/example_datasets/List_of_United_States_counties_by_per_capita_income.csv", 
    "https://raw.githubusercontent.com/usc-isi-i2/datamart-userend/master/example_datasets/Most-Recent-Cohorts-Scorecard-Elements.csv", 
    "https://raw.githubusercontent.com/usc-isi-i2/datamart-userend/master/example_datasets/Unemployment.csv", 
    "https://raw.githubusercontent.com/usc-isi-i2/datamart-userend/master/example_datasets/educate.csv", 
    "https://raw.githubusercontent.com/usc-isi-i2/datamart-userend/master/example_datasets/population.csv", 
    "https://raw.githubusercontent.com/usc-isi-i2/datamart-userend/master/example_datasets/poverty.csv"
    ]
for input_dir in all_dir:
    df,meta=a.load_and_preprocess(input_dir=input_dir,file_type="online_csv")
    a.model_data(df, meta)
    a.upload()

# input_dir = "/Users/minazuki/Downloads/usda/population_new.csv"
# df,meta=a.load_and_preprocess(input_dir)
# a.model_data(df, meta)
# a.output_to_ttl("2")

# input_dir = "/Users/minazuki/Downloads/usda/poverty_new.csv"
# df,meta=a.load_and_preprocess(input_dir)
# a.model_data(df, meta)
# a.output_to_ttl("3")

# input_dir = "/Users/minazuki/Downloads/usda/Unemployment_new.csv"
# df,meta=a.load_and_preprocess(input_dir)
# a.model_data(df, meta)
# a.output_to_ttl("4")

