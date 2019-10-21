# for wikifier parameters
input_type = "pandas"
use_cache = True
# target_columns(typing.List[int]): target columns to find with wikidata
target_columns = None
# target_p_nodes(typing.List[str]): user-speicified P node want to get, can be None if want automatic search
target_p_nodes = None
# wikifier_choice(typing.List[str]): including three choices [ identifier, new_wikifier and automatic].
wikifier_choice = None
# threshold(float): minimum coverage ratio for a wikidata columns to be appended
threshold = 0.7
# choose how many qnodes columns for each column
top_k = 1
# qnodes which are regarded as unacceptable and should be avoided
blacklist = []

# for find_identity parameters
is_top_k = False
common_top_k = 5

