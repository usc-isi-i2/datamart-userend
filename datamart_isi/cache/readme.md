[TOC]

#Cache system

The isi implement datamart use memcache as the base cache system, it has following cache part:

## Wikidata query cache

This part's cache will cache all wikidata query made from system and save. If the save query called second time, the cache system will return the cached results. Detail codes can be found at [wikidata_cache.py](https://github.com/usc-isi-i2/datamart-userend/blob/d3m/datamart_isi/utilities/wikidata_cache.py "wikidata_cache.py").
There is also a query updater  that can automatically update the query, the default setting is running updater each day. The detail codes and operation method can be found at `datamart_upload` repo.
Currently the maximum caching value size is set to be 100MB which should be available for most of wikidata query.

## General query cache
This part's cache will cache the augmented results so that when next time when the system received augment request on same supplied data and search results, the system can return the cached results instead of running the whole augment process again.
Different from wikidata cache, due to the reason that the size of the augment results may be very large, all augment results are stored in the pickled format in the disk instead of the memory. The cache system will only record the key of augment record and the place where the pickle file is stored to reduce usage on the memory space. Detail codes can be found at [general_search_cache.py](https://github.com/usc-isi-i2/datamart-userend/blob/d3m/datamart_isi/utilities/general_search_cache.py "general_search_cache.py")

## Materilaizer cache
This part's cache will cache the dataset uploaded to datamart database so that when next time calling the `download/ augment` function, no need to download from the original website and run the preprocess again. The detail codes can be found at [materializer_cache.py](https://github.com/usc-isi-i2/datamart-userend/blob/d3m/datamart_isi/cache/materializer_cache.py "materializer_cache.py")
This materialize process may be very slow ( > 10 min) if the target materialized dataset is very big (e.g. with 1 million rows). Based on the choice of doing wikifier or not, the materializer will have 2  version for each dataset: with wikifiered columns version and original version.

## Dataset metadata cache
Due to current datamart interaction interface is from NYU, the metadata of the input dataset will not be sent with the main data. So this part's cache will load the correct metadata if there already exists dataset that match all column names. This part is specicially aimed for d3m's official dataset so that fewer and only correct columns will be detected that need to be wikified instead of nearly all columns. This cache will also save the metadata of the augmented result to ensure the second time's augment still get correct metadata. Detail codes can be found at 

