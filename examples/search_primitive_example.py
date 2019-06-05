from common_primitives_datamart_augment import Hyperparams as hyper_augment, DataMartAugmentPrimitive
from common_primitives_datamart_download import Hyperparams as hyper_download, DataMartDownloadPrimitive

from d3m.container.dataset import Dataset, D3MDatasetLoader
from common_primitives.denormalize import Hyperparams as hyper_denormalize, DenormalizePrimitive
import os
from dsbox.datapreprocessing.cleaner.wikifier import WikifierHyperparams, Wikifier

example_search_wikidata1 = b'\x80\x03cdatamart.entries\nDatamartSearchResult\nq\x00)\x81q\x01}q\x02(X\r\x00\x00\x00search_resultq\x03}q\x04(X\x0e\x00\x00\x00p_nodes_neededq\x05]q\x06(X\x05\x00\x00\x00P1082q\x07X\x05\x00\x00\x00P2046q\x08X\x04\x00\x00\x00P571q\teX\x19\x00\x00\x00target_q_node_column_nameq\nX\r\x00\x00\x00FIPS_wikidataq\x0buX\n\x00\x00\x00query_jsonq\x0ch\x04X\x0b\x00\x00\x00search_typeq\rX\x08\x00\x00\x00wikidataq\x0eub.'

example_search_wikidata2 = b'\x80\x03cdatamart.entries\nDatamartSearchResult\nq\x00)\x81q\x01}q\x02(X\r\x00\x00\x00search_resultq\x03}q\x04(X\x0e\x00\x00\x00p_nodes_neededq\x05]q\x06(X\x05\x00\x00\x00P1082q\x07X\x05\x00\x00\x00P1449q\x08X\x05\x00\x00\x00P1451q\tX\x05\x00\x00\x00P1549q\nX\x05\x00\x00\x00P1705q\x0bX\x05\x00\x00\x00P1813q\x0cX\x05\x00\x00\x00P2044q\rX\x05\x00\x00\x00P2046q\x0eX\x05\x00\x00\x00P2927q\x0fX\x04\x00\x00\x00P571q\x10X\x05\x00\x00\x00P6591q\x11eX\x19\x00\x00\x00target_q_node_column_nameq\x12X\x0e\x00\x00\x00State_wikidataq\x13uX\n\x00\x00\x00query_jsonq\x14NX\x0b\x00\x00\x00search_typeq\x15X\x08\x00\x00\x00wikidataq\x16ub.'

example_search_datamart1 = b'\x80\x03cdatamart.entries\nDatamartSearchResult\nq\x00)\x81q\x01}q\x02(X\r\x00\x00\x00search_resultq\x03}q\x04(X\x05\x00\x00\x00scoreq\x05}q\x06(X\x08\x00\x00\x00datatypeq\x07X\'\x00\x00\x00http://www.w3.org/2001/XMLSchema#doubleq\x08X\x04\x00\x00\x00typeq\tX\x07\x00\x00\x00literalq\nX\x05\x00\x00\x00valueq\x0bX\x13\x00\x00\x000.22097086912079575q\x0cuX\x04\x00\x00\x00rankq\r}q\x0e(h\x07X$\x00\x00\x00http://www.w3.org/2001/XMLSchema#intq\x0fh\tX\x07\x00\x00\x00literalq\x10h\x0bX\x01\x00\x00\x002q\x11uX\x08\x00\x00\x00variableq\x12}q\x13(h\tX\x03\x00\x00\x00uriq\x14h\x0bXV\x00\x00\x00http://www.wikidata.org/entity/statement/D1000006-d547afad-3309-46b4-990d-946df3efbabdq\x15uX\x07\x00\x00\x00datasetq\x16}q\x17(h\tX\x03\x00\x00\x00uriq\x18h\x0bX\'\x00\x00\x00http://www.wikidata.org/entity/D1000006q\x19uX\x03\x00\x00\x00urlq\x1a}q\x1b(h\tX\x03\x00\x00\x00uriq\x1ch\x0bXa\x00\x00\x00https://raw.githubusercontent.com/usc-isi-i2/datamart-userend/master/example_datasets/poverty.csvq\x1duX\x05\x00\x00\x00titleq\x1e}q\x1f(X\x08\x00\x00\x00xml:langq X\x02\x00\x00\x00enq!h\tX\x07\x00\x00\x00literalq"h\x0bX\x0b\x00\x00\x00poverty.csvq#uX\x08\x00\x00\x00keywordsq$}q%(h\x07X\'\x00\x00\x00http://www.w3.org/2001/XMLSchema#stringq&h\tX\x07\x00\x00\x00literalq\'h\x0bX5\x02\x00\x00FIPStxt State Area_Name Rural-urban_Continuum_Code_2003 Urban_Influence_Code_2003 Rural-urban_Continuum_Code_2013 Urban_Influence_Code_2013 POVALL_2017 CI90LBAll_2017 CI90UBALL_2017 PCTPOVALL_2017 CI90LBALLP_2017 CI90UBALLP_2017 POV017_2017 CI90LB017_2017 CI90UB017_2017 PCTPOV017_2017 CI90LB017P_2017 CI90UB017P_2017 POV517_2017 CI90LB517_2017 CI90UB517_2017 PCTPOV517_2017 CI90LB517P_2017 CI90UB517P_2017 MEDHHINC_2017 CI90LBINC_2017 CI90UBINC_2017 POV04_2017 CI90LB04_2017 CI90UB04_2017 PCTPOV04_2017 CI90LB04P_2017 CI90UB04P_2017 FIPStxt_wikidata State_wikidataq(uX\x11\x00\x00\x00extra_informationq)}q*(h\x07X\'\x00\x00\x00http://www.w3.org/2001/XMLSchema#stringq+h\tX\x07\x00\x00\x00literalq,h\x0bX\x02\x00\x00\x00{}q-uX\x0c\x00\x00\x00datasetLabelq.}q/(h X\x02\x00\x00\x00enq0h\tX\x07\x00\x00\x00literalq1h\x0bX\x08\x00\x00\x00D1000006q2uX\x0c\x00\x00\x00variableNameq3}q4(h\x07X\'\x00\x00\x00http://www.w3.org/2001/XMLSchema#stringq5h\tX\x07\x00\x00\x00literalq6h\x0bX\x10\x00\x00\x00FIPStxt_wikidataq7uuX\n\x00\x00\x00query_jsonq8}q9(X\x08\x00\x00\x00keywordsq:]q;X\t\x00\x00\x00variablesq<}q=X\r\x00\x00\x00FIPS_wikidataq>X-\x06\x00\x00Q488175 Q495310 Q490436 Q375008 Q502345 Q338939 Q507353 Q52250 Q484582 Q495209 Q490014 Q26676 Q502984 Q488693 Q54089 Q511135 Q374358 Q304065 Q512732 Q495479 Q110904 Q108626 Q431826 Q5092 Q502447 Q137562 Q484450 Q506300 Q501248 Q489159 Q484590 Q486218 Q502952 Q27021 Q497702 Q835104 Q156613 Q488690 Q491762 Q494556 Q48905 Q511084 Q496292 Q389573 Q211360 Q280844 Q502050 Q220005 Q490019 Q495677 Q156273 Q461562 Q501796 Q156503 Q503492 Q61330 Q111851 Q430938 Q491911 Q182644 Q490357 Q376034 Q112137 Q109160 Q501092 Q26760 Q156575 Q496729 Q485420 Q490745 Q488219 Q512951 Q490813 Q492021 Q484401 Q512713 Q495013 Q114948 Q13188841 Q491623 Q156213 Q113823 Q490134 Q491831 Q115413 Q501976 Q504379 Q112869 Q490884 Q114479 Q51733 Q111254 Q494192 Q111744 Q489886 Q111876 Q500958 Q500686 Q376042 Q376004 Q483973 Q428902 Q496771 Q507169 Q421963 Q26526 Q504428 Q74704 Q502200 Q488528 Q501055 Q489088 Q506172 Q490215 Q506068 Q54254 Q502784 Q505987 Q56154 Q428298 Q619609 Q110130 Q503023 Q111304 Q484791 Q2613601 Q375608 Q493243 Q489702 Q502576 Q485502 Q490383 Q112107 Q494228 Q373953 Q501568 Q490774 Q507016 Q491982 Q510915 Q404898 Q110340 Q491035 Q339724 Q508288 Q400757 Q485361 Q490652 Q26740 Q820502 Q177678 Q111622 Q506235 Q501163 Q376505 Q61160 Q497810 Q111235 Q115433 Q486229 Q500845 Q495151 Q486243 Q496475 Q112957 Q58694 Q279452 Q488859 Q495691 Q509798 Q56149 Q504450 Q487415 Q490443 Q203049 Q506220 Q493599 Q494931 Q484282 Q156191 Q490450 Q56151 Q450159 Q494815 Q257311 Q108386 Q498377 Q375108 Q193230 Q491991 Q498072 Q312737 Q54446 Q156431 Q484551 Q115061 Q502250 Q495393 Q489327 Q49149q?suX\x0b\x00\x00\x00search_typeq@X\x07\x00\x00\x00generalqAub.'

a = hyper_augment.defaults()
b = hyper_download.defaults()
a = a.replace({"search_result":example_search_wikidata1})
b = b.replace({"search_result":example_search_wikidata1})
c = DataMartAugmentPrimitive(hyperparams=a)
d = DataMartDownloadPrimitive(hyperparams=b)
loader = D3MDatasetLoader()
path = "/Users/minazuki/Desktop/studies/master/2018Summer/data/datasets/seed_datasets_data_augmentation/DA_poverty_estimation/TRAIN/dataset_TRAIN/datasetDoc.json"
# path = "/Users/minazuki/Desktop/studies/master/2018Summer/data/datasets/seed_datasets_current/LL1_336_MS_Geolife_transport_mode_prediction/TRAIN/dataset_TRAIN/datasetDoc.json"
json_file = os.path.abspath(path)
all_dataset_uri = 'file://{}'.format(json_file)
all_dataset = loader.load(dataset_uri=all_dataset_uri)

# denormalize it
denormalize_hyperparams = hyper_denormalize.defaults()
denormalize_primitive = DenormalizePrimitive(hyperparams = denormalize_hyperparams)
all_dataset = denormalize_primitive.produce(inputs = all_dataset).value

# wikifier it
wikifier_hyperparams = WikifierHyperparams.defaults()
wikifier_primitive = Wikifier(hyperparams = wikifier_hyperparams)
all_dataset = wikifier_primitive.produce(inputs = all_dataset).value

# run augment
augment_result = c.produce(inputs=all_dataset).value

# run download
download_result = d.produce(inputs=all_dataset).value

# run augment second time
e = a.replace({"search_result":example_search_wikidata2})
f = DataMartAugmentPrimitive(hyperparams=e)
augment_result2 = f.produce(inputs=augment_result).value

# run augment third time
g = a.replace({"search_result":example_search_datamart1})
h = DataMartAugmentPrimitive(hyperparams=g)
augment_result3 = h.produce(inputs=augment_result2).value