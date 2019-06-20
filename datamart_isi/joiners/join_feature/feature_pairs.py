from datamart_isi.joiners.join_feature.feature_factory import *
import typing
import rltk
from rltk.io.reader.dataframe_reader import DataFrameReader

"""
@rltk.set_id("dataframe_default_index", function_=lambda s: str(s))
class LeftDynamicRecord(rltk.AutoGeneratedRecord):
    @property
    def id(self):
        return self.raw_object['id']

    @property
    def join_column(self):
        return self.raw_object['join_column']
"""



@rltk.set_id("dataframe_default_index", function_=lambda s: str(s))
class LeftDynamicRecord(rltk.AutoGeneratedRecord):
    pass


@rltk.set_id("dataframe_default_index", function_=lambda s: str(s))
class RightDynamicRecord(rltk.AutoGeneratedRecord):
    pass


class FeaturePairs:
    def __init__(self,
                 left_df: pd.DataFrame,
                 right_df: pd.DataFrame,
                 left_columns: typing.List[typing.List[int]],
                 right_columns: typing.List[typing.List[int]],
                 left_metadata: dict,
                 right_metadata: dict,
    ):
        l1 = len(left_columns)
        l2 = len(right_columns)
        if not (l1 == l2 and l1 and l2):
            # TODO: throw error or warning
            return

        self._length = l1

        self._left_df = left_df
        self._right_df = right_df
        self._left_columns = left_columns
        self._right_columns = right_columns
        self._left_metadata = left_metadata
        self._right_metadata = right_metadata

        self._left_rltk_dataset = self._init_rltk_dataset(left_df, LeftDynamicRecord)
        self._right_rltk_dataset = self._init_rltk_dataset(right_df, RightDynamicRecord)

        self._pairs = self._init_pairs()

    @property
    def left_rltk_dataset(self):
        return self._left_rltk_dataset

    @property
    def right_rltk_dataset(self):
        return self._right_rltk_dataset

    @property
    def pairs(self):
        return self._pairs

    def get_rltk_block(self) -> typing.Optional[rltk.Block]:
        prime_key_l = []
        prime_key_r = []
        str_key_l = []
        str_key_r = []
        for f1, f2 in self.pairs:
            # why need to be string for old code???
            # if f1.data_type == DataType.STRING:
            # # 2019.4.10: TOKEN_CATEGORICAL should also considered here
            # if f1.distribute_type == DistributeType.CATEGORICAL or f1.distribute_type == DistributeType.TOKEN_CATEGORICAL:
            #     prime_key_l.append(f1.name)
            #     prime_key_r.append(f2.name)
            # elif f1.distribute_type == DistributeType.NON_CATEGORICAL:
            str_key_l.append(f1.name)
            str_key_r.append(f2.name)

        # if prime_key_l and prime_key_r:
        #     try:
        #         bg = rltk.HashBlockGenerator()
        #         block = bg.generate(
        #             bg.block(self.left_rltk_dataset, function_=lambda r: ''.join([str(getattr(r, pk)).lower()
        #                                                                           for pk in prime_key_l])),
        #             bg.block(self.right_rltk_dataset, function_=lambda r: ''.join([str(getattr(r, pk)).lower()
        #                                                                            for pk in prime_key_r])))
        #         return block
        #     except Exception as e:
        #         print(' - BLOCKING EXCEPTION: %s' % str(e))
        #         raise ValueError("failed to get blocking!")

        # if the datasets are too large, use each key's first char as blocking key
        # if str_key_l and str_key_r and len(self._left_df) * len(self._right_df) > 10000:
        try:
            bg = rltk.HashBlockGenerator()
            block = bg.generate(
                # original: str(getattr(r, pk))[0]
                bg.block(self.left_rltk_dataset, function_=lambda r: ''.join([str(getattr(r, pk)).lower()
                                                                              for pk in str_key_l])),
                bg.block(self.right_rltk_dataset, function_=lambda r: ''.join([str(getattr(r, pk)).lower()
                                                                               for pk in str_key_r]))
                )
            return block
        except Exception as e:
            print(' - BLOCKING EXCEPTION: %s' % str(e))

        raise ValueError("failed to get blocking!")

    def __len__(self):
        return self._length

    def _init_pairs(self):
        return [(FeatureFactory.create(self._left_df, self._left_columns[i], self._left_metadata),
                 FeatureFactory.create(self._right_df, self._right_columns[i], self._right_metadata))
                for i in range(self._length)]

    @staticmethod
    def _init_rltk_dataset(df, record_class):
        rltk_dataset = rltk.Dataset(reader=DataFrameReader(df, True), record_class=record_class)
        return rltk_dataset

