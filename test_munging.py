import pytest
import numpy as np
from pandas.api.types import is_string_dtype, is_numeric_dtype
from hypothesis import given, example, reproduce_failure, assume
import hypothesis.strategies as st
from hypothesis.extra.pandas import column, columns, data_frames
# import hypothesis.extra.numpy as hnp
# from hypothesis import given, example
# import hypothesis.strategies as st
# import hypothesis.extra as ex

import munging as mg

base_df = data_frames([column('ints', dtype=int),
                       column('more_ints', dtype=int),
                       column('floats', dtype=float),
                       column('more_floats', dtype=float),
                       column('dates', dtype='datetime64[ns]'),
                       column('more_dates', dtype='datetime64[ns]'),
                       column('strings', elements=st.text()),
                       column('more_strings', elements=st.text()),
                      ])
numeric_df = data_frames([column('ints', dtype=int),
                          column('more_ints', dtype=int),
                          column('floats', dtype=float),
                          column('more_floats', dtype=float),
                         ])
num_str_df = data_frames([column('ints', dtype=int),
                          column('more_ints', dtype=int),
                          column('floats', dtype=float),
                          column('more_floats', dtype=float),
                          column('strings', elements=st.text()),
                          column('more_strings', elements=st.text()),
                      ])

@given(df=base_df,
       fldname=st.just('dates'),
       time=st.booleans())
def test_add_datepart(df, fldname, time):
    ori_shape = df.shape
    if time:
        mg.add_datepart(df, fldname, time=time)
        # del the ori col, add 18 features
        assert df.shape[1] == ori_shape[1]-1+17        
    else:
        # del the ori col, add 13 features
        mg.add_datepart(df, fldname, time=time)
        assert df.shape[1] == ori_shape[1]-1+12
    
@given(df=base_df)
def test_transform_dates(df):
    date_cols = list(df.select_dtypes('datetime').columns)
    non_date_cols_len = len(df.columns) - len(date_cols)
    mg.transform_dates(df)
    assert df.shape[1] == len(date_cols)*12 + non_date_cols_len

@given(df=numeric_df,
       ret_cols = st.booleans())
def test_remove_zerovar_cols(df, ret_cols):
    assume(len(df) > 1)
    if ret_cols:
        zerovar_cols, consider_drop_cols = mg.remove_zerovar_cols(df, ret_cols)
    else:
        mg.remove_zerovar_cols(df, ret_cols)
        
    for col in df.columns:
        var = df[col].var()
        print(col, var)
        assert np.isfinite(var)
        assert var > 0
        
    if ret_cols:
        assert zerovar_cols != None
        assert consider_drop_cols != None
              
@given(df=base_df)
def test_get_null_cols(df):
    mg.get_null_cols(df)
  
@given(df=base_df)
def test_make_null_ind_cols(df):
    ori_shape = df.shape
    null_cols = mg.get_null_cols(df)
    df, _ = mg.make_null_ind_cols(df)
    assert df.shape[1] == ori_shape[1] + len(null_cols)
   
@given(df=num_str_df)
def test_get_fill_values(df):
    assume(len(df) > 1)
    fill_dict = mg.get_fill_values(df)
    df_cols = df.columns
    for col, val in fill_dict.items():
        assert col in df_cols
        try:
            assert type(val) == str
        except (AssertionError, TypeError) as err:
            assert np.isfinite(val)
            
@given(df=num_str_df)
def test_fill_values(df):
    fill_dict = mg.get_fill_values(df)
    mg.fill_values(df, fill_dict)
    assert df.isnull().values.sum() == 0
    
@given(df=base_df)    
def test_get_categories(df):
    cats_dict = mg.get_categories(df)
    obj_cols = df.select_dtypes('object')
    assert set(obj_cols) == set(cats_dict.keys())
    
@given(df1=base_df, df2=base_df)
def test_encode_categories(df1, df2):
    assume(len(df1) > 1)
    assume(len(df2) > 1)
    cats_dict = mg.get_categories(df1)
    df2_ori = df2.copy()
    mg.encode_categories(df2, cats_dict)
    for col in cats_dict.keys():
        vals_not_in_cats_dict = set(df2_ori[col][df2[col]==0].values)
        assert all(val not in cats_dict[col] for val in vals_not_in_cats_dict)

@given(df=base_df)
def test_get_normalize_info(df):
    assume(len(df) > 2)
    norm_dict = mg.get_normalize_info(df)
    for col, val in norm_dict['means'].items():
        assert np.isfinite(val)
    for col, val in norm_dict['stds'].items():
        assert np.isfinite(val)
    
@given(df=base_df)
def test_normalize_df(df):
    ori_shape = df.shape
    norm_dict = mg.get_normalize_info(df)
    mg.normalize_df(df, norm_dict)
    assert ori_shape == df.shape

    
@given(df=base_df)
def test_compress_memory(df):
    assume(len(df)>0)
    ori_mem_usage = df.memory_usage(deep=True).values.sum()
    _, df = mg.compress_memory(df)
    new_mem_usage = df.memory_usage(deep=True).values.sum()
    assert new_mem_usage <= ori_mem_usage
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    