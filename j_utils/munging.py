# %load munging.py
import numpy as np
import pandas as pd
from pandas.api.types import is_string_dtype, is_numeric_dtype
import re
from tqdm import tqdm


# MUNGING ____________________________________________________________________
def sort_train_eval(train, eva, id_col, sort_col, assert_shape=True):
    '''
    Helps to sort train df by sort_col which is only in eva df. train and
    eva should both contain id_col (the unique identifier).
    assert_shape checks that every id_col in train and eva are one-to-one
    '''
    if assert_shape:
        train_id = set(train[id_col])
        eva_id = set(eva[id_col])
        assert not train.duplicated(id_col).any(), print('first df has id dupes')
        assert not eva.duplicated(id_col).any(), print('second df has id dupes')
        assert train_id == eva_id, print('some ids not present in both dfs')
        assert train.shape[0] == eva.shape[0], print('ids not one to one')
    sorted_id = eva.sort_values(sort_col)[id_col]
    sort_map = dict(zip(sorted_id, range(len(sorted_id))))
    train['sort_temp'] = train[id_col].map(sort_map)
    train = train.sort_values('sort_temp')
    train = train.drop('sort_temp', axis=1)
    return train
    

def check_train_valid(train, valid, id_col, original=None):
    '''
    Function to check that two df's (train, valid) do not contain the same
    data samples based on id_col. If passing original full set, it checks
    that all samples are accounted for as well (if splitting train and valid
    from original)
    '''
    train_id = set(train[id_col])
    test_id = set(valid[id_col])
    assert len(train_id.intersection(test_id)) == 0
    if original:
        assert train_id.union(test_id) == set(original[id_col])
        print('all ids accounted for')
    print('no overlapping id cols found')


def train_proc(df, normalize = True, verbose=True, isnull=True):
    '''
    get_noninf_val, replace_infs with respective max or min value
    get_normalize_info, normalize_df (if needed)
    transform dates
    make_null_ind_cols
    get_fill_values, fill_values
    get_categories, encode_categories
    remove_zerovar_cols
    
    Returns list of:
    df, all_train_colnames, max_dict, min_dict, new_null_colnames, fill_dict, cats_dict, (norm_dict)
    '''

    max_dict, min_dict = get_noninf_val(df)
    replace_infs(df, max_dict, min_dict)
    if normalize:
        norm_dict = get_normalize_info(df)
        normalize_df(df, norm_dict)
    transform_dates(df)
    if isnull:
        df, new_null_colnames = make_null_ind_cols(df)
    fill_dict = get_fill_values(df)
    fill_values(df, fill_dict)
    cats_dict = get_categories(df)
    encode_categories(df, cats_dict)
    remove_zerovar_cols(df, verbose=verbose)
    to_ret = [df, df.columns, max_dict, min_dict, fill_dict, cats_dict]
    if isnull:
        to_ret.append(new_null_colnames)
    if normalize:
        to_ret.append(norm_dict)
    return to_ret

def val_test_proc(df, all_train_colnames, max_dict, min_dict, fill_dict, cats_dict, norm_dict={}, verbose=True, isnull = True):
    '''
    # Make cols match
    # replace_infs
    # noramlize_df (if needed)
    # transform dates
    # make_null_ind_cols
    # fill_values
    # encode_categories
    # select cols to match train
    '''
    for col in all_train_colnames:
        if col not in df.columns:
            df[col] = np.nan
    
    transform_dates(df)
    replace_infs(df, max_dict, min_dict)
    if norm_dict:
        normalize_df(df, norm_dict)
    if isnull:
        df, new_null_colnames = make_null_ind_cols(df)
    fill_values(df, fill_dict)
    encode_categories(df, cats_dict)
    df = df[all_train_colnames]

    return df
# sub_functions

def add_datepart(df, fldname, drop=True, time=False):
    """add_datepart converts a column of df from a datetime64 to many columns containing
    the information from the date. This applies changes inplace.

    Parameters:
    -----------
    df: A pandas data frame. df gain several new columns.
    fldname: A string that is the name of the date column you wish to expand.
        If it is not a datetime64 series, it will be converted to one with pd.to_datetime.
    drop: If true then the original date column will be removed.
    time: If true time features: Hour, Minute, Second will be added.

    Examples:
    ---------

    >>> df = pd.DataFrame({ 'A' : pd.to_datetime(['3/11/2000', '3/12/2000', '3/13/2000'], infer_datetime_format=False) })
    >>> df

        A
    0   2000-03-11
    1   2000-03-12
    2   2000-03-13

    >>> add_datepart(df, 'A')
    >>> df

        AYear AMonth AWeek ADay ADayofweek ADayofyear AIs_month_end AIs_month_start AIs_quarter_end AIs_quarter_start AIs_year_end AIs_year_start AElapsed
    0   2000  3      10    11   5          71         False         False           False           False             False        False          952732800
    1   2000  3      10    12   6          72         False         False           False           False             False        False          952819200
    2   2000  3      11    13   0          73         False         False           False           False             False        False          952905600
    """
    fld = df[fldname]
    if not np.issubdtype(fld.dtype, np.datetime64):
        df[fldname] = fld = pd.to_datetime(fld, infer_datetime_format=True)
    targ_pre = re.sub('[Dd]ate$', '', fldname)
    attr = ['Year', 'Month', 'Week', 'Day', 'Dayofweek', 'Dayofyear',
            'Is_month_end', 'Is_month_start', 'Is_quarter_end', 'Is_quarter_start', 'Is_year_end', 'Is_year_start']
    if time: attr = attr + ['Hour', 'Minute', 'Second', 'Microsecond', 'Nanosecond']
    for n in attr: df[targ_pre + n] = getattr(fld.dt, n.lower())
#     df[targ_pre + 'Elapsed'] = fld.astype(np.int64) // 10 ** 9
    if drop: df.drop(fldname, axis=1, inplace=True)

def transform_dates(df):
    '''
    Looks for datetime columns in the df, uses fastai's add_datepart to turn it
    into several columns (year, day, is quarter end, etc.)
    Does this inplace.
    '''
    date_cols = list(df.select_dtypes('datetime').columns)
    for col in date_cols:
        add_datepart(df, col,)


def get_inf_cols(df):
    '''
    Gets colnames with inf in them
    '''
    num_cols = df.select_dtypes([np.number]).columns
    inf_cols = [col for col in num_cols if any(np.isinf(df[col]))]
    return inf_cols

def get_noninf_val(df):
    inf_cols = get_inf_cols(df)
    max_dict = df[inf_cols].replace([np.inf], np.nan).max().to_dict()
    min_dict = df[inf_cols].replace([-np.inf], np.nan).min().to_dict()
    for k,v in max_dict.items():
        max_dict[k] = v*100 #was inf so make it 100x larger than max
    for k,v in min_dict.items():
        if v > 0:
            v = v/100
        elif v <= 0:
            v = v*100
        min_dict[k] = v #was -inf so make it 100x larger than min    
    return max_dict, min_dict

def replace_infs(df, max_dict, min_dict):
    for col, val in max_dict.items():
        df[col] = df[col].replace([np.inf], val)
    for col, val in min_dict.items():
        df[col] = df[col].replace([-np.inf], val)
                
def get_null_cols(df):
    '''
    Gets null colnames in df if there are any
    '''
    null_cols = [col for col in df.columns if any(df[col].isnull())]
    return null_cols

def make_null_ind_cols(df):
    '''
    Had to return or else cols weren't actually added inplace
    '''
    null_cols = get_null_cols(df)
    for col in null_cols:
        df[col+'_isnull'] = np.where(df[col].isnull(), 1, 0)
    return df, [col+'_isnull' for col in null_cols]        
        
def get_fill_values(df):
    '''
    Only to be used on train. Get the right fill values for every col.
    '''
    assert len(df.columns) == len(df.select_dtypes([np.number, 'object', 'bool']).columns)
    all_cols = df.columns
    fill_dict = {}
    for col in all_cols:
        if is_numeric_dtype(df[col]):
            fill_val = df[col].median() # take median
            if not np.isfinite(fill_val):
                try:
                    fill_val = df[col].mode()[0] # take mode
                except IndexError:
                    fill_val = 0
                finally:
                    if not np.isfinite(fill_val):
                        fill_val = 0
                
        elif is_string_dtype(df[col]):
            try:
                fill_val = df[col].mode()[0] # take mode
            except IndexError:
                fill_val = '0'
            
        # if fill val still not finite, set to 0
#         if not np.isfinite(fill_val):
#             fill_val = 0 # if mode is nan or inf
        fill_dict[col] = fill_val
    return fill_dict

def fill_values(df, fill_dict):
#     assert set(df.columns) == set(fill_dict.keys())
    df.fillna(fill_dict, inplace=True)

def get_categories(df):
    '''
    only gets categories.
    '''
    obj_cols = df.select_dtypes('object')
    cats_dict = {}
    for col in obj_cols:
        cats_dict[col] = df[col].astype('category').cat.categories
    return cats_dict
            
def encode_categories(df, cats_dict):
    for col, cats in cats_dict.items():
        df[col] = pd.Categorical(df[col], categories = cats)
        df[col] = df[col].cat.codes + 1 #so nan -> -1 becomes 0
            
def remove_zerovar_cols(df, ret_cols=False, verbose=False):
    '''
    Iterates throught columns and checks nunique(). If nunique == 1, then the whole
    column has only one value, and will be dropped inplace. Prints out which
    columns it has dropped and can return them if desired.
    '''
    zerovar_cols = []
    consider_drop_cols = []
    for col in df.columns:
        nunique = df[col].nunique(dropna=False)
        if nunique <= 1:
            zerovar_cols.append(col)
        else:
            var = df[col].var()
            if (not np.isfinite(var)) | (var == 0):
                zerovar_cols.append(col)
    df.drop(zerovar_cols, axis=1, inplace=True)

            
    for col in df.columns:
        nunique = df[col].nunique(dropna=False)
        if nunique == 2:
            if '_isnull' not in col:
                consider_drop_cols.append(col)

    if verbose:
        print('dropping the following cols: \n{0}'.format(zerovar_cols))           
        print('only 2 values, consider dropping the following cols: \n{0}'.format(consider_drop_cols))
    if ret_cols:
        return zerovar_cols, consider_drop_cols        
        
def all_numeric(df):
    '''
    Returns True if the df is completely numeric.
    '''
    if df.shape[1] == df._get_numeric_data().shape[1]:
        return True
    else:
        return False

def get_normalize_info(df):
    num_df = df.select_dtypes(np.number)
    assert all_numeric(num_df)
    norm_dict = {}
    means = {}
    stds = {}
    rel_cols = [col for col in num_df.columns if '_isnull' not in col]
    std_notfin = {}
    for col in rel_cols:
        mean = num_df[col].mean()
        std = num_df[col].std()
        if np.isfinite(std):
            means[col] = mean
            stds[col] = std
        else:
            std_notfin[col] = std
    norm_dict['means'] = means
    norm_dict['stds'] = stds
    norm_dict['std_notfin'] = std_notfin
    return norm_dict

def normalize_df(df, norm_dict):
    means = norm_dict['means']
    stds = norm_dict['stds']
    for col in means.keys():
        mean = means[col]
        std = stds[col]
        df[col] = (df[col]-mean)/std
        
def reduce_memory(df, verbose=True):
    '''
    Tries to cast float and int dtype cols to smallest possible for dataframe.
    For saving RAM/disk space.
    '''
    if verbose:
        print('trying to change columns to smaller dtypes when possible')
        ori_mem = df.memory_usage(deep=True).sum()
        print('original dataframe is {0} MB or {1} GB'.format(ori_mem/(1024**2), ori_mem/(1024**3)))
    dict_to_df = {}
    changed_type_cols = []
    reducible = df.select_dtypes(['int', 'float'])
    irreducible = df[[col for col in df.columns if col not in reducible.columns]]
    for col in tqdm(reducible.columns):
        col_type = df[col].dtypes.name
        max_val = df[col].max()
        min_val = df[col].min()
        int_types = ['int32', 'int16', 'int8']
        float_types = ['float32'] #float 16 not supported in feather format?, 'float16']
        np_typedict = np.typeDict
        if 'float' in col_type:
            type_list = float_types
            infoer = np.finfo
        elif 'int' in col_type:
            type_list = int_types
            infoer = np.iinfo
        ok_dtypes = []
        for dtype in type_list:
            dt_max = infoer(dtype).max
            dt_min = infoer(dtype).min
            if (max_val <= dt_max) & (min_val >= dt_min):
                ok_dtypes.append(dtype)
        try:
            cast_dtype = ok_dtypes[-1]
        except IndexError:
            cast_dtype = col_type
        if cast_dtype != col_type:
            dict_to_df[col] = df[col].astype(cast_dtype)
            changed_type_cols.append(col)
    print('changed dtypes of {0} cols'.format(len(changed_type_cols)))
    reduced = pd.DataFrame(dict_to_df)
    smaller = pd.concat([irreducible, reduced], axis=1)        
    if verbose:
        small_mem = smaller.memory_usage(deep=True).sum()
        print('reduced dataframe is {0} MB or {1} GB'.format(small_mem/(1024**2), small_mem/(1024**3)))
    return changed_type_cols, smaller
