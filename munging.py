import numpy as np
import pandas as pd
from pandas.api.types import is_string_dtype, is_numeric_dtype
import re
from tqdm import tqdm

# MUNGING ____________________________________________________________________
def jproc_df(df, target=None, one_hot=False, copy=True, summary=True): #broken out into functions below, probably deprecated?
    '''
    Should be run on df after:
    1. mg.transform_dates, (turns datetime columns into ML usable)
    2. mg.remove_zerovar_cols, and consider drop cols is examined
    3. fastai's train_cats (turns obj/str cols into categorical)
    
    This function will convert categoricals to their codes, adding +1 (so nan
    is 0 instead of -1), and then will calculate means/stddev for
    normalizing and median for filling nan values. Also creates new cols
    demarkating whether value was originally nan via
    mg.get_medians_make_nullcols_fill_values.
    
    copy could be set to False if you suspect memory issues
    
    returns x, y, na_dict, mapper
    '''
    
    if target:
        y = df[target]
    else:
        print('No specified target column, assuming target already separated')
        y = []
    if copy:
        df = df.copy()
    
    # deal with cat cols, can either onehot or just turn into the categorical code
    cat_cols = df.select_dtypes('category').columns
    if one_hot:
        print('Turning categoricals into one_hot representation')
        dummied = pd.get_dummies(df[cat_cols])
        df.drop(cat_cols, axis=1, inplace=True)
    else:
        print('Converting categoricals to their codes . . .')
        for col in tqdm(cat_cols):
            df[col] = df[col].cat.codes+1
    
    # all other (numeric) cols
    # gather the means/stddevs/nas __________________
    print('Calculating means/medians/std_devs . . .')
    all_other_cols = [col for col in df.columns if col not in cat_cols]
    mapper = {}
    na_dict = {}
    for col in tqdm(all_other_cols):
        mapper[col] = {'mean': df[col].mean(),
                       'std_dev': df[col].std()}
        na_dict[col] = df[col].median()
        
    # make na cols, fill nas with median
    print('Making _isnull indicator columns . . .')
    has_nulls = [col for col in df.columns if any(df[col].isnull())]
    for col in tqdm(has_nulls):
        df[col+'_isnull'] = np.where(df[col].isnull(), 1, 0)
        df[col] = df[col].fillna(na_dict[col])
        
    # normalize the df excluding cat_cols
    print('Normalizing all non-categorical and non-_isnull columns . . .')
    all_other_cols = [col for col in df.columns if col not in cat_cols]
    for col in tqdm(all_other_cols):
        if '_isnull' not in col:
            df[col] = (df[col]-mapper[col]['mean'])/mapper[col]['std_dev']
    
    if one_hot:
        df = pd.concat([df, dummied], axis=1)
    
    if summary:
        print('Categorical cols: {0}\n\n'.format(list(cat_cols)))
        print('Made _isnull cols for: {0}\n\n'.format(list(has_nulls)))
        print('Normalized all other cols: {0}\n\n'.format([col for col in all_other_cols if '_isnull' not in col]))
            
    return df, y, na_dict, mapper

def transform_dates(df):
    '''
    Looks for datetime columns in the df, uses fastai's add_datepart to turn it
    into several columns (year, day, is quarter end, etc.)
    Does this inplace.
    '''
    date_cols = list(df.select_dtypes('datetime').columns)
    for col in date_cols:
        add_datepart(df, col,)

def remove_zerovar_cols(df, ret_cols=False):
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
        elif nunique == 2:
            consider_drop_cols.append(col)
    df.drop(zerovar_cols, axis=1, inplace=True)
    print('dropping the following cols: \n{0}'.format(zerovar_cols))
    print('only 2 values, consider dropping the following cols: \n{0}'.format(consider_drop_cols))
    if ret_cols:
        return zerovar_cols, consider_drop_cols

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
    if time: attr = attr + ['Hour', 'Minute', 'Second']
    for n in attr: df[targ_pre + n] = getattr(fld.dt, n.lower())
    df[targ_pre + 'Elapsed'] = fld.astype(np.int64) // 10 ** 9
    if drop: df.drop(fldname, axis=1, inplace=True)

def get_medians_make_nullcols_fill_values(df, na_map = {}, catboost=False):
    '''
    This function makes new columns marking if a column is null or not, and
    then fills the original column with median values. Does this inplace on
    the passed dataframe, but still returns the dataframe as well. IF col 
    dtype is not numeric, fills with mode instead. IF df[col].median() is 
    nan, fills with 0
    '''
    if not na_map:
        has_nulls = [col for col in df.columns if any(df[col].isnull())]
        for col in tqdm(has_nulls):
            if catboost:
                # catboost wanted nans in string dtypes to be string...
                if is_string_dtype(df[col]):
                    df[col] = df[col].replace(np.nan, 'nan')
                    na_map[col] = 'nan'

            # normal stuff
            if is_numeric_dtype(df[col]):
                if np.isnan(df[col].median()):
                    # handles case where the whole col is nan and you don't want to drop
                    # for reasons (old data doesn't have feature, but new data does)
                    df[col] = df[col].fillna(0)
                    na_map[col] = 0
                else:
                    median = df[col].median()
                    df[col] = df[col].fillna(median)
                    na_map[col] = median
            else:
                print('{0} col is likely of non-int/float dtype, filling with mode instead'.format(col))
                mode = df[col].mode() #is pandas series
                if len(mode) == 0:
                    mode = 0
                else:
                    mode = np.random.choice(mode) #if len(1), return mode, else ranodmly choose 1
                df[col] = df[col].fillna(mode)
                na_map[col] = mode
        return df, na_map
    else:
        print('Was passed an na_map. Will fill accordingly (test or valid data?)')
        for col, nafill in na_map.items():
            df[col] = df[col].fillna(nafill)
        return df

def normalize_df(df, means_stds = {}):
    '''
    This function should be run after get_medians_make_nullcols_fill_values.
    Doesn't normalize the "_isnull" cols that are added by the aforemetioned
    function. Will print out a list of columns that it did not normalize but
    those cols should be checked if normalization is actually desired. 
    means_stds are for valid or test sets, normalized with same values as train
    '''
    if not means_stds:
        unsure_cols = []
        means = {}
        stds = {}
        for col in tqdm(df.columns):
            if '_isnull' not in col:
                mean = df[col].mean()
                means[col] = mean
                std = df[col].std()
                stds[col] = std
                df[col] = (df[col]-mean)/std
            elif '_isnull' in col:
                pass
            else:
                if 'null' in col.lower() or 'is_null' in col.lower():
                    unsure_cols.append(col)
        print('These cols have word null or is_null in them. Double check. {0}'.format(
            unsure_cols))
        means_stds['means'] = means
        means_stds['stds'] = stds
        return df, means_stds
    else:
        print('Passed means_stds, normalizing cols according to means_stds (is valid or test set)')
        means = means_stds['means']
        stds = means_stds['stds']
        for col in means.keys():
            df[col] = (df[col] - means[col])/stds[col]
        return df
    

# def prepare_nulls_valid_test(valid_test, train, 

def no_nulls(df):
    '''
    Returns True if there are no nulls anywhere in the dataframe.
    '''
    has_nulls = [col for col in df.columns if any(df[col].isnull())]
    if len(has_nulls) != 0:
        return False
    else:
        return True
    
def all_numeric(df):
    '''
    Returns True if the df is completely numeric.
    '''
    if df.shape[1] == df._get_numeric_data().shape[1]:
        return True
    else:
        return False
    

def compress_memory(df):
    '''
    Tries to cast float and int dtype cols to smallest possible for dataframe.
    For saving RAM/disk space.
    '''
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
        cast_dtype = ok_dtypes[-1]
        if cast_dtype != col_type:
            dict_to_df[col] = df[col].astype(cast_dtype)
            changed_type_cols.append(col)
    print('changed dtypes of {0} cols'.format(len(changed_type_cols)))
    reduced = pd.DataFrame(dict_to_df)
    return changed_type_cols, pd.concat([irreducible, reduced], axis=1)