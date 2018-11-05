
import numpy as np
import pandas as pd
from torch import nn
import torch
import torch.nn.functional as F
from munging import no_nulls, all_numeric
from sklearn.model_selection import KFold
import lightgbm as lgb
from sklearn.metrics import roc_auc_score


# Pytorch/Fastai _____________________________________________________________
def emb_init(x):
    x = x.weight.data
    sc = 2/(x.size(1)+1)
    x.uniform_(-sc,sc)

class EmbedMixedInputModel(nn.Module):
    '''
    Like fastai's MixedInputModel but embeds continuous variables as well. Currently all embeddings have same dimension, selected by emb_dim
    '''
    def __init__(self, emb_dim, emb_szs, n_cont, emb_drop, out_sz, szs, drops,
                 y_range=None, use_bn=False, is_reg=True, is_multi=False):
        super().__init__()
        self.cat_embs = [nn.Embedding(c, emb_dim) for c, s in emb_szs]
        self.cont_embs =  [nn.Embedding(2, emb_dim) for i in range(n_cont)]
        self.embs = nn.ModuleList(self.cat_embs + self.cont_embs)
        for emb in self.embs: emb_init(emb)
        self.n_cat = len(emb_szs)
        self.n_cont = n_cont
        self.n_feats = self.n_cat+self.n_cont
        szs = [emb_dim*self.n_feats]+szs
        self.lins = nn.ModuleList([
            nn.Linear(szs[i], szs[i+1]) for i in range(len(szs)-1)])
        self.bns = nn.ModuleList([
            nn.BatchNorm1d(sz) for sz in szs[1:]])
        for o in self.lins: nn.init.kaiming_normal_(o.weight.data)
        self.outp = nn.Linear(szs[-1], out_sz)
        nn.init.kaiming_normal_(self.outp.weight.data)
        self.emb_drop = nn.Dropout(emb_drop)
        self.drops = nn.ModuleList([nn.Dropout(drop) for drop in drops])
        self.bn = nn.BatchNorm1d(n_cont)
        self.use_bn,self.y_range = use_bn,y_range
        self.is_reg = is_reg
        self.is_multi = is_multi
        
    def forward(self, x_cat, x_cont):
        if self.n_cat > 0:
            cat_emb_vecs = []
            for i,e in enumerate(self.embs[:-self.n_cont]):
                cat_emb_vecs.append(e(x_cat[:, i]))
        if self.n_cont > 0:
            # to get indicies on cont vars, 0 if nan, 1 if not nan
            x_cont_emb_idxs = torch.where(torch.isnan(x_cont), torch.zeros(x_cont.shape).cuda(), torch.ones(x_cont.shape).cuda()).long()
            cont_emb_vecs = []
            x_cont_clone = x_cont.clone()
            x_cont_clone[x_cont_clone != x_cont_clone]=0
            # scale the embedding lookup (0 or 1) by the actual value
            for i, e in enumerate(self.embs[-self.n_cont:]):
                cont_emb_vecs.append(torch.unsqueeze(x_cont_clone[:,i],1) * e(x_cont_emb_idxs[:, i]))
        if self.n_cat > 0:
            list_of_vecs = cat_emb_vecs + cont_emb_vecs
        else:
            list_of_vecs = cont_emb_vecs
        x = torch.cat(list_of_vecs, dim=1)
        for l,d,b in zip(self.lins, self.drops, self.bns):
            x = F.relu(l(x))
            if self.use_bn: x = b(x)
            x = d(x)
        x = self.outp(x)
        if not self.is_reg:
            if self.is_multi:
                x = F.sigmoid(x)
            else:
                x = F.log_softmax(x)
        elif self.y_range:
            x = F.sigmoid(x)
            x = x*(self.y_range[1] - self.y_range[0])
            x = x+self.y_range[0]
        return x

# Lightgbm ___________________________________________________________________
def cv_lgbm(train, train_y, test, ids_col = None, target_col = None, do_checks = False, n_folds = 5, make_submission=False):
    
    """Train and test a light gradient boosting model using
    cross validation. Assuming that the dataframe index are the ids.
    
    Parameters
    --------
        train (pd.DataFrame): 
            dataframe of training train to use 
            for training a model. Must include the TARGET column.
        train_y (np.array or pd.Series):
            the targets/labels aligned with train
        test (pd.DataFrame): 
            dataframe of testing train to use
            for making predictions with the model. 
        ids_col (string):
            column in dataframe that marks unique ids. If None, takes the index
        target_col (string):
            column in dataframe that marks the target/label. Used for dropping.
            If None, doesn't try to drop.
        do_checks (bool):
            Checks for nulls in train and test. Probably only wants to be run
            once when checking a new dataset. Afterwards, should be left False
            for speed. Also checks for all numeric datatypes in dfs.
        encoding (str, default = 'ohe'): 
            method for encoding categorical variables. Either 'ohe' for one-hot encoding or 'le' for integer label encoding
            n_folds (int, default = 5): number of folds to use for cross validation
        
    Return
    --------
        submission (pd.DataFrame): 
            dataframe with `SK_ID_CURR` and `TARGET` probabilities
            predicted by the model.
        feature_importances (pd.DataFrame): 
            dataframe with the feature importances from the model.
        valid_metrics (pd.DataFrame): 
            dataframe with training and validation metrics (ROC AUC) for each fold and overall.
        
    """
    if do_checks:
        print('Checking that no column has null values')
        assert no_nulls(train), 'train has null values'
        assert all_numeric(train), 'train has non-numeric cols'
        print('train has no null values and is all numeric')
        assert no_nulls(test), 'test has null values'
        assert all_numeric(test), 'test has non-numeric cols'
        print('test has no null values and is all numeric')
    print('Assuming the index of passed dataframes are their ids. Verify')
    
    # Extract the ids
    if ids_col:
        train_ids = train.ids_col.values
        test_ids = test.ids_col.values        
    else:
        train_ids = train.index.values
        test_ids = test.index.values
    
    # Extract the labels for training
    labels = np.array(train_y)
    
    # Remove the target
    if target_col:
        train = train.drop(columns = [target_col])
    
    # Reset the index so train_test_split can index in on position, not on id
    train = train.reset_index(drop=True)
    test = test.reset_index(drop=True)
        
    print('Training Data Shape: ', train.shape)
    print('Testing Data Shape: ', test.shape)
    
    # Extract feature names
    feature_names = list(train.columns)
    
    # Convert to np arrays
    train = np.array(train)
    test = np.array(test)
    
    # Create the kfold object
    k_fold = KFold(n_splits = n_folds, shuffle = False, random_state = 50)
    
    # Empty array for feature importances
    feature_importance_values = np.zeros(len(feature_names))
    
    # Empty array for test predictions
    test_predictions = np.zeros(test.shape[0])
    
    # Empty array for out of fold validation predictions
    out_of_fold = np.zeros(train.shape[0])
    
    # Lists for recording validation and training scores
    valid_scores = []
    train_scores = []
    models = []
    
    
    # Iterate through each fold
    for train_indices, valid_indices in k_fold.split(train):
        
        # Training data for the fold
        train_features, train_labels = train[train_indices,:], labels[train_indices]
        # Validation data for the fold
        valid_features, valid_labels = train[valid_indices,:], labels[valid_indices]
        
        # Create the model
        model = lgb.LGBMClassifier(n_estimators=10000, objective = 'binary', boosting_type='goss',
                                   class_weight = 'balanced', learning_rate = 0.05, 
                                   reg_alpha = 0.1, reg_lambda = 0.1, n_jobs = -1, random_state = 50)
        
        # Train the model
        model.fit(train_features, train_labels, eval_metric = 'auc',
                  eval_set = [(valid_features, valid_labels), (train_features, train_labels)],
                  eval_names = ['valid', 'train'],
                  early_stopping_rounds = 100, verbose = 200)
        
        # Record the best iteration
        best_iteration = model.best_iteration_
        
#         import ipdb; ipdb.set_trace()
        # Record the feature importances
        feature_importance_values += model.feature_importances_ / k_fold.n_splits
        
        # Make predictions
        test_predictions += model.predict_proba(test, num_iteration = best_iteration)[:, 1] / k_fold.n_splits
        
        # Record the out of fold predictions
        out_of_fold[valid_indices] = model.predict_proba(valid_features, num_iteration = best_iteration)[:, 1]
        
        # Record the best score
        valid_score = model.best_score_['valid']['auc']
        train_score = model.best_score_['train']['auc']
        
        valid_scores.append(valid_score)
        train_scores.append(train_score)
        
        # store the model
        models.append(model)
    
    # Make the feature importance dataframe
    feature_importances = pd.DataFrame({'feature': feature_names, 'importance': feature_importance_values})
    
    # Overall validation score
    valid_auc = roc_auc_score(labels, out_of_fold)
    
    # Add the overall scores to the metrics
    valid_scores.append(valid_auc)
    train_scores.append(np.mean(train_scores))
    
    # Needed for creating dataframe of validation scores
    fold_names = list(range(n_folds))
    fold_names.append('overall')
    
    # Dataframe of validation scores
    metrics = pd.DataFrame({'fold': fold_names,
                            'train': train_scores,
                            'valid': valid_scores}) 
    
    # Make the submission dataframe
    if make_submission:
        submission = pd.DataFrame({'ids': test_ids, 'TARGET': test_predictions})
        return submission, feature_importances, metrics, models
    else:
        return feature_importances, metrics, models