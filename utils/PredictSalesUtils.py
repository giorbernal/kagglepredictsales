# Group of support methods for Predict - sales process

import numpy as np
import pandas as pd
import featuretools as ft

import math

############ Datasets Functions #########################################

def getTrainEnriched(trainFile, itemsFile):
  sales_df = pd.read_csv(trainFile)
  items_df = pd.read_csv(itemsFile)

  sales_df.drop(labels=['date'],inplace=True,axis=1)
  sales_df = sales_df.reset_index()
  items_df.drop(labels=['item_name'],inplace=True,axis=1)

  dict_aux = {}
  sales_df['ID_pair'] = sales_df[['shop_id','item_id']].apply(setPair, args=[dict_aux], axis=1)
  sales_df = sales_df.merge(items_df)

  dict_aux = {}
  sales_df['ID_CAT_pair'] = sales_df[['shop_id','item_category_id']].apply(setPair, args=[dict_aux], axis=1)
  return sales_df

def getTestEnriched(testFile, itemsFile):
	test_df = pd.read_csv(testFile)
	items_df = pd.read_csv(itemsFile)

	items_df.drop(labels=['item_name'], inplace=True, axis=1)

	test_df_enriched = test_df.merge(right=items_df, on='item_id', how='left')
	dict_aux = {}
	test_df_enriched['ID_pair'] = test_df_enriched[['shop_id','item_id']].apply(setPair, args=[dict_aux], axis=1)
	dict_aux = {}	
	test_df_enriched['ID_CAT_pair'] = test_df_enriched[['shop_id','item_category_id']].apply(setPair, args=[dict_aux], axis=1)
	return test_df_enriched

############ Commons functions #########################################

def setPair(x, d):
    i = str(x[0]) + '-' + str(x[1])
    try:
        return d[i]
    except:
        result = i
        d[i] = result
        return result

def getCatAgg(sales_months_df):
    es = ft.EntitySet(id="prediction_sales")
    es = es.entity_from_dataframe(entity_id='sales',dataframe=sales_months_df, index='index')
    es = es.normalize_entity(base_entity_id='sales',
                         new_entity_id='idsCat',
                         index='ID_CAT_pair')
    feature_matrix_idsCat, feature_defs_idsCat = ft.dfs(entityset=es, target_entity='idsCat')
    idsCat = feature_matrix_idsCat.reset_index()
    idsCat_agg = idsCat[['ID_CAT_pair','SUM(sales.item_cnt_day)',
                     'MEAN(sales.item_cnt_day)','MEAN(sales.item_price)',
                     'STD(sales.item_cnt_day)','STD(sales.item_price)',
                     'MAX(sales.item_cnt_day)','MAX(sales.item_price)',
                     'MIN(sales.item_cnt_day)','MIN(sales.item_price)',
                     'SKEW(sales.item_cnt_day)','SKEW(sales.item_price)'
                    ]]
    idsCat_agg.columns = ['ID_CAT_pair','sum_shop_cat_sales',
                      'mean_shop_cat_day','mean_shop_cat_item_price',
                      'std_shop_cat_day','std_shop_cat_item_price',
                      'max_shop_cat_day','max_shop_cat_item_price',
                      'min_shop_cat_day','min_shop_cat_item_price',
                      'skew_shop_cat_day','skew_shop_cat_item_price',
                     ]
    return idsCat_agg

def getItemAgg(sales_months_df):
    es = ft.EntitySet(id="prediction_sales")
    es = es.entity_from_dataframe(entity_id='sales',dataframe=sales_months_df, index='index')
    es = es.normalize_entity(base_entity_id='sales',
                         new_entity_id='ids',
                         index='ID_pair',
                         additional_variables=['ID_CAT_pair'])
    feature_matrix_ids, feature_defs_ids = ft.dfs(entityset=es, target_entity='ids')
    ids = feature_matrix_ids.reset_index()
    return ids

def cleanEnsembleDataset(df, imputeTarget=False):
    # Removing targets without evidence
    df_cleaned = df[~(df['SUM(sales.item_cnt_day)'].isna() & df['sum_shop_cat_sales'].isna())]
    
    # Imputing features (na when there is not data for the item but theres is for the category)
    df_cleaned['isThereItem'] = df_cleaned[['SUM(sales.item_cnt_day)','COUNT(sales)']].apply(lambda x: 0 if (math.isnan(x[0]) & math.isnan(x[1])) else 1, axis=1)
    df_cleaned['SUM(sales.item_cnt_day)_imputed'] = df_cleaned['SUM(sales.item_cnt_day)'].apply(lambda x: 0 if (math.isnan(x)) else x)
    df_cleaned['SUM(sales.item_price)_imputed'] = df_cleaned['SUM(sales.item_price)'].apply(lambda x: 0 if (math.isnan(x)) else x)
    df_cleaned['MAX(sales.item_cnt_day)_imputed'] = df_cleaned['MAX(sales.item_cnt_day)'].apply(lambda x: 0 if (math.isnan(x)) else x)
    df_cleaned['MAX(sales.item_price)_imputed'] = df_cleaned['MAX(sales.item_price)'].apply(lambda x: 0 if (math.isnan(x)) else x)
    df_cleaned['MIN(sales.item_cnt_day)_imputed'] = df_cleaned['MIN(sales.item_cnt_day)'].apply(lambda x: 0 if (math.isnan(x)) else x)
    df_cleaned['MIN(sales.item_price)_imputed'] = df_cleaned['MIN(sales.item_price)'].apply(lambda x: 0 if (math.isnan(x)) else x)
    df_cleaned['MEAN(sales.item_cnt_day)_imputed'] = df_cleaned['MEAN(sales.item_cnt_day)'].apply(lambda x: 0 if (math.isnan(x)) else x)
    df_cleaned['MEAN(sales.item_price)_imputed'] = df_cleaned['MEAN(sales.item_price)'].apply(lambda x: 0 if (math.isnan(x)) else x)
    df_cleaned['COUNT(sales)_imputed'] = df_cleaned['COUNT(sales)'].apply(lambda x: 0 if (math.isnan(x)) else x)
    
    # Imputing features (na in case of unity)
    df_cleaned['onlyOneItem'] = df_cleaned[['SUM(sales.item_cnt_day)','STD(sales.item_cnt_day)']].apply(lambda x: 1 if (~math.isnan(x[0]) & math.isnan(x[1])) else 0, axis=1)
    df_cleaned['STD(sales.item_cnt_day)_imputed'] = df_cleaned['STD(sales.item_cnt_day)'].apply(lambda x: 0 if (math.isnan(x)) else x)
    df_cleaned['STD(sales.item_price)_imputed'] = df_cleaned['STD(sales.item_price)'].apply(lambda x: 0 if (math.isnan(x)) else x)
    df_cleaned['SKEW(sales.item_cnt_day)_imputed'] = df_cleaned['SKEW(sales.item_cnt_day)'].apply(lambda x: 0 if (math.isnan(x)) else x)
    df_cleaned['SKEW(sales.item_price)_imputed'] = df_cleaned['SKEW(sales.item_cnt_day)'].apply(lambda x: 0 if (math.isnan(x)) else x)
    df_cleaned['onlyOneCat'] = df_cleaned[['mean_shop_cat_day','skew_shop_cat_day']].apply(lambda x: 1 if (~math.isnan(x[0]) & math.isnan(x[1])) else 0, axis=1)
    df_cleaned['std_shop_cat_day_imputed'] = df_cleaned['std_shop_cat_day'].apply(lambda x: 0 if (math.isnan(x)) else x)
    df_cleaned['std_shop_cat_item_price_imputed'] = df_cleaned['std_shop_cat_item_price'].apply(lambda x: 0 if (math.isnan(x)) else x)
    df_cleaned['skew_shop_cat_day_imputed'] = df_cleaned['skew_shop_cat_day'].apply(lambda x: 0 if (math.isnan(x)) else x)
    df_cleaned['skew_shop_cat_item_price_imputed'] = df_cleaned['skew_shop_cat_item_price'].apply(lambda x: 0 if (math.isnan(x)) else x)

    # remove old variables
    df_cleaned.drop(labels=['SUM(sales.item_cnt_day)','SUM(sales.item_price)',
                            'MAX(sales.item_cnt_day)','MAX(sales.item_price)',
                            'MIN(sales.item_cnt_day)','MIN(sales.item_price)',
                            'MEAN(sales.item_cnt_day)','MEAN(sales.item_price)',
                            'COUNT(sales)'
                           ], inplace=True, axis=1)  
    df_cleaned.drop(labels=['STD(sales.item_cnt_day)','STD(sales.item_price)',
                            'SKEW(sales.item_cnt_day)','SKEW(sales.item_price)',
                            'std_shop_cat_day','std_shop_cat_item_price',
                            'skew_shop_cat_day','skew_shop_cat_item_price'
                           ], inplace=True, axis=1)
    
    # imputing target
    if (imputeTarget):
      df_cleaned['target_imputed'] = df_cleaned['total_sales'].apply(lambda x: 0 if (math.isnan(x)) else x)
      df_cleaned.drop(labels=['total_sales'], inplace=True, axis=1)
    
    return df_cleaned

def cleanStackingDataset(df,rowsToImputate):
    for rti in rowsToImputate:
        df[rti + '_imputated'] = df[rti].apply(lambda x: 0 if math.isnan(x) else x)
        df[rti + '_na_indicator'] = df[rti].apply(lambda x: 1 if math.isnan(x) else 0)
    df.drop(labels=rowsToImputate, inplace=True, axis=1)
    return df

############ Training functions #########################################

def getTargetAgg(sales_result_df):
    es = ft.EntitySet(id="target_sales")
    es = es.entity_from_dataframe(entity_id='sales',dataframe=sales_result_df, index='index')
    es = es.normalize_entity(base_entity_id='sales',
                         new_entity_id='target',
                         index='ID_pair',
                         additional_variables=['ID_CAT_pair'])
    feature_matrix_target, feature_defs_target = ft.dfs(entityset=es, target_entity='target')
    target = feature_matrix_target.reset_index()
    target_agg = target[['ID_pair','ID_CAT_pair','SUM(sales.item_cnt_day)']]
    target_agg.columns = ['ID_pair','ID_CAT_pair','total_sales']
    return target_agg

def joinTrainThreeParts(ids, idsCat, target):
    df = ids.merge(right=idsCat,on='ID_CAT_pair',how='left').merge(right=target,on='ID_pair',how='outer')
    df.drop(labels=['ID_CAT_pair_x'], inplace=True, axis=1)
    df.columns = ['ID_pair', 'SUM(sales.item_price)', 'SUM(sales.item_cnt_day)',
       'STD(sales.item_price)', 'STD(sales.item_cnt_day)',
       'MAX(sales.item_price)', 'MAX(sales.item_cnt_day)',
       'SKEW(sales.item_price)', 'SKEW(sales.item_cnt_day)',
       'MIN(sales.item_price)', 'MIN(sales.item_cnt_day)',
       'MEAN(sales.item_price)', 'MEAN(sales.item_cnt_day)', 'COUNT(sales)',
       'sum_shop_cat_sales', 'mean_shop_cat_day', 'mean_shop_cat_item_price',
       'std_shop_cat_day', 'std_shop_cat_item_price', 'max_shop_cat_day',
       'max_shop_cat_item_price', 'min_shop_cat_day',
       'min_shop_cat_item_price', 'skew_shop_cat_day',
       'skew_shop_cat_item_price', 'ID_CAT_pair', 'total_sales']
    df_with_ids = df[~df['SUM(sales.item_price)'].isna()]
    df_without_ids = df[df['SUM(sales.item_price)'].isna()]
    df_without_ids.drop(labels=['sum_shop_cat_sales', 'mean_shop_cat_day', 'mean_shop_cat_item_price',
       'std_shop_cat_day', 'std_shop_cat_item_price', 'max_shop_cat_day',
       'max_shop_cat_item_price', 'min_shop_cat_day',
       'min_shop_cat_item_price', 'skew_shop_cat_day',
       'skew_shop_cat_item_price'], inplace=True, axis=1)
    df_without_ids_enriched = df_without_ids.merge(right=idsCat, on='ID_CAT_pair', how='left')
    df_without_ids_enriched_sorted = df_without_ids_enriched[['ID_pair','SUM(sales.item_price)','SUM(sales.item_cnt_day)','STD(sales.item_price)','STD(sales.item_cnt_day)','MAX(sales.item_price)','MAX(sales.item_cnt_day)','SKEW(sales.item_price)','SKEW(sales.item_cnt_day)','MIN(sales.item_price)','MIN(sales.item_cnt_day)','MEAN(sales.item_price)','MEAN(sales.item_cnt_day)','COUNT(sales)','sum_shop_cat_sales','mean_shop_cat_day','mean_shop_cat_item_price','std_shop_cat_day','std_shop_cat_item_price','max_shop_cat_day','max_shop_cat_item_price','min_shop_cat_day','min_shop_cat_item_price','skew_shop_cat_day','skew_shop_cat_item_price','ID_CAT_pair','total_sales']]
    df_completed = pd.concat(objs=[df_with_ids,df_without_ids_enriched_sorted], axis=0)
    df_completed.drop(labels=['ID_CAT_pair'], inplace=True, axis=1)
    return df_completed

def generateFeaturesForTraining(sales_df, months_feature, month_target):
    print('features window:',months_feature,', target:',month_target)
    sales_months_df = sales_df[sales_df['date_block_num'].isin(months_feature)]
    sales_result_df = sales_df[sales_df['date_block_num'] == month_target]
    sales_months_df.drop(labels=['date_block_num','shop_id','item_id','item_category_id'], inplace=True, axis=1)
    sales_result_df.drop(labels=['date_block_num','shop_id','item_id','item_category_id'], inplace=True, axis=1)
    
    idsCat = getCatAgg(sales_months_df)
    ids = getItemAgg(sales_months_df)
    target = getTargetAgg(sales_result_df)
    target.head(3)
    
    joined = joinTrainThreeParts(ids, idsCat, target)
    
    # Insert the slot component for correlation purposes
    joined['slot'] = joined['COUNT(sales)'].apply(lambda x: months_feature[-1])
    return joined

############ Evaluation functions #########################################

def joinEvaluationThreeParts(test, ids, idsCat):
    df = test.merge(right=ids,on='ID_pair',how='left')
    
    # Adapting ID_CAT_pair
    df.drop(labels=['ID_CAT_pair_y'], inplace=True, axis=1)
    columns = np.array(df.columns)
    columns[5]='ID_CAT_pair'
    df.columns = columns
    
    df_completed = df.merge(right=idsCat,on='ID_CAT_pair',how='left')
    
    df_completed_sorted = df_completed[['ID_pair','SUM(sales.item_price)','SUM(sales.item_cnt_day)','STD(sales.item_price)','STD(sales.item_cnt_day)','MAX(sales.item_price)','MAX(sales.item_cnt_day)','SKEW(sales.item_price)','SKEW(sales.item_cnt_day)','MIN(sales.item_price)','MIN(sales.item_cnt_day)','MEAN(sales.item_price)','MEAN(sales.item_cnt_day)','COUNT(sales)','sum_shop_cat_sales','mean_shop_cat_day','mean_shop_cat_item_price','std_shop_cat_day','std_shop_cat_item_price','max_shop_cat_day','max_shop_cat_item_price','min_shop_cat_day','min_shop_cat_item_price','skew_shop_cat_day','skew_shop_cat_item_price','ID_CAT_pair']]
    df_completed_sorted.drop(labels=['ID_CAT_pair'], inplace=True, axis=1)
    return df_completed_sorted

def generateFeaturesForEvaluation(sales_df, months_feature, test_df):
    print('evaluation features window:',months_feature)
    sales_months_df = sales_df[sales_df['date_block_num'].isin(months_feature)]
    sales_months_df.drop(labels=['date_block_num','shop_id','item_id','item_category_id'], inplace=True, axis=1)
    
    idsCat = getCatAgg(sales_months_df)
    ids = getItemAgg(sales_months_df)
    
    joined = joinEvaluationThreeParts(test_df, ids, idsCat)
    
    # Insert the slot component for correlation purposes
    joined['slot'] = joined['COUNT(sales)'].apply(lambda x: months_feature[-1])
    return joined
