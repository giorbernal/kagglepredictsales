# Testing PredictSalesUtils Module

import sys
import numpy as np
import pandas as pd
import unittest

sys.path.append('./')

from utils.PredictSalesUtils import getTestEnriched 
from utils.PredictSalesUtils import getItemAgg, getCatAgg
from utils.PredictSalesUtils import getTargetAgg, joinTrainThreeParts
from utils.PredictSalesUtils import joinEvaluationThreeParts

class basicTest(unittest.TestCase):

	def testGetCatAgg(self):
		print('testing getCatAgg ... ')
		sales_months_df = self.__getSalesMonths()
		idsCat = getCatAgg(sales_months_df)

		self.assertTrue( idsCat.shape[0] == idsCat['ID_CAT_pair'].nunique())
		self.assertTrue( idsCat.shape[0] == 1977)
		self.assertTrue( idsCat.shape[1] == 12)

	def testGetItemAgg(self):
		print('testing getItemAgg ... ')
		sales_months_df = self.__getSalesMonths()
		ids = getItemAgg(sales_months_df)

		self.assertTrue( ids.shape[0] == ids['ID_pair'].nunique())
		self.assertTrue( ids.shape[0] == 65588)
		self.assertTrue( ids.shape[1] == 15)

	def testGetTargetAgg(self):
		print('testing getTargetAgg ... ')
		sales_target_df = self.__getTargetMonth()
		target = getTargetAgg(sales_target_df)

		self.assertTrue( target.shape[0] == target['ID_pair'].nunique())
		self.assertTrue( target.shape[0] == 31531)
		self.assertTrue( target.shape[1] == 3)

	def testJoinTrainThreeParts(self):
		print('testing joinTrainThreeParts ... ')
		sales_months_df = self.__getSalesMonths()
		sales_target_df = self.__getTargetMonth()
		ids = getItemAgg(sales_months_df)
		idsCat = getCatAgg(sales_months_df)
		target = getTargetAgg(sales_target_df)

		joined = joinTrainThreeParts(ids, idsCat, target)

		self.assertTrue( joined.shape[0] == joined['ID_pair'].nunique())
		self.assertTrue( joined.shape[0] == 78640)
		self.assertTrue( joined.shape[1] == 26)

	def testJoinEvaluationThreeParts(self):
		print('testing joinEvaluationThreeParts ... ')
		sales_months_df = self.__getSalesMonths()
		test_df_enriched = getTestEnriched('datasets/test.csv','datasets/items.csv')
		ids = getItemAgg(sales_months_df)
		idsCat = getCatAgg(sales_months_df)
		joined = joinEvaluationThreeParts(test_df_enriched, ids, idsCat)

		self.assertTrue( joined.shape[0] == joined['ID_pair'].nunique())
		self.assertTrue( joined.shape[0] == 214200)
		self.assertTrue( joined.shape[1] == 25)

	####### Support methods ####################

	def __getSalesMonths(self):
		sales_df = pd.read_csv('datasets/sales_train_enriched.csv')
		sales_df.drop(labels=['Unnamed: 0'], inplace=True, axis=1)

		months = [30,31,32]
		sales_months_df = sales_df[sales_df['date_block_num'].isin(months)]
		sales_months_df.drop(labels=['date_block_num','item_id','shop_id','item_category_id'], inplace=True, axis=1)
		return sales_months_df

	def __getTargetMonth(self):
		sales_df = pd.read_csv('datasets/sales_train_enriched.csv')
		sales_df.drop(labels=['Unnamed: 0'], inplace=True, axis=1)

		month = 33
		sales_target_df = sales_df[sales_df['date_block_num'] == month]
		sales_target_df.drop(labels=['date_block_num','item_id','shop_id','item_category_id'], inplace=True, axis=1)
		return sales_target_df

if __name__ == '__main__':
    unittest.main()
