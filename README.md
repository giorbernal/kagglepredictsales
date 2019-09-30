# kaggle: Predict-Future-Sales

This is a solution for a kaggle competition: [Predict Future Sales](https://www.kaggle.com/c/competitive-data-science-predict-future-sales/overview).

In this competition we have tested two subjects in time series context:

* Work with **featuretools** library as a feature generator.
* Test Ensemble/Stacking method in this context.

# Dataset Download

The dataset can be set as follows:
```
> mkdir datasets
> cd datasets
> kaggle competitions download -c competitive-data-science-predict-future-sales
```

# Notebooks

We have developed our model in 4 steps (4 notebooks):

## TimeSeries-Predict-Sales-01-GenDataSet
In this notebook we have develop the required aggregations to generate the different N-slots sliding window datasets. In this case, we have generate 6 datasets (6 folds). We have used **featuretools** library to do so. Previously we have splitter the dataset for ensemble/stacking process (50-50 % aprox).

The first part of this notebook execute an enrichment process of the original dataset. Such enriched dataset will be used for the rest of the process, including the utils test (See next point in the README).

## TimeSeries-Predict-Sales-02-Cleaning
In this notebook, we just clean the previous folds. They all the same way.

## TimeSeries-Predict-Sales-03-Regression
Here, we are training the model as follows:
* **Ensembling**: each one of the folds is a **LightGB** model with hyperparaeters *partially* fitted to avoid overfitting
* **Stacking**: For stacking process we have used a **Neural Network** to perform a regression model.

## TimeSeries-Predict-Sales-04-Evaluation
Here, we have just evaluate the test set, first generating the proper dataset, and then executing predictions throghout the previously obtained models.

# Utils testing
To perform the previous notebooks we have developed a utils module ([utils/PredictSalesUtils.py](utils/PredictSalesUtils.py)) with can be tested with ([test/PredictSalesUtilsTest.py](test/PredictSalesUtilsTest.py)) by executing next commands:
```
> python -u test/PredictSalesUtilsTest.py
```

# Conclusions and next steps
On one hand, the featuretools library seems to be a useful tool to quick and easily  generate a set of aggregate data. on the other hand, although the base of the project could help us to get a good  global model, we should automatize the process at a higher level to be able to do next:
* Test different sliding windows sizes and different numbers of folds.
* Iterate in the models to search a better set of hyper-parameters to improve the performance of every models involved in the process.
* Evaluate other algorithms, at least in the ensembling phase.





