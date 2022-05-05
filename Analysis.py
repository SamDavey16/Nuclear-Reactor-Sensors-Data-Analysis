import pyspark
import pandas
import matplotlib.pyplot as plt
from pyspark.sql import SparkSession
import numpy as np
import pyspark.sql.functions as func
import seaborn as sns
from os import environ
from pyspark.ml.stat import Correlation
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import DecisionTreeClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer
from pyspark.mllib.evaluation import MulticlassMetrics
from sklearn.model_selection import train_test_split
from pyspark.ml.classification import LinearSVC
from pyspark.ml.classification import MultilayerPerceptronClassifier
import os
import sys
import itertools

# Set pyspark environment variables
os.environ['PYSPARK_PYTHON'] = sys.executable
os.environ['PYSPARK_DRIVER_PYTHON'] = sys.executable

spark = SparkSession.builder.config('spark.driver.memory','32G').config('spark.ui.showConsoleProgress', 'false').getOrCreate()
spark.sparkContext.setLogLevel("ERROR")
df = spark.read.csv("nuclear_plants_small_dataset.csv", inferSchema=True,header=True)
names = df.schema.names
for name in names[1:]:
    print(name + ":")
    df.groupBy("Status").agg(func.percentile_approx(name, 0.5).alias("median"), func.mean(name).alias("mean"), func.max(name).alias("max"), func.min(name).alias("min")).show() #retrieves summary data

pandas_df = df.toPandas() #converts the dataframe from a pyspark dataframe to a pandas dataframe
with pandas.option_context('display.max_columns', None):
    mode = pandas_df.groupby(['Status']).agg(lambda x:x.value_counts().index[0])
    print("Mode: ", mode)
    var = pandas_df.var()
    print("Variance: ", var)
boxplot = pandas_df.boxplot()
correlation = pandas_df.corr()
plt.show()
with pandas.option_context('display.max_rows', None, 'display.max_columns', None):
    print(correlation)

stringIndexer = StringIndexer(inputCol="Status", outputCol="Status_index").fit(df) #adds a status index to make the status numerical for the classification models
df = stringIndexer.transform(df)
test, training = df.randomSplit([0.3, 0.7], 20) #splits the data into a test set and training set
print("Test count: ", test.count())
print("Training count: ", training.count())
va = VectorAssembler(inputCols = ["Power_range_sensor_1", "Power_range_sensor_2", "Power_range_sensor_3 ", "Power_range_sensor_4", "Pressure _sensor_1", "Pressure _sensor_2", "Pressure _sensor_3",
                                  "Pressure _sensor_4", "Vibration_sensor_1", "Vibration_sensor_2", "Vibration_sensor_3", "Vibration_sensor_4"], outputCol='sensors') #the classifiers require the data to be in vector form
test = va.transform(test)
training = va.transform(training)

df_classifier = DecisionTreeClassifier(featuresCol="sensors", labelCol="Status_index")
df_model = df_classifier.fit(training) #train the training data
df_predictions = df_model.transform(test)
df_predictions.show() #output the first 20 lines to show the prediction data
predict_accuracy1 = MulticlassClassificationEvaluator(labelCol="Status_index", metricName="accuracy")
my_eval = BinaryClassificationEvaluator(labelCol='Status_index')
weightedPrecision1 = predict_accuracy1.evaluate(df_predictions, {predict_accuracy1.metricName: "weightedPrecision"})
weightedRecall1 = predict_accuracy1.evaluate(df_predictions, {predict_accuracy1.metricName: "weightedRecall"})
FalsePos = df_predictions.where((df_predictions["prediction"] == 1.0) & (df_predictions["Status_index"] == 0.0)).count() #When the prediction is wrong about stability
TrueNeg = df_predictions.where((df_predictions["prediction"] == 0.0) & (df_predictions["Status_index"] == 0.0)).count() #When the prediction is correct about abnormality
print("Error rate", 1 - my_eval.evaluate(df_predictions), "Specificity %",TrueNeg / (TrueNeg + FalsePos), "Sensitivity", weightedRecall1)

lsvc = LinearSVC(featuresCol="sensors", labelCol="Status_index", maxIter=10, regParam=0.1)
lsvcModel = lsvc.fit(training)
predict = lsvcModel.transform(test)
predict.show()
predict_accuracy = MulticlassClassificationEvaluator(labelCol="Status_index", metricName="accuracy")
weightedPrecision2 = predict_accuracy.evaluate(predict, {predict_accuracy.metricName: "weightedPrecision"})
weightedRecall2 = predict_accuracy.evaluate(predict, {predict_accuracy.metricName: "weightedRecall"})
FalsePos = predict.where((predict["prediction"] == 1.0) & (predict["Status_index"] == 0.0)).count()
TrueNeg = predict.where((predict["prediction"] == 0.0) & (predict["Status_index"] == 0.0)).count()
print("Error rate", 1 - my_eval.evaluate(predict), "Specificity %",TrueNeg / (TrueNeg + FalsePos), "Sensitivity", weightedRecall2)

features = df.columns
features.remove("Status")

mlpc=MultilayerPerceptronClassifier(featuresCol="sensors", labelCol="Status_index", layers = [12, 5, 4, 2], blockSize=8,seed=7)
ann = mlpc.fit(training)
pred = ann.transform(test)
pred.show()
predict_accuracy = MulticlassClassificationEvaluator(labelCol="Status_index", metricName="accuracy")
weightedPrecision3 = predict_accuracy.evaluate(pred, {predict_accuracy.metricName: "weightedPrecision"})
weightedRecall3 = predict_accuracy.evaluate(pred, {predict_accuracy.metricName: "weightedRecall"})
FalsePos = pred.where((pred["prediction"] == 1.0) & (pred["Status_index"] == 0.0)).count()
TrueNeg = pred.where((pred["prediction"] == 0.0) & (pred["Status_index"] == 0.0)).count()
print("Error rate", 1 - my_eval.evaluate(pred), "Specificity %",TrueNeg / (TrueNeg + FalsePos), "Sensitivity", weightedRecall3)