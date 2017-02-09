import org.apache.spark.ml.feature.VectorSlicer
import org.apache.spark.ml.regression.LinearRegression
import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.sql.SaveMode

val result = StringBuilder.newBuilder
result.append("r2-train, rmse-train, mse-train, mae-train, r2-test, rmse-test, mse-test, mae-test\n")
var training = sqlContext.read.format("libsvm").load("train.csv")
var test = sqlContext.read.format("libsvm").load("test.csv")

var slicer = new VectorSlicer().setInputCol("features").setOutputCol("features_temp")

//Features selection
slicer.setIndices(Array(0,1,2,3,9,10,11,13,14,15,16,17,18,19,20,21,22)) // M4
var output = slicer.transform(training)
training = output.select(output("label"),output("features_temp"))
training = training.withColumnRenamed("features_temp", "features")
var output = slicer.transform(test)
test = output.select(output("label"),output("features_temp"))
test = test.withColumnRenamed("features_temp", "features")

var lr = new LinearRegression()
lr.setMaxIter(10)
lr.setRegParam(0.3)
lr.setElasticNetParam(0.8)

// Fit the model
var lrModel = lr.fit(training)


var trainSummary = lrModel.summary
var trainR2=(math rint trainSummary.r2 * 100) / 100
var trainRmse= (math rint trainSummary.rootMeanSquaredError * 100) / 100
var trainMse= (math rint trainSummary.meanSquaredError * 100) / 100
var trainMae=(math rint trainSummary.meanAbsoluteError * 100) / 100
var trainPrediction = trainSummary.predictions
trainPrediction= trainPrediction.withColumn("residuals",trainPrediction("label")-trainPrediction("prediction"))
trainPrediction=trainPrediction.withColumn("set",lit("train"))

var testPrediction = lrModel.transform(test)
var evaluator = new RegressionEvaluator()
evaluator.setLabelCol("label")
evaluator.setPredictionCol("prediction")
evaluator.setMetricName("r2")
var rsquare = (math rint evaluator.evaluate(testPrediction) * 100) / 100
evaluator.setMetricName("rmse")
var rmse = (math rint evaluator.evaluate(testPrediction) * 100) / 100
evaluator.setMetricName("mse")
var mse =(math rint evaluator.evaluate(testPrediction) * 100) / 100
evaluator.setMetricName("mae")
var mae = (math rint evaluator.evaluate(testPrediction) * 100) / 100
testPrediction= testPrediction.withColumn("residuals",testPrediction("label")-testPrediction("prediction"))
testPrediction=testPrediction.withColumn("set",lit("test"))
var allPrediction = trainPrediction.unionAll(testPrediction)

allPrediction.write.mode(SaveMode.Overwrite).format("com.databricks.spark.csv").option("header", "true").save("global.csv")

result.append(s"${trainR2},${trainRmse},${trainMse},${trainMae}, ${rsquare},${rmse},${mse},${mae}\n")
println(result)
