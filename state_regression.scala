import org.apache.spark.ml.regression.LinearRegression
import org.apache.spark.ml.evaluation.RegressionEvaluator

var testBase = "/store/train_"
var testBase = "/store/test_"
val result = StringBuilder.newBuilder
result.append("Store, r2-test,mae\n")
for (i <- 1 to 45 ){

var trainFile = trainBase.concat(i.toString).concat("_conv.csv")
var testFile = testBase.concat(i.toString).concat("_conv.csv")
println("Model for store "+i+". Test File:"+testFile)

//Loading
var training = sqlContext.read.format("libsvm").load(trainFile)
var test = sqlContext.read.format("libsvm").load(testFile)

var slicer = new VectorSlicer().setInputCol("features").setOutputCol("features_temp")
slicer.setIndices(Array(0,1,2,3,9,10,11,13,14,15,16,17,18,19,20,21,22)) // M4
var output = slicer.transform(training)
var output = slicer.transform(test)
training = output.select(output("label"),output("features_temp"))
training = training.withColumnRenamed("features_temp", "features")
var output = slicer.transform(test)
test = output.select(output("label"),output("features_temp"))
test = test.withColumnRenamed("features_temp", "features")

// Fit the model
var lrModel = lr.fit(training)

var trainSummary = lrModel.summary
var trainR2=(math rint trainSummary.r2 * 100) / 100
var trainRmse= (math rint trainSummary.rootMeanSquaredError * 100) / 100
var trainMse= (math rint trainSummary.meanSquaredError * 100) / 100
var trainMae=(math rint trainSummary.meanAbsoluteError * 100) / 100
var trainPrediction = trainSummary.predictions

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
result.append(s"${i}, ${rsquare}, ${mae}\n")
}
println(result)
