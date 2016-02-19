package com.huntdreams.spark.rdf

import org.apache.spark.mllib.evaluation.MulticlassMetrics
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.tree.DecisionTree
import org.apache.spark.mllib.tree.model.DecisionTreeModel
import org.apache.spark.rdd.RDD
import org.apache.spark.{SparkConf, SparkContext}

/**
 * 决策树和随即森林
 *
 * @author tyee.noprom@qq.com
 * @time 2/16/16 10:08 AM.
 */
object RDF {

  def main(args: Array[String]) {
    val sc = new SparkContext(new SparkConf().setAppName("RDF"))
    val rawData = sc.textFile("resources/covtype.data");

    // 准备数据
    val data = rawData.map{ line =>
      val values = line.split(",").map(_.toDouble)
      // .init 返回除了最后一列的元素,并将其转为vector
      val featureVector = Vectors.dense(values.init)
      val label = values.last - 1
      // 在监督性学习里面使用LabeledPoint
      // regression and classification
      LabeledPoint(label, featureVector)
    }

    // 拆分数据，80%训练，10%交叉验证，10%测试
    val Array(trainData, cvData, testData) = data.randomSplit(Array(0.8, 0.1, 0.1))
    trainData.cache()
    cvData.cache()
    testData.cache()

    // 开始训练并且分析数据
  }

  def simpleDecisionTree(trainData: RDD[LabeledPoint], cvData: RDD[LabeledPoint]): Unit = {
    // 创建一个简单默认的决策树模型
    val model = DecisionTree.trainClassifier(trainData, 7, Map[Int, Int](), "gini", 4, 100)

  }

  def getMetrics(model: DecisionTreeModel, data: RDD[LabeledPoint]): MulticlassMetrics = {
    val predictionsAndLabels = data.map(example =>
      (model.predict(example.features), example.label)
    )
    new MulticlassMetrics(predictionsAndLabels)
  }
}