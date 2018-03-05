import pickle
from pyspark import SparkContext
from pyspark import SQLContext
from pyspark.ml.classification import NaiveBayes
from pyspark.ml.classification import MultilayerPerceptronClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
import time


def load_tool_classification_result():
    with open('D:\\pythonspace\\evaluation\\result.dat', 'r', encoding='utf-8') as f:
        gc = f.read()
    tmp_list = gc[0:].split('\n')
    result = []
    for i in tmp_list:
        if len(i) > 0:
            i = i[1:-1]
            parts = i.split(", ")
            if parts[0] >= parts[1]:
                result.append(1.0)
            else:
                result.append(0.0)
    f.close()
    return result


def compare_classification_with_tool(sql_context, model):
    pkl_file2 = open('D:\\pythonspace\\comment\\comment_evaluation_preprocessed_1000f_1000fbi.pkl', 'rb')
    compare_comments_process = pickle.load(pkl_file2)
    pkl_file2.close()
    data_compare = sql_context.createDataFrame(compare_comments_process)
    result = model.transform(data_compare)
    result_pandas = result.select('prediction').toPandas()
    result_tool = load_tool_classification_result()
    total = len(result_tool)
    pos_dif = 0
    same_count = 0
    diff = []
    for i in range(0,total):
        if result_tool[i] == result_pandas.loc[i, 'prediction']:
            same_count += 1
        else:
            diff.append(i)
            if result_tool[i] == 1.0:
                pos_dif += 1
    print()
    print('Same prediction rate : '+str(same_count/(total+0.0)))
    print(diff)
    print(pos_dif)
    print(len(diff)-pos_dif)


def predict_comment(sql_context, model):
    pkl_file2 = open('D:\\pythonspace\\comment\\comment_test_preprocessed_250f_250fbi.pkl', 'rb')
    test_comments_process = pickle.load(pkl_file2)
    pkl_file2.close()
    data_test = sql_context.createDataFrame(test_comments_process)
    result = model.transform(data_test)
    result.select('features', 'prediction').show()


def evaluate_classification(predictions):
    evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="f1")
    # print(evaluator.explainParams())
    f1 = evaluator.evaluate(predictions)
    evaluator.setMetricName('weightedPrecision')
    weighted_precision = evaluator.evaluate(predictions)
    evaluator.setMetricName('weightedRecall')
    weighted_recall = evaluator.evaluate(predictions)
    evaluator.setMetricName('accuracy')
    accuracy = evaluator.evaluate(predictions)
    print()
    print("Test set accuracy = " + str(accuracy))
    print("Test set weightedPrecision = " + str(weighted_precision))
    print("Test set weightedRecall = " + str(weighted_recall))
    print("Test set f1 = " + str(f1))


def naive_bayes_classify(comment_preprocessed):
    sc = SparkContext(appName="Classification")
    sql_context = SQLContext(sc)
    data = sql_context.createDataFrame(comment_preprocessed)

    train, test = data.randomSplit([0.7, 0.3], 1234)
    nb = NaiveBayes(smoothing=1.0, modelType="multinomial")
    model = nb.fit(train)

    predictions = model.transform(test)
    evaluate_classification(predictions)

    time.sleep(1)
    # predict_comment(sql_context, model)
    compare_classification_with_tool(sql_context, model)


def multilayer_perceptron_classify(comment_preprocessed):
    sc = SparkContext(appName="Classification")
    sql_context = SQLContext(sc)
    data = sql_context.createDataFrame(comment_preprocessed)

    train, test = data.randomSplit([0.7, 0.3], 1234)
    layers = [len(comment_preprocessed[0].features), 11, 2]
    # sqrt(2000) = 45, sqrt(4000) = 63, log(2000, 2) = 11
    trainer = MultilayerPerceptronClassifier(maxIter=100, layers=layers, blockSize=128, seed=1234)
    model = trainer.fit(train)
    predictions = model.transform(test)
    evaluate_classification(predictions)

    time.sleep(1)
    # predict_comment(sql_context, model)
    compare_classification_with_tool(sql_context, model)


if __name__ == '__main__':
    pkl_file = open('D:\\pythonspace\\comment\\comment_train_preprocessed_1000f_1000fbi.pkl', 'rb')
    comment_preprocessed = pickle.load(pkl_file)
    pkl_file.close()

    naive_bayes_classify(comment_preprocessed)

    multilayer_perceptron_classify(comment_preprocessed)
