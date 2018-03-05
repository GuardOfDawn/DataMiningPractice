import pickle
from pyspark.sql import Row
from pyspark.ml.linalg import Vectors
import math


def load_comment_words_for_classify():
    greats = []  # 正面评价分词结果
    with open('D:\\pythonspace\\great.dat', 'r', encoding='utf-8') as f:
        gc = f.read()
    gcs = gc[1:].split('][')
    for gcsi in gcs:
        gcsi = gcsi[1:]
        gcsi = gcsi[:-1]
        great = gcsi.split("', '")
        greats.append(great)
    f.close()
    bads = []  # 负面评价分词结果
    with open('D:\\pythonspace\\bad.dat', 'r', encoding='utf-8') as f:
        bc = f.read()
    bcs = bc[1:].split('][')
    for bcsi in bcs:
        bcsi = bcsi[1:]
        bcsi = bcsi[:-2]
        bad = bcsi.split("', '")
        bads.append(bad)
    return greats, bads


def load_test_comments(file_name):
    comments = []
    with open(file_name, 'r', encoding='utf-8') as f:
        gc = f.read()
    gcs = gc[1:].split('][')
    for gcsi in gcs:
        gcsi = gcsi[1:]
        gcsi = gcsi[:-1]
        great = gcsi.split("', '")
        comments.append(great)
    f.close()
    return comments


def comment_train_vector(comment_preprocess, comments_list,
                         type, label, feature_list, feature_bi_list):
    vector_length = len(feature_list)+len(feature_bi_list)
    for words in comments_list:
        index_vector = []
        feature_vector = []
        index = 0
        for feature in feature_list:
            if feature[0] in words:
                index_vector.append(index)
                feature_vector.append(1.0)
            index += 1
        for feature_bi in feature_bi_list:
            if feature_bi[0][0] != feature_bi[0][1]:
                if (feature_bi[0][0] in words) and (feature_bi[0][1] in words):
                    index_vector.append(index)
                    feature_vector.append(1.0)
            else:
                if words.count(feature_bi[0][0]) >= 2:
                    index_vector.append(index)
                    feature_vector.append(1.0)
            index += 1
        if type == 'train':
            comment_preprocess.append(
                Row(label=label, features=Vectors.sparse(vector_length, index_vector, feature_vector))
            )
        elif type == 'test':
            comment_preprocess.append(Row(features=Vectors.sparse(vector_length, index_vector, feature_vector)))
            # for i in index_vector:
            #     if i < len(feature_list):
            #         print(feature_list[i][0], end=' ')
            #     else:
            #         print(feature_bi_list[i-len(feature_list)][0], end=' ')
            # print()


def comment_train_preprocess(features, features_bi):
    comments_positive, comments_negative = load_comment_words_for_classify()
    comment_train_preprocessed = []
    comment_train_vector(comment_train_preprocessed, comments_positive, 'train', 1, features, features_bi)
    comment_train_vector(comment_train_preprocessed, comments_negative, 'train', 0, features, features_bi)

    output = open('D:\\pythonspace\\comment\\comment_train_preprocessed_250f_250fbi.pkl', 'wb')
    pickle.dump(comment_train_preprocessed, output)
    output.close()


def comment_test_preprocess(features, features_bi, file_name):
    test_comments = load_test_comments(file_name)
    test_comments_preprocess = []
    comment_train_vector(test_comments_preprocess, test_comments, 'test', -1, features, features_bi)

    # print(test_comments_preprocess)

    output = open('D:\\pythonspace\\comment\\comment_evaluation_preprocessed_250f_250fbi.pkl', 'wb')
    pickle.dump(test_comments_preprocess, output)
    output.close()


if __name__ == '__main__':
    pkl_file = open('D:\\pythonspace\\feature\\features_filter250_tfidf.pkl', 'rb')
    features = pickle.load(pkl_file)
    pkl_file.close()
    pkl_file = open('D:\\pythonspace\\feature\\features_bi_filter250_tfidf.pkl', 'rb')
    features_bi = pickle.load(pkl_file)
    pkl_file.close()

    # 处理训练数据
    comment_train_preprocess(features, features_bi)
    # 处理测试数据
    comment_test_preprocess(features, features_bi, 'D:\\pythonspace\\test.dat')
    # 处理检验数据
    comment_test_preprocess(features, features_bi, 'D:\\pythonspace\\evaluation\\contentsfc.dat')
