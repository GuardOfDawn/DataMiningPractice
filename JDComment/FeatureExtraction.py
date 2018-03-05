from nltk.collocations import BigramCollocationFinder
from nltk.metrics import BigramAssocMeasures
import math
import pickle


def load_comment_words(file_path_list):
    all_comment_words = []
    comment_words_one_file = []  # 正面评价分词结果
    for file_path in file_path_list:
        with open(file_path, 'r', encoding='utf-8') as f:
            gc = f.read()
        gcs = gc[1:].split('][')
        for gcsi in gcs:
            gcsi = gcsi[1:]
            gcsi = gcsi[:-1]
            great = gcsi.split("', '")
            comment_words_one_file.append(great)
        all_comment_words.append(comment_words_one_file)
        f.close()
    return all_comment_words


def load_stop_words(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        gc = f.read()
    stop_words = gc[1:].split('\n')
    f.close()
    return stop_words


def load_brand_words(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        gc = f.read()
    words = gc[1:].split('\n')
    brand_words = []
    for word in words:
        word = word.replace(' ', '')
        brand_words.append(word[1:len(word)-2])
    f.close()
    return brand_words


def remove_target_word(words, words_remove):
    for re in words_remove:
        while re in words:
            words.remove(re)
    return words


def frequency_of_words(words, feature_list):
    for word in words:
        if type(word) == tuple:
            word = list(word)
            word.sort()
            word = tuple(word)
        if feature_list.__contains__(word):
            feature_list[word][0] += 1
        else:
            feature_list[word] = [1, 0, 0.0]
    word_occurred = []
    for word in words:
        if type(word) == tuple:
            word = list(word)
            word.sort()
            word = tuple(word)
        if word not in word_occurred:
            feature_list[word][1] += 1
            word_occurred.append(word)


def bigram(words, features_bi, score_fn=BigramAssocMeasures.chi_sq, n=10):
    bigram_finder = BigramCollocationFinder.from_words(words)
    try:
        bigrams = bigram_finder.nbest(score_fn, n)
        frequency_of_words(bigrams, features_bi)
    except:
        # print(words)
        return


def cal_max_words_times(feature_list):
    max_words_times = 0
    max_word = ''
    for word in feature_list:
        if feature_list[word][0] > max_words_times:
            max_words_times = feature_list[word][0]
            max_word = word
    return max_words_times, max_word


def feature_extract_tfidf(feature_list, total_comments, file_to_save):
    max_word_count, max_word = cal_max_words_times(feature_list)
    for feature in feature_list:
        tf = feature_list[feature][0]/(max_word_count + 0.0)
        idf = math.log(total_comments/(feature_list[feature][1]+1.0), 10)
        feature_list[feature][2] = tf*idf
    feature_list_sorted = sorted(feature_list.items(), key=lambda d: d[1][2], reverse=True)
    # 保存所有feature
    file_object = open(file_to_save, 'w')
    for fre in feature_list_sorted:
        file_object.write(str(fre)+"\n")
    file_object.close()
    return feature_list_sorted[:2000]


if __name__ == '__main__':
    stop_words_chinese = load_stop_words('D:\\pythonspace\\stop_words_chinese.dat')
    brand_words_chinese = load_brand_words('D:\\pythonspace\\brand.txt')

    words_remove_list = ['', '手机', '京东', '买', '天', '说']
    for remove in words_remove_list:
        stop_words_chinese.append(remove)
    for brand in brand_words_chinese:
        stop_words_chinese.append(brand)
    file_root = 'D:\\pythonspace\\comment_words\\'
    file_name_list = ['great.dat', 'bad.dat']
    # file_name_list = ['qc0.dat', 'qc1.dat', 'qc2.dat', 'qc3.dat', 'qc4.dat', 'qc5.dat', 'qc6.dat', 'qc7.dat', 'qc8.dat', 'qc9.dat']
    comment_file_path_list = []
    for file in file_name_list:
        comment_file_path_list.append(file_root+file)
    comment_words = load_comment_words(comment_file_path_list)
    features = {}
    features_bi = {}
    comment_total_count = 0
    print('start extracting features...')
    for words_file in comment_words:
        comment_total_count += len(words_file)
        for words in words_file:
            words_after = remove_target_word(words, stop_words_chinese)
            frequency_of_words(words_after, features)
            bigram(words_after, features_bi)
    print('start sorting features...')
    features_save_file = 'D:\\pythonspace\\feature\\feature_all.dat'
    features_bi_save_file = 'D:\\pythonspace\\feature\\feature_bi_all.dat'
    features_filter = feature_extract_tfidf(features, comment_total_count, features_save_file)
    features_bi_filter = feature_extract_tfidf(features_bi, comment_total_count, features_bi_save_file)
    print(features_filter)
    print(features_bi_filter)

    output = open('D:\\pythonspace\\feature\\features_filter2000_tfidf.pkl', 'wb')
    pickle.dump(features_filter, output)
    output.close()
    output = open('D:\\pythonspace\\feature\\features_bi_filter2000_tfidf.pkl', 'wb')
    pickle.dump(features_bi_filter, output)
    output.close()

    output = open('D:\\pythonspace\\feature\\features_filter1000_tfidf.pkl', 'wb')
    pickle.dump(features_filter[:1000], output)
    output.close()
    output = open('D:\\pythonspace\\feature\\features_bi_filter1000_tfidf.pkl', 'wb')
    pickle.dump(features_bi_filter[:1000], output)
    output.close()

    output = open('D:\\pythonspace\\feature\\features_filter500_tfidf.pkl', 'wb')
    pickle.dump(features_filter[:500], output)
    output.close()
    output = open('D:\\pythonspace\\feature\\features_bi_filter500_tfidf.pkl', 'wb')
    pickle.dump(features_bi_filter[:500], output)
    output.close()

    output = open('D:\\pythonspace\\feature\\features_filter250_tfidf.pkl', 'wb')
    pickle.dump(features_filter[:250], output)
    output.close()
    output = open('D:\\pythonspace\\feature\\features_bi_filter250_tfidf.pkl', 'wb')
    pickle.dump(features_bi_filter[:250], output)
    output.close()
