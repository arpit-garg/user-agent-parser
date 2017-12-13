__author__ = 'arpitgarg'

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import SGDClassifier
import numpy as np
import argparse

parser = argparse.ArgumentParser(
    description="This code takes in tab separated test and training data of user agent strings"
                "and outputs them into new file and predicts the user agent family and "
                "major version of the user agent.", formatter_class=argparse.RawTextHelpFormatter)
parser.add_argument('--training', default=None, dest='train_data',
                    help="Enter the path of training data")
parser.add_argument('--test', default=None, dest='test_data',
                    help="Enter the path of test data")
parser.add_argument('--prediction-results', default=None, dest='pred_file',
                    help="Enter the path where you want to store prediction data")


def training(train_file):
    ua_train = []
    family_train = []
    major_version_train = []

    with open(train_file) as f:
        for line in f:
            ua_train.append(line.split("\t")[0])
            family_train.append(line.split("\t")[1])
            major_version_train.append(line.split("\t")[2].strip())

    # Using the text vectorization method to convert the words in User Agent Strings to countVectors
    # and using SGD classifier to classify them based on the user agent families

    print("Vectorizing..")
    text_clf_family = Pipeline([('vect', CountVectorizer()), (
        'clf',
        SGDClassifier(loss='hinge', penalty='l2', alpha=1e-3, random_state=42, max_iter=5, tol=None, shuffle=True)), ])

    print("Classifying..")
    text_clf_family.fit(ua_train, family_train)

    # Classifier
    return text_clf_family


def testing(test_file, clf):
    ua_test = []
    family_test = []
    major_version_test = []

    with open(test_file) as f:
        for line in f:
            ua_test.append(line.split("\t")[0])
            family_test.append(line.split("\t")[1])
            major_version_test.append(line.split("\t")[2].strip())
    print("Predicting..")
    y_test_pred = clf.predict(ua_test)
    print("Accuracy of the SGD Classifier is: {}".format(np.mean(y_test_pred == family_test)))

    # original strings, original families, predicted families, original versions
    return ua_test, family_test, y_test_pred, major_version_test


def chrome(ua_data):
    if "Chrome/" in ua_data:
        return ua_data.split("Chrome/")[1].split('.')[0]
    else:
        return 'null'


def chrome_mobile(ua_data):
    if "Chrome/" in ua_data:
        return ua_data.split("Chrome/")[1].split('.')[0]
    else:
        return 'null'


def chrome_mobil_ios(ua_data):
    if "CriOS/" in ua_data:
        return ua_data.split("CriOS/")[1].split('.')[0]
    else:
        return 'null'


def safari(ua_data):
    if "Version/" in ua_data:
        return ua_data.split("Version/")[1].split('.')[0]
    else:
        return 'null'


def safari_mobile(ua_data):
    if "Version/" in ua_data:
        return ua_data.split("Version/")[1].split('.')[0]
    elif "OS " in ua_data:
        return ua_data.split("OS ")[1].split()[0].split('_')[0]
    else:
        return 'null'


def ie(ua_data):
    if "rv:" in ua_data:
        return ua_data.split("rv:")[1].split('.')[0]
    elif "MSIE " in ua_data and "Trident/" in ua_data:
        return ua_data.split("MSIE ")[1].split('.')[0]
    elif "MSIE " in ua_data:
        return ua_data.split("MSIE ")[1].split('.')[0]
    elif "Trident/" in ua_data:
        return ua_data.split("Trident/")[1].split('.')[0]
    else:
        return 'null'


def ie_mobile(ua_data):
    if "Trident/" in ua_data:
        return ua_data.split("IEMobile/")[1].split('.')[0]
    elif "IEMobile " in ua_data:
        return ua_data.split("IEMobile ")[1].split('.')[0]
    else:
        return 'null'


def edge(ua_data):
    if "Edge/" in ua_data:
        return ua_data.split("Edge/")[1].split('.')[0]
    else:
        return 'null'


def edge_mobile(ua_data):
    if "Edge/" in ua_data:
        return ua_data.split("Edge/")[1].split('.')[0]
    else:
        return 'null'


def firefox(ua_data):
    if "Firefox/" in ua_data:
        return ua_data.split("Firefox/")[1].split('.')[0]
    else:
        return 'null'


def firefox_mobile(ua_data):
    if "Firefox/" in ua_data:
        return ua_data.split("Firefox/")[1].split('.')[0]
    else:
        return 'null'


def firefox_ios(ua_data):
    if "FxiOS/" in ua_data:
        return ua_data.split("FxiOS/")[1].split('.')[0]
    else:
        return 'null'


def opera(ua_data):
    if "OPR/" in ua_data:
        return ua_data.split("OPR/")[1].split('.')[0]
    elif "Version/" in ua_data:
        return ua_data.split("Version/")[1].split('.')[0]
    elif "Opera/" in ua_data:
        return ua_data.split("Opera/")[1].split('.')[0]
    elif "Opera " in ua_data:
        return ua_data.split("Opera ")[1].split('.')[0]
    else:
        return 'null'


def opera_mobile(ua_data):
    if "OPR/" in ua_data:
        return ua_data.split("OPR/")[1].split('.')[0]
    elif "Version/" in ua_data:
        return ua_data.split("Version/")[1].split('.')[0]
    else:
        return 'null'


def opera_mini(ua_data):
    if "Opera Mini/" in ua_data:
        return ua_data.split("Opera Mini/")[1].split('.')[0]
    elif "OPiOS/" in ua_data:
        return ua_data.split("OPiOS/")[1].split('.')[0]
    else:
        return 'null'


def uc(ua_data):
    if "UCBrowser/" in ua_data:
        return ua_data.split("UCBrowser/")[1].split('.')[0]
    elif "UCBrowser" in ua_data:
        return ua_data.split("UCBrowser")[1].split('.')[0]
    elif "UCWEB" in ua_data:
        return ua_data.split("UCWEB")[1].split('.')[0]
    elif "UC Browser" in ua_data:
        return ua_data.split("UC Browser")[1].split('.')[0]
    else:
        return 'null'


def sogou(ua_data):
    if "MetaSr " in ua_data:
        return ua_data.split("MetaSr ")[1].split('.')[0]
    else:
        return 'null'


def qq(ua_data):
    if " QQBrowser/" in ua_data:
        return ua_data.split(" QQBrowser/")[1].split('.')[0]
    else:
        return 'null'


def qq_mobile(ua_data):
    if "MQQBrowser/" in ua_data:
        return ua_data.split("MQQBrowser/")[1].split('.')[0]
    else:
        return 'null'


def maxthon(ua_data):
    if "Maxthon/" in ua_data:
        return ua_data.split("Maxthon/")[1].split('.')[0]
    elif "Maxthon " in ua_data:
        return ua_data.split("Maxthon ")[1].split('.')[0]
    elif "Maxthon;" in ua_data:
        return 0
    else:
        return 'null'


def aol(ua_data):
    if "AOL " in ua_data:
        return ua_data.split("AOL ")[1].split('.')[0]
    else:
        return 'null'


def facebook(ua_data):
    if "FBAV/" in ua_data:
        return ua_data.split("FBAV/")[1].split('.')[0]
    else:
        return 'null'


def applemail(ua_data):
    if "AppleWebKit/" in ua_data:
        return ua_data.split("AppleWebKit/")[1].split('.')[0]
    else:
        return 'null'


def puffin(ua_data):
    if "Puffin/" in ua_data:
        return ua_data.split("Puffin/")[1].split('.')[0]
    else:
        return 'null'


def android(ua_data):
    if "Version/" in ua_data:
        return ua_data.split("Version/")[1].split('.')[0]
    elif "Android/" in ua_data:
        return ua_data.split("Android/")[1].split('.')[0]
    elif "Android " in ua_data:
        return ua_data.split("Android ")[1].split('.')[0]
    else:
        return 'null'


def yandex(ua_data):
    if "YandexSearch/" in ua_data:
        return ua_data.split("YandexSearch/")[1].split('.')[0]
    else:
        return 'null'


def bb(ua_data):
    if "Version/" in ua_data:
        return ua_data.split("Version/")[1].split('.')[0]
    else:
        return 'null'


def silk(ua_data):
    if "Silk/" in ua_data:
        return ua_data.split("Silk/")[1].split('.')[0]
    else:
        return 'null'


def predictVersion(ua_data, family):
    predicted_version = [0] * len(ua_data)
    print("Extracting Versions from Predicted UA String")
    if len(ua_data) == len(family):
        for i in range(len(family)):
            version = ''

            if family[i] == 'Chrome':
                version = chrome(ua_data[i])

            elif family[i] == 'Chrome Mobile':
                version = chrome_mobile(ua_data[i])

            elif family[i] == 'Chrome Mobile iOS':
                version = chrome_mobil_ios(ua_data[i])

            elif family[i] == 'Safari':
                version = safari(ua_data[i])

            elif family[i] == 'Mobile Safari':
                version = safari_mobile(ua_data[i])

            elif family[i] == 'IE':
                version = ie(ua_data[i])

            elif family[i] == 'IE Mobile':
                version = ie_mobile(ua_data[i])

            elif family[i] == 'Edge':
                version = edge(ua_data[i])

            elif family[i] == 'Edge Mobile':
                version = edge_mobile(ua_data[i])

            elif family[i] == 'Firefox':
                version = firefox(ua_data[i])

            elif family[i] == 'Firefox Mobile':
                version = firefox_mobile(ua_data[i])

            elif family[i] == 'Firefox iOS':
                version = firefox_ios(ua_data[i])

            elif family[i] == 'Opera':
                version = opera(ua_data[i])

            elif family[i] == 'Opera Mobile':
                version = opera_mobile(ua_data[i])

            elif family[i] == 'Opera Mini':
                version = opera_mini(ua_data[i])

            elif family[i] == 'UC Browser':
                version = uc(ua_data[i])

            elif family[i] == 'Sogou Explorer':
                version = sogou(ua_data[i])

            elif family[i] == 'QQ Browser':
                version = qq(ua_data[i])

            elif family[i] == 'QQ Browser Mobile':
                version = qq_mobile(ua_data[i])

            elif family[i] == 'Maxthon':
                version = maxthon(ua_data[i])

            elif family[i] == 'AOL':
                version = aol(ua_data[i])

            elif family[i] == 'Facebook':
                version = facebook(ua_data[i])

            elif family[i] == 'AppleMail':
                version = applemail(ua_data[i])

            elif family[i] == 'Puffin':
                version = puffin(ua_data[i])

            elif family[i] == 'Android':
                version = android(ua_data[i])

            elif family[i] == 'YandexSearch':
                version = yandex(ua_data[i])

            elif family[i] == 'BlackBerry WebKit':
                version = bb(ua_data[i])

            elif family[i] == 'Amazon Silk':
                version = silk(ua_data[i])
            else:
                version = 'null'

            predicted_version[i] = version
    return predicted_version


if __name__ == '__main__':

    args = parser.parse_args()
    train_data = ''
    test_data = ''
    pred_file = ''

    if args.train_data:
        train_data = args.train_data
    if args.test_data:
        test_data = args.test_data
    if args.pred_file:
        pred_file = args.pred_file

    clf = training(train_data)

    ua_test, family_test, family_test_pred, major_version_test = testing(test_data, clf)

    major_version_test_pred = predictVersion(ua_test, family_test_pred)

    with open(pred_file, 'w') as f:
        for i in range(len(ua_test)):
            f.write(ua_test[i] + "\t" + family_test[i] + "\t" + major_version_test[i] + "\t" +
                    family_test_pred[i] + "\t" + major_version_test_pred[i] + "\n")
    print("Predicitons saved to {}".format(pred_file))
