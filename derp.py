from sklearn import metrics
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.cross_validation import train_test_split
from sklearn.svm import SVC


x = [[1,2,3],[3,3,2],[8,8,7],[3,7,1],[4,5,6]]
y = [['bar','foo'],['bar'],['foo'],['foo','jump'],['bar','fox','jump']]

mlb = MultiLabelBinarizer()
y_enc = mlb.fit_transform(y)


print(x)
print(type(x))

print(y_enc)
print(type(y_enc))



clf = OneVsRestClassifier(SVC(probability=True))
clf.fit(x, y_enc)
predictions = clf.predict([[1,2,3]])

#my_metrics = metrics.classification_report(test_y, predictions)
print (predictions)