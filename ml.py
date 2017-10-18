import json

from skmultilearn.problem_transform import BinaryRelevance
from sklearn.naive_bayes import MultinomialNB
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC
import numpy as np

class ML(object):
    def __init__(self):
        print("123")
        self.counts = {}
        self.champions = []
        self.builds = []
        self.results = []

        with open('short') as f:
            self.lines = f.readlines()

        for line in self.lines:
            self.matchid = line.split('\t')[0]
            self.js = line.split('\t')[1][1:-2]
            self.decoded = json.loads(self.js)

            for i in range(0,10):
                self.player = self.decoded[i]
                self.stats = self.player['stats']

                self.champion = self.player['championId']
                self.item0 = self.stats['item0']
                self.item1 = self.stats['item1']
                self.item2 = self.stats['item2']
                self.item3 = self.stats['item3']
                self.item4 = self.stats['item4']
                self.item5 = self.stats['item5']
                self.win = [1] if self.stats['winner'] == True else [0]

                self.champions.append([self.champion])
                self.build = [self.item0, self.item1, self.item2, self.item3, self.item4, self.item5]
                self.build.sort()
                self.build = [y for y in self.build if y != 0]
                self.builds.append(self.build)
                self.results.append(self.win)

        self.ml(self.champions, self.results, self.builds, len(self.champions))

    def ml(self, cs, rs, bs, len):
        x_train=[]
        y_train=[]

        for i in range(0, len):
            champ_train = cs[i]
            champ_train.extend(rs[i])
            build_train = self.oneHot(bs[i], 5000)

            #for j in bs[i]:
            #    x_train.append(champ_train)
            #    y_train.append([j])

            x_train.append(champ_train)
            y_train.append(build_train)


        #print(x_train)

        y_train = np.array(y_train)
        #print(y_train.sum(axis=0).all())
        #print(y_train)
        #print(type(y_train))

        #clf = BinaryRelevance(MultinomialNB())
        clf = OneVsRestClassifier(SVC(probability=True))
        clf.fit(x_train, y_train)

        test = [[236, 1]]

        print(test)

        res = clf.predict(test)

        print(res)

    def oneHot(self, arr, size):
        out = []

        for i in range(0, size):
            if i in arr:
                out.append(1)
            else:
                out.append(0)

        return out

test = ML()