#-------------------------------------------------------------------------------
# Name:        module1
# Purpose:
#
# Author:      Administrator
#
# Created:     30/10/2012
# Copyright:   (c) Administrator 2012
# Licence:     <your licence>
#-------------------------------------------------------------------------------

import re
import math

document = 'hello world, this is my first classifier classifier'

def sampletrain(cl):
    cl.train('Nobody owns the water','good')
    cl.train('the quick rabbit jumps fences.','good')
    cl.train('buy pharmeceuticals now','bad')
    cl.train('make quick money at the online casino','bad')
    cl.train('the quick brown fox jumps','good')

def sampletrain_from_csv(cl,file_name,item_id_field,category_field):
    file = open(file_name,'r')
    for n in file:
        cl.train(n[item_id_field], n[category_field])


def get_sku_features(item_id):
    return item_id.split('-')

def get_words(doc):
    splitter = re.compile('\\W*')
    words = [s.lower() for s in splitter.split(doc)
                        if len(s) > 2 and len(s) < 19]
    words_dict = {}
    for w in words:
        if words_dict.has_key(w):
            words_dict[w] += 1
        else:
            words_dict[w] = 1
    return words_dict

class classifier:
    def __init__(self,getfeatures,filename=None):
        self.feat_cat_dict = {} #Counts of feature/category combinations
        self.cat_dict = {} #count of documsents in each category
        self.getfeatures = getfeatures

    #increment feature|category pair counter
    def incr_feat_cat(self,feat,cat):
        self.feat_cat_dict.setdefault(feat,{})
        self.feat_cat_dict[feat].setdefault(cat,0)
        self.feat_cat_dict[feat][cat] += 1

    #increment category counter
    def incr_cat_count(self,cat):
        self.cat_dict.setdefault(cat,0)
        self.cat_dict[cat] += 1

    #number of times a feature has appeared in a category
    def feat_cat_count(self,feat,cat):
        if self.feat_cat_dict.has_key(feat) and self.feat_cat_dict[feat].has_key(cat):
            return float(self.feat_cat_dict[feat][cat])
        return 0.0

    #number of items in a category
    def cat_count(self,cat):
        if self.cat_dict.has_key(cat):
            return float(self.cat_dict[cat])
        return 0.0

    #total number of items
    def totalcount(self):
        return sum(self.cat_dict.values())

    def categories(self):
        return self.cat_dict.keys()

    def keys(self):
        return self.feat_cat_dict.keys()

    def train(self,item,cat):
        features = self.getfeatures(item)
        for f in features:
            self.incr_feat_cat(f,cat)
        self.incr_cat_count(cat)

    def fprob(self,feat,cat):
        if self.cat_count(cat) == 0: return 0
        return self.feat_cat_count(feat,cat)/self.cat_count(cat)

    def weightedprob(self, feat, cat, prf,weight = 1.0, ap=.5):
        #prf stores currently probability function
        basicprob = prf(feat,cat)

        # Count the number of times this feature has appeared in all categories
        total = sum([self.feat_cat_count(feat,c) for c in self.categories()])

        #calculated the weighted Average
        bp = ((weight*ap) + (total*basicprob))/(weight + total)

        return bp #bp = Base Probability


class naivebayes(classifier):
    #bayes formula: P(A|B) = P(B|A)*P(A)/P(B)
    # P(Category|Document) = P(Document|Category)*P(Category)
    # Can drop P(Document), because the same for all Categories

    def __init__(self,getfeatures):
        classifier.__init__(self,getfeatures)
        self.thresholds = {}

    def setthreshold(self,cat,t):
        self.thresholds[cat] = t

    def getthreshold(self,cat):
        if cat not in self.thresholds: return 1.0
        return self.thresholds[cat]

    def docprob(self,item,cat): #equivalent P(Document|Category)
        features = self.getfeatures(item)
        p=1
        for f in features:
            p*=self.weightedprob(f,cat,self.fprob)
        return p

    def prob(self,item,cat):
        p_cat = self.cat_count(cat)/self.totalcount()
        doc_prob = self.docprob(item,cat)
        return p_cat*doc_prob


    def classify(self,item,default=None):
        probs = {}
        # Find the category with the highest probability
        max = 0.0
        for cat in self.categories():
            probs[cat] = self.prob(item,cat)
            if probs[cat] > max:
                max = probs[cat]
                best = cat

        #ensure probability exceeds threshold*next_best
        for cat in self.categories():
            if cat == best: continue
            if probs[cat]*self.getthreshold(best) > probs[best]: return default
        print probs
        return best





