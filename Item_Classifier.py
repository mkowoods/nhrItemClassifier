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
import inspect

document = 'hello world, this is my first classifier classifier'

def sampletrain(cl):
    cl.train('Nobody owns the water','good')
    cl.train('the quick rabbit jumps fences.','good')
    cl.train('buy pharmeceuticals now','bad')
    cl.train('make quick money at the online casino','bad')
    cl.train('the quick brown fox jumps','good')

def sampletrain_from_csv(cl,file_name,item_field,category_field,desc_field=None):
    file = open(file_name,'r')

    for n in file:
        n  = n.strip('\n')
        n = n.split(',')
        input_string = n[item_field]
        #print input_string
        if desc_field: input_string += '|'+n[desc_field]
        cl.train(input_string, n[category_field])
    file.close()

def test_data(file_name,item_field,category_field,desc_field=None):
    file = open(file_name,'r')
    test_list = []
    for n in file:
        n  = n.strip('\n')
        n = n.split(',')
        item = n[item_field]
        cat = n[category_field]
        item_cat_tuple = (item,cat)
        test_list.append(item_cat_tuple)
    file.close()
    return test_list


def test_bayes_model(cl,test_data,set_thresh = None):
    """test of open"""
    correct_results = 0
    incorrect_results = 0
    insufficient_data = 0
    total = 0

    default = inspect.getargspec(cl.classify)[3][0]

    for test_item,test_cat in test_data:
        guess,guess_prob = cl.classify(test_item)
        #print guess, test_item, test_cat
        if test_cat == guess:
            correct_results +=1
        elif guess == default:
            insufficient_data += 1
        else:
            incorrect_results += 1
        total += 1

    rate_incorrect = float(incorrect_results)/float(total)
    return rate_incorrect


def test_bayes_model2(cl,test_data,set_thresh = None):
    """evaluates success of alg"""
    correct_results = 0
    incorrect_results = 0
    insufficient_data = 0
    total = 0

    default = inspect.getargspec(cl.classify)[3][0]

    for test_item,test_cat in test_data:
        guess,guess_prob = cl.classify(test_item)
        #print guess, test_item, test_cat
        if test_cat == guess:
            correct_results +=1
        elif guess == default:
            insufficient_data += 1
        else:
            incorrect_results += 1
        total += 1

    rate_incorrect = float(incorrect_results)/float(total)
    return total, correct_results, incorrect_results, insufficient_data



def get_words(doc):
    splitter = re.compile('\\W*')
    words = [s.lower() for s in splitter.split(doc)
                        if len(s) >= 2 and len(s) < 19]
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
        """used to create bias between categories
            i.e. (to be labeled bad must be 3*better than good)"""
        self.thresholds[cat] = t

    def getthreshold(self,cat):
        if cat not in self.thresholds: return 1.0
        return self.thresholds[cat]

    def docprob(self,item,cat): #equivalent P(Document|Category)
        features = self.getfeatures(item)
        #print features
        p=1
        for f in features:
            p*=self.weightedprob(f,cat,self.fprob)
           # print p
        return p

    def prob(self,item,cat):
        p_cat = self.cat_count(cat)/self.totalcount()
        doc_prob = self.docprob(item,cat)
        #print cat,doc_prob, p_cat
        return p_cat*doc_prob


    def classify(self,item,default= 'Insufficient Data'):
        #if item.split('-')[0] == 'NETSURE': return ('WARRANTY','N/A')
        probs = {}
        # Find the category with the highest probability
        max = 0.0
        for cat in self.categories():
            probs[cat] = self.prob(item,cat)
            if probs[cat] > max:
                max = probs[cat]
                best = cat
            #print cat, probs[cat]

        #ensure probability exceeds threshold*next_best
        #improve_over_next = 9999999999999999999
        for cat in self.categories():
            if cat == best: continue
            if probs[cat]*self.getthreshold(best) > probs[best]: return (default,'N/A')
        return best, max

class fisherclassifier(classifier):
    def __init__(self,getfeatures):
        classifier.__init__(self,getfeatures)
        self.minimums = {}

    def setminimums(self,cat,min):
        self.minimums[cat] = min

    def getminimums(self,cat,min):
        if cat not in self.minimums: return 0
        else: return self.minimums[cat]


    def cprob(self,feat,cat):
        #the frequency of this feature in this category
        clf = self.fprob(feat,cat)
        if clf == 0: return 0

        #the frequency of this feature in all categories
        freqsum = sum([self.prob(f,c) for c in self.categories()])

        #The probability is the frequency in this category divided by
        #the overall frequency
        p = clf/(freqsum)
        return p
    def fisherprob(self,item,cat):
        #Mult. all probabilities together
        p = 1
        features = self.getfeatures(item)

        for f in features:
            p *= (self.weightedprob(f,cat,self.cprob))

        fscore = -2*(math.log(p))

    def invchi2(self,chi,df):
        m = chi / 2.0
        sum = term = math.exp(-m)
        for i in range(1,df//2):
            term *= m/i
            sum += term
        return min(sum,1.0)


test_sample = [('CVPN3030-RED','SECURITY'),
('AS54-CT3-648-DC','ACCESS SERVER'),
('S2951UK9-15001M','ROUTER'),
('WS-X45-SUP7-E','SWITCH'),
('CRS-MSC-B','ROUTER'),
('CAB-N5K6A-NA','CABLE'),
('SL-39-DATA-K9','ROUTER'),
('CE-510-K9','CONTENT'),
('ASR1002-5G-SHA/K9','ROUTER'),
('CP-7945G-CH1','VOIP'),
('WS-C3560E-48PD-SF','SWITCH'),
('UCS-CPU-E5-2650','SERVER'),
('ASR1006-20G-HA/K9','ROUTER'),
('NM-1ATM-25','ROUTER'),
('RE-333-768-WW-S','ROUTER'),
('SX-FI424C','SWITCH'),
('7300-I/O-CFM-128M','ROUTER'),
('SFP-1OC48-SR-GEN','OPTIC'),
('PA-MC-STM-1MM','ROUTER'),
('CISCO2811-CCME/K9','ROUTER'),
('SUN-SERVER-V240','SERVER'),
('MEM2800-128CF','ROUTER'),
('WS-C2960G-48TC-L','SWITCH'),
('MEM-FD4G','ROUTER'),
('A9K-2X100GE-TR','ROUTER'),
('MEM-C4K-FLD64M','SWITCH'),
('C2821-VSEC-SRST/K9','ROUTER'),
('CISCO886-SEC-K9','ROUTER'),
('5759','SERVER'),
('MAS-GSR-BLANK','ROUTER'),
('PWR-MX480-1200-AC-S','ROUTER'),
('OMNI-PS9-500P','OTHER'),
('ACS-2821-51-FAN','ROUTER'),
('CAB-SMF-LC/UPC-LC/UPC-1M-YE','CABLE'),
('CP-7912G-A','VOIP'),
('ASA-SSM-CSC-10-K9','SECURITY'),
('SV-S2501GE24PDC-4','SWITCH'),
('CISCO4500-M','ROUTER'),
('503746-B21','SERVER'),
('ME-C3750-24TE-B','SWITCH'),
('CISCO804-IDSL','ROUTER'),
('NM-4T','ROUTER'),
('CWDM-XFP-1490-80KM','OPTIC'),
('SRW248G4P-K9-NA','SWITCH')]