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

import docclass


"""
cl = docclass.classifier(docclass.get_words)

docclass.sampletrain(cl)

print cl.cat_count('good')
print cl.feat_cat_count('money','bad')
print cl.feat_cat_count('money','good')
print cl.fprob('money','good')
print cl.weightedprob('money','good',cl.fprob)

docclass.sampletrain(cl)

print cl.weightedprob('money','good',cl.fprob)
"""

cl = docclass.naivebayes(docclass.get_words)

docclass.sampletrain(cl)

print cl.cat_count('good')
print cl.feat_cat_count('money','bad')
print cl.feat_cat_count('money','good')
print cl.fprob('money','good')
print 'test'
print cl.weightedprob('money','good',cl.fprob)
print cl.weightedprob('money','good',cl.fprob)
print cl.prob('quick rabbit','good')
print cl.prob('QUICK RABBIT','good')

print cl.classify('quick rabbit',default = 'unkown')

print cl.classify('test',default = 'unkown')#check to see if feature are in data set

print cl.classify('quick money',default = 'unkown')

cl.setthreshold('bad',3.0)

print cl.classify('quick money',default = 'unkown')

#print cl.prob('ccccc','bad')