import Item_Classifier as ic
import time

FOLDER = 'C:\\!Coding\\AI Examples\\Item_ID_Classifier\\'

cl = ic.naivebayes(ic.get_words)
ic.sampletrain_from_csv(cl,FOLDER+'test2.csv',0,1)

for cat in cl.categories():
    cl.setthreshold(cat,2)

sample = ic.test_sample

print ic.test_bayes_model(cl,sample)
print ic.test_bayes_model2(cl,sample)

train_data = ic.test_data(FOLDER+'test2.csv',0,1)



def tune_thresholds(cl,sample):
    """Example of tuning based on a sample population"""
    categories = cl.categories()
    for cat in categories:
        cl.setthreshold(cat,1)
    eps = 0.025
    prior_case = base_case = ic.test_bayes_model(cl,sample)
    print 'Running Tune...'
    if base_case < .05: return 'current thresholds satisfactory'
    i = 0
    while base_case > .025 or i == 200:
        for cat in categories:
            t = cl.getthreshold(cat)
            x = t + eps #should you make it so that it increases/decreases epsilon
            cl.setthreshold(cat,x)
            test_case = ic.test_bayes_model(cl,sample)

            if test_case >= base_case:
                cl.setthreshold(cat,t)
            else:
                prior_case = base_case
                base_case = test_case
        if (prior_case - base_case)> .00000001:
            eps = .025
            prior_case = base_case
        else:
            eps = eps + .005
        i = i + 1
        if i%5 == 0 or i == 1: print i,eps, base_case

print 'Results pre-tuning default 2 multiple: ' + str(ic.test_bayes_model2(cl,sample))
start_time= time.time()
print tune_thresholds(cl,sample)
end_time = time.time()
print end_time - start_time
print 'Results after-tuning default 2 multiple: ' + str(ic.test_bayes_model2(cl,sample))

"""
#sample = sample[:1000]



sample = train_data

#print ic.test_bayes_model2(cl,sample)

#tune_thresholds(cl,sample)
def stoch_tune(cl,train_data):
    alpha = .5
    categories = cl.categories()
    for cat in categories:
        cl.setthreshold(cat,1)
    for train_case in train_data:
        train_item, train_cat = train_case
        correct_prob = cl.prob(train_item,train_cat)
        #neeed to come up with language that tests the guess and compares to the error#
        guess_cat, guess_prob = '',0
        for cat in categories:
            t = cl.prob(train_item,cat)
            if t > guess_prob:
                guess_cat,guess_prob = cat, t

        guess_thresh = cl.getthreshold(guess_cat)
        #print ((guess_prob/guess_thresh) - correct_prob)/guess_prob
        #when an error occurs it's because the guess_thresh
        guess_thresh = guess_thresh + alpha*((guess_prob/guess_thresh) - correct_prob)/guess_prob
        cl.setthreshold(guess_cat,guess_thresh)
        #print guess_cat, train_cat

stoch_tune(cl,sample)
print cl.thresholds
print ic.test_bayes_model2(cl,sample)
"""




