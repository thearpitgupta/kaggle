import pandas as pd
import csv as csv
import numpy as np
import math as math
from sklearn.ensemble import RandomForestClassifier

df = pd.read_csv('../Data/train.csv', header=0)

# In order to analyse the price column I need to bin up that data
# here are my binning parameters, the problem we face is some of the fares are very large
# So we can either have a lot of bins with nothing in them or we can just lose some
# information by just considering that anythng over 39 is simply in the last bin.
# So we add a ceiling
fare_ceiling = 40
# then modify the data in the Fare column to = 39, if it is greater or equal to the ceiling

df.loc[df['Fare'] >= fare_ceiling,'Fare'] = fare_ceiling - 1.0


fare_bracket_size = 10
number_of_price_brackets = fare_ceiling / fare_bracket_size


number_of_classes = len(pd.unique(df['Pclass']))   # But it's better practice to calculate this from the Pclass directly:
                                                  # just take the length of an array of UNIQUE values in column index 2


age_ceiling = math.floor(np.max(df['Age']) / 10) * 10
age_bucket_size = 10
#number_of_age_buckets = int(age_ceiling / age_bucket_size + 1)

number_of_age_buckets = 5

df['BinAge'] = df['Age']
df.loc[df['Age']<=3, 'BinAge']=1
df.loc[(df['Age']>3) & (df['Age']<=10), 'BinAge']=2
df.loc[(df['Age']>10) & (df['Age']<=65), 'BinAge']=3
df.loc[df['Age']>65, 'BinAge']=4

#df.loc[df['Age'] >= age_ceiling,'Age'] = age_ceiling - 1.0

df['BinAge'] = df['BinAge'].fillna(5)  
print df.head(10)
#This is a different technique for filling nas as below
#df['Age'].fillna(age_ceiling, inplace = True)


# This reference matrix will show the proportion of survivors as a sorted table of
# gender, class and ticket fare.
# First initialize it with all zeros

survival_table = np.zeros([2,number_of_classes,number_of_price_brackets,number_of_age_buckets],float)

# I can now find the stats of all the women and men on board
for i in xrange(number_of_classes):
    for j in xrange(number_of_price_brackets):
        for k in xrange(number_of_age_buckets):
            women_only_stats = df.loc[ (df['Sex'] == "female") \
                                & (df['Pclass'] == i+1) \
                                & (df['Fare'] >= j*fare_bracket_size) \
                                & (df['Fare'] < (j+1)*fare_bracket_size) \
                                & (df['BinAge'] == k+1), 'Survived']

            men_only_stats = df.loc[ (df['Sex'] != "female") \
                                & (df['Pclass'] == i+1) \
                                & (df['Fare'] >= j*fare_bracket_size) \
                                & (df['Fare'] < (j+1)*fare_bracket_size) \
                                & (df['BinAge'] == k+1), 'Survived']

            survival_table[0,i,j,k] = np.mean(women_only_stats)  # Female stats
            survival_table[1,i,j,k] = np.mean(men_only_stats)    # Male stats


# Since in python if it tries to find the mean of an array with nothing in it
# (such that the denominator is 0), then it returns nan, we can convert these to 0
# by just saying where does the array not equal the array, and set these to 0.

survival_table[ survival_table != survival_table ] = 0.

# Now I have my proportion of survivors, simply round them such that if <0.5
# I predict they dont surivive, and if >= 0.5 they do
survival_table[ survival_table < 0.5 ] = 0
survival_table[ survival_table >= 0.5 ] = 1



# Now I have my indicator I can read in the test file and write out
# if a women then survived(1) if a man then did not survived (0)
# First read in test
test_file = open('../Data/test.csv', 'rb')
test_file_object = csv.reader(test_file)
header = test_file_object.next()

# Also open the a new file so I can write to it. 
predictions_file = open("output_gender_class_age.csv", "wb")
predictions_file_object = csv.writer(predictions_file)
predictions_file_object.writerow(["PassengerId", "Survived"])

# First thing to do is bin up the price file
for row in test_file_object:
    for j in xrange(number_of_price_brackets):
        # If there is no fare then place the price of the ticket according to class
        try:
            row[8] = float(row[8])	  # No fare recorded will come up as a string so
                                      # try to make it a float
        except:                       # If fails then just bin the fare according to the class
            bin_fare = 3 - float(row[1])
            break                     # Break from the loop and move to the next row
        if row[8] > fare_ceiling:     # Otherwise now test to see if it is higher
                                      # than the fare ceiling we set earlier
            bin_fare = number_of_price_brackets - 1
            break                     # And then break to the next row

        if row[8] >= j*fare_bracket_size\
            and row[8] < (j+1)*fare_bracket_size:     # If passed these tests then loop through
                                                      # each bin until you find the right one
                                                      # append it to the bin_fare
                                                      # and move to the next loop
            bin_fare = j
            break
    for k in xrange(number_of_age_buckets):
    	try:
            row[4] = float(row[4])	  # No fare recorded will come up as a string so
                                      # try to make it a float
        except:                       # If fails then just bin the fare according to the class
            age_bin = number_of_age_buckets
            break
        if row[4] > 65:     
            age_bin = 4
            break                     # And then break to the next row
        if row[4] >= 0\
            and row[4] <= 3:    
            age_bin = 1
            break
        if row[4] > 3\
        	and row[4] <=10:
        	age_bin = 2
        	break
        if row[4] > 10\
        	and row[4] <=65:
        	age_bin = 3
        	break
        # Now I have the binned fare, passenger class, and whether female or male, we can
        # just cross ref their details with our survival table
    if row[3] == 'female':
        predictions_file_object.writerow([row[0], "%d" % int(survival_table[ 0, float(row[1]) - 1, bin_fare, age_bin-1 ])])
    else:
        predictions_file_object.writerow([row[0], "%d" % int(survival_table[ 1, float(row[1]) - 1, bin_fare, age_bin-1 ])])

# Close out the files
test_file.close()
predictions_file.close()