import pandas as pd

def print_stats (df):
	# Simple function to print survivor stats for any dataframe with Sex and Survived
	number_passengers = len(df.index)
	number_survived = len(df[(df.Survived == 1)].index)
	proportion_survivors = float(number_survived) / float(number_passengers )

	women_onboard = len(df[(df.Sex == 'female')].index)
	men_onboard = len(df[(df.Sex == 'male')].index)

	women_survived =  len(df[( (df.Sex == 'female') & (df.Survived ==1) )].index)
	male_survived = len(df[( (df.Sex == 'male') & (df.Survived ==1) )].index)
 
	proportion_women_survived = float( len(df[( (df.Sex == 'female') & (df.Survived ==1) )].index) )  / float(women_onboard )
	proportion_men_survived = float( len(df[( (df.Sex == 'male') & (df.Survived ==1) )].index) )  / float(men_onboard )

	df_stat_for_review = pd.DataFrame([[women_onboard,women_survived, (proportion_women_survived*100)],
								   [men_onboard,male_survived, (proportion_men_survived*100)],
								   [number_passengers,number_survived, (proportion_survivors*100)]
								],
                  columns=["Onboard", "Survived" , "% Survived"],
                  index=["female","male", "total"])
                  
	print df_stat_for_review                  

print "\tStats on the Training Dataset"
print_stats(pd.read_csv('../Data/train.csv', header=0))
df_test = pd.read_csv('../Data/test.csv', header=0)

## Gender Model
model_name = "gender"              
df_test['Survived'] = 0
df_test.loc[df_test['Sex']=='female', 'Survived']=1
print "\n\tStats on the Test Dataset"
print_stats(df_test)

# Export to csv
df_test[['PassengerId', 'Survived']].to_csv("output_{}.csv".format(model_name), index= None)