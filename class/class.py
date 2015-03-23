import pandas as pd

def print_stats_for_survivors_based_on_Pclass (df):
	# Simple function to print survivor stats for any dataframe with Pclass and Survived
	number_passengers = len(df.index)
	number_survived = len(df[(df.Survived == 1)].index)
	proportion_survivors = float(number_survived) / float(number_passengers )
	
	df_stat_for_review = pd.DataFrame([[ len(df[df.Pclass == 1].index) ,len(df[(df.Pclass == 1) & (df.Survived==1)].index), float( len(df[(df.Pclass == 1) & (df.Survived==1)].index)) / float (len(df[df.Pclass == 1].index))],
									   [ len(df[df.Pclass == 2].index) ,len(df[(df.Pclass == 2) & (df.Survived==1)].index), float( len(df[(df.Pclass == 2) & (df.Survived==1)].index)) / float (len(df[df.Pclass == 2].index))],
									   [ len(df[df.Pclass == 3].index) ,len(df[(df.Pclass == 3) & (df.Survived==1)].index), float( len(df[(df.Pclass == 3) & (df.Survived==1)].index)) / float (len(df[df.Pclass == 3].index))],
									   [ number_passengers, number_survived, proportion_survivors]
									],
                  columns=["Onboard", "Survived" , "% Survived"],
                  index=["Class 1" ,"Class 2", "Class 3", "Total"])


	print df_stat_for_review     	

print "\tStats on the Training Dataset"
print_stats_for_survivors_based_on_Pclass(pd.read_csv('../Data/train.csv', header=0))
df_test = pd.read_csv('../Data/test.csv', header=0)

## Pclass Model
model_name = "class"              
df_test['Survived'] = 0
df_test.loc[df_test['Pclass']==1, 'Survived']=1
print "\n\tStats on the Test Dataset"
print_stats_for_survivors_based_on_Pclass(df_test)

# Export to csv
df_test[['PassengerId', 'Survived']].to_csv("output_{}.csv".format(model_name), index= None)
