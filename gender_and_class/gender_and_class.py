import pandas as pd

def print_stats_based_on_Sex_and_Pclass (df):
	# Simple function to print survivor stats for any dataframe with Sex, Pclass, and Survived
	
	df_stat_for_review = pd.DataFrame([[ len(df[(df.Pclass == 1) & (df.Sex=='female')].index) ,len(df[(df.Pclass == 1) & (df.Sex=='female') & (df.Survived==1)].index), float( len(df[(df.Pclass == 1) & (df.Sex=='female')  & (df.Survived==1)].index)) / float (len(df[(df.Pclass == 1) & (df.Sex=='female') ].index))],
							[ len(df[(df.Pclass == 2) & (df.Sex=='female')].index) ,len(df[(df.Pclass == 2) & (df.Sex=='female') & (df.Survived==1)].index), float( len(df[(df.Pclass == 2) & (df.Sex=='female')  & (df.Survived==1)].index)) / float (len(df[(df.Pclass == 2) & (df.Sex=='female') ].index))],
							[ len(df[(df.Pclass == 3) & (df.Sex=='female')].index) ,len(df[(df.Pclass == 3) & (df.Sex=='female') & (df.Survived==1)].index), float( len(df[(df.Pclass == 3) & (df.Sex=='female')  & (df.Survived==1)].index)) / float (len(df[(df.Pclass == 3) & (df.Sex=='female') ].index))],
							[ len(df[(df.Pclass == 1) & (df.Sex=='male')].index) ,len(df[(df.Pclass == 1) & (df.Sex=='male') & (df.Survived==1)].index), float( len(df[(df.Pclass == 1) & (df.Sex=='male')  & (df.Survived==1)].index)) / float (len(df[(df.Pclass == 1) & (df.Sex=='male') ].index))],
							[ len(df[(df.Pclass == 2) & (df.Sex=='male')].index) ,len(df[(df.Pclass == 2) & (df.Sex=='male') & (df.Survived==1)].index), float( len(df[(df.Pclass == 2) & (df.Sex=='male')  & (df.Survived==1)].index)) / float (len(df[(df.Pclass == 2) & (df.Sex=='male') ].index))],
							[ len(df[(df.Pclass == 3) & (df.Sex=='male')].index) ,len(df[(df.Pclass == 3) & (df.Sex=='male') & (df.Survived==1)].index), float( len(df[(df.Pclass == 3) & (df.Sex=='male')  & (df.Survived==1)].index)) / float (len(df[(df.Pclass == 3) & (df.Sex=='male') ].index))],
							[ len(df.index) , len(df[(df.Survived == 1)].index)	, float(len(df[(df.Survived == 1)].index)) / float(len(df.index) ) ]   ],
                  columns=["Onboard", "Survived" , "% Survived"],
                  index=["Female/Class 1" ,"Female/Class 2", "Female/Class 3", "Male/Class 1" ,"Male/Class 2", "Male/Class 3", "Total"])
	print df_stat_for_review   

print "\tStats on the Training Dataset"
print_stats_based_on_Sex_and_Pclass(pd.read_csv('../Data/train.csv', header=0))
df_test = pd.read_csv('../Data/test.csv', header=0)

## Sex and Pclass Model
model_name = "Sex_and_class"              
df_test['Survived'] = 0
df_test.loc[df_test['Sex']=='female', 'Survived']=1  # All females still get Survived = 1
# Male/Class 1 only have survival rate of 35% but still applying all Male/Class 1 to make slight 
# 	improvement in the model. 

df_test.loc[ (df_test['Sex']=='male') & (df_test['Pclass']==1), 'Survived']=1
print "\n\tStats on the Test Dataset"
print_stats_based_on_Sex_and_Pclass(df_test)

# Export to csv
df_test[['PassengerId', 'Survived']].to_csv("output_{}.csv".format(model_name), index= None)