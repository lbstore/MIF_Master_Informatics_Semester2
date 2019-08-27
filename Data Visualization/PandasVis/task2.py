import pandas as pd
import matplotlib.pyplot as plt

baseDir = "E:/University/MIF_Master_Informatics_Semester2/Data Visualization/"

dataSet = pd.read_csv(baseDir+"spamFileRemovedResampledComma.csv")
print(dataSet.info())
dataSet = dataSet.drop(columns=["maxRepeatedWordCount","maxWordLength","lineCount","averageWordLengh","alphaCount","alphaNumCount","alphaWordCount","numberCount"])
dataSet = dataSet.sample(frac=0.1)
print(dataSet.info())

# curves = pd.plotting.andrews_curves(dataSet,class_column="spam")
# plt.show()

# pd.plotting.parallel_coordinates(dataSet,class_column="spam")
# plot1 = dataSet['charSeq5'].plot(kind='hist',bins=50)
# plt.scatter(dataSet)
# plt.show()



