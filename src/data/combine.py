import pandas as pd 
import re

df_news_train = pd.read_csv("./train/News_Train.tsv", sep="\t")
df_wikinews_train = pd.read_csv("./train/WikiNews_Train.tsv", sep="\t")
df_wikipedia_train = pd.read_csv("./train/Wikipedia_Train.tsv", sep="\t")

df_news_dev = pd.read_csv("./train/News_Dev.tsv", sep="\t")
df_wikinews_dev = pd.read_csv("./train/WikiNews_Dev.tsv", sep="\t")
df_wikipedia_dev = pd.read_csv("./train/Wikipedia_Dev.tsv", sep="\t")

df_news_test = pd.read_csv("./test/News_Test.tsv", sep="\t")
df_wikinews_test = pd.read_csv("./test/WikiNews_Test.tsv", sep="\t")
df_wikipedia_test = pd.read_csv("./test/Wikipedia_Test.tsv", sep="\t")

df_train = pd.concat([df_news_train, df_wikinews_train, df_wikipedia_train])
df_dev = pd.concat([df_news_dev, df_wikinews_dev, df_wikipedia_dev])
df_test = pd.concat([df_news_test, df_wikinews_test, df_wikipedia_test])

# df_train["Sentence"] = df_train.apply(lambda row: re.sub(r'\.(?=[A-Za-z])', '. ', row["Sentence"]), axis=1)
# df_dev["Sentence"] = df_dev.apply(lambda row: re.sub(r'\.(?=[A-Za-z])', '. ', row["Sentence"]), axis=1)
# df_test["Sentence"] = df_test.apply(lambda row: re.sub(r'\.(?=[A-Za-z])', '. ', row["Sentence"]), axis=1)

df_train.to_csv('train.csv', index=False)
df_dev.to_csv('dev.csv', index=False)
df_test.to_csv('test.csv', index=False)