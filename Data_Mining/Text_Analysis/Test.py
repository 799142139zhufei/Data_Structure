from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer


vec = CountVectorizer(analyzer='word',max_features=4000,lowercase = False)
vectorizer = TfidfVectorizer(analyzer='word',max_features=4000,lowercase = False)
list = ["there is a dog dog", "here is a cat"]
count_vec = vec.fit_transform(list)
tv = vectorizer.fit_transform(list)
print(count_vec)
print(tv)