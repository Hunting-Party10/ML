#Present in Sklearn use it to vectorize and preprocess documents count/weighted count
#these options work almost always
TfidfVectorizer(min_df = 3, max_features= None,
                     strip_accents='unicode', analyzer='word', token_pattern=r'\w{1,}',
                     ngram_range=(1, 3), use_idf=1,smooth_idf=1,sublinear_tf=1,
                     stop_words = 'english')