    import numpy as np
    import pandas as pd
    import re
    from sklearn.model_selection import train_test_split
    from sklearn import svm
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score
    from nltk.stem import PorterStemmer, WordNetLemmatizer
    from nltk.corpus import stopwords 
    from sklearn.feature_extraction.text import TfidfTransformer
    from sklearn.feature_extraction.text import CountVectorizer
    
    
    dict_lookup = {'I':0, 'E':1, 'N':0, 'S':1, 
                   'F':0, 'T':1, 'J':0, 'P':1}
    dict_lookup_reverse = [{0:'I', 1:'E'}, {0:'N', 1:'S'},
                           {0:'F', 1:'T'}, {0:'J', 1:'P'}]
    #To remove these words if any present in the i/p.
    mbti_words = ['INFJ', 'ENTP', 'INTP', 'INTJ',
                  'ENTJ', 'ENFJ', 'INFP', 'ENFP',
                  'ISFP', 'ISTP', 'ISFJ', 'ISTJ',
                  'ESTP', 'ESFP', 'ESTJ', 'ESFJ']
    mbti_words = [x.lower() for x in mbti_words]
    
    stemmer = PorterStemmer()
    lemmatiser = WordNetLemmatizer()
    
    #To remove the stop words like the,a,an,in
    stop_words = stopwords.words("english")
    
    def to_vector(personality):
        # transform mbti to binary vector
        return [dict_lookup[l] for l in personality]
    
    def to_mbti(personality):
        s = ""
        for i, l in enumerate(personality):
            s += dict_lookup_reverse[i][l]
        return s
    
    def data_refine(data, remove_stop_words=True, remove_mbti_profiles=True):
    
        personality = []
        processed_posts = []
        len_data = len(data)
        i=0
        
        for row in data.iterrows():
            i+=1
            if (i % 500 == 0 or i == 1 or i == len_data):
                print("%s of %s rows" % (i, len_data))
            #To remove all the unwanted strings(eg.https,site names etc)
            posts = row[1].posts
            temp = re.sub('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', ' ', posts)
            temp = re.sub("[^a-zA-Z]", " ", temp)
            temp = re.sub(' +', ' ', temp).lower()
            if remove_stop_words:
                temp = " ".join([lemmatiser.lemmatize(w) 
                         for w in temp.split(' ') if w not in stop_words])
            else:
                temp = " ".join([lemmatiser.lemmatize(w) 
                        for w in temp.split(' ')])
                
            #To remove the mbti words in the i/p.
            if remove_mbti_profiles:
                for t in mbti_words:
                    temp = temp.replace(t,"")
    
            type_labelized = to_vector(row[1].type)
            personality.append(type_labelized)
            processed_posts.append(temp)
    
        processed_posts = np.array(processed_posts)
        personality = np.array(personality)
        return processed_posts, personality
    
    data=pd.read_csv("mbti.csv")
    data['I']=data['type'].apply(lambda x:0 if x[0]=='I' else 1)
    data['N']=data['type'].apply(lambda x:0 if x[1]=='N' else 1)
    data['F']=data['type'].apply(lambda x:0 if x[2]=='F' else 1)
    data['J']=data['type'].apply(lambda x:0 if x[3]=='J' else 1)
    
    
    #[p.split('|||') for p in data.head(2).posts.values]
    #Processing the data to correct format to work with
    processed_posts, personality  = data_refine(data)
    
    
    # Section to assign a unique float number to the words
    # according to the tfidf vectorizer method
    
    # To get the words occuring 10%-70% of the posts
    vector = CountVectorizer(analyzer="word",  
                                 max_features=1500, 
                                 tokenizer=None,    
                                 preprocessor=None, 
                                 stop_words=None,  
                                 max_df=0.7,
                                 min_df=0.1)
    print("Finished the CountVectorizer")
    X_cnt = vector.fit_transform(processed_posts)
    
    tfid_vectorizer = TfidfTransformer()
    
    # Learn the idf vector (fit) and transform a 
    # count matrix to a tf-idf representation
    X_tfidf =  tfid_vectorizer.fit_transform(X_cnt).toarray()
    
    type_indicators = ["Introversion (I) / Extroversion (E)",
                       "Intuition (N)/Sensing (S)", 
                       "Feeling (F)/Thinking (T)", 
                       "Judging (J)/Perceiving (P)" ]
    X = X_tfidf
    print("Xgboost:")
    for l in range(len(type_indicators)):
        print("%s ..." % (type_indicators[l]))
        
        Y = personality[:,l]
    
        seed = 7
        test_size = 0.33
        X_train, X_test, y_train, y_test = train_test_split(X, Y,
                          test_size=test_size, random_state=seed)
    
        model = XGBClassifier()
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        predictions = [round(value) for value in y_pred]
        # evaluate predictions
        accuracy = accuracy_score(y_test, predictions)
        print("* %s Accuracy: %.2f%%" % (type_indicators[l], 
                                         accuracy * 100.0))
    ar=['I','N','F','J']
    Y_preddlog=[[]]
    Y_preddknn=[[]]
    Y_preddsvm=[[]]
    print("SVM:")
    for i in range(4):
        X = X_tfidf
        y = data[ar[i]].values
        
        XX_train,XX_test,yy_train,yy_test=train_test_split(X,y,
                                           test_size = 0.33, random_state=7)
        
        sv = svm.SVC(gamma="auto")  
        print("%s ..." % (type_indicators[i]))
        sv.fit(XX_train, yy_train)
        Y_preddsvm.insert(i,sv.predict(XX_test).tolist())
        acc_logg = round(sv.score(XX_train, yy_train) * 100, 2)
        print("Accuracy:",end='')
        print(round(acc_logg,2,), "%")   
    #data1=pd.read_csv("mbti.csv")
    #h=vector.fit_transform(data1['posts']);
    #yr=data[ar[2]].values
    #XXr_train,XXr_test,yyr_train,yyr_test=train_test_split(h,yr,test_size = 0.33, random_state=7)
    #lr=LogisticRegression(C=0.01,solver='liblinear').fit(XXr_train,yyr_train)
    #u=lr.predict(XXr_test)
    #acc_logg = round(lr.score(XXr_train, yyr_train) * 100, 2)
    #print(round(acc_logg,2,), "%")
    print("------------------------------------------------\nLog:")
    for i in range(4):
        X = X_tfidf
        y = data[ar[i]].values
        
        XX_train,XX_test,yy_train,yy_test=train_test_split(X,y,
                                    test_size = 0.2, random_state=7)
        
        logregg = LogisticRegression(C=0.09,solver='liblinear')
        print("%s ..." % (type_indicators[i]))
        logregg.fit(XX_train, yy_train)         
        Y_preddlog.insert(i,logregg.predict(XX_test).tolist())
        print("Accuracy:",end='')
        acc_logg = round(logregg.score(XX_train, yy_train) * 100, 2)
        print(round(acc_logg,2,), "%")
    #print("------------------------------------------------\nKNN:")
    #for i in range(4):
    #    X = data.drop(['type','posts',ar[i]], axis=1).values
    #    y = data[ar[i]].values
    #    
    #    XX_train,XX_test,yy_train,yy_test=train_test_split(X,y,test_size = 0.2, random_state=7)
    #    
    #    knnn = KNeighborsClassifier(n_neighbors = 10)
    #    knnn.fit(XX_train, yy_train)
    #    Y_preddknn.insert(i,knnn.predict(XX_test).tolist())
    #    acc_knnn = round(knnn.score(XX_train, yy_train) * 100, 2)
    #    print(round(acc_knnn,2,), "%")
    
    #knnn = KNeighborsClassifier(n_neighbors = 3)
    #knnn.fit(XX_train, yy_train)
    
    #Y_predd = knnn.predict(XX_test)
    ##print(yy_test)
    #print(Y_predd)
    #
    #acc_knnn = round(knnn.score(XX_train, yy_train) * 100, 2)
    #print(round(acc_knnn,2,), "%")
