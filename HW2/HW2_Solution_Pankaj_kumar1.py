import matplotlib.pyplot as plt
from collections import Counter, defaultdict
import math
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification


class JCodeDataAnalyser:

    def __init__(self, file_path):
        self.file_path = file_path

    def read_jcode_data(self):
        with open(self.file_path, 'r') as f:
            claims = f.readlines()
        claims_data = [x.split(',') for x in claims]
        self.header = claims_data[0]
        jcode_claims = [x for x in claims_data[1:] if x[claims_data[0].index('Procedure.Code')].startswith('J')]
        return jcode_claims

    '''
    A medical claim is denoted by a claim number ('Claim.Number'). Each claim consists of one or more medical lines denoted by a claim line number ('Claim.Line.Number').

    1. J-codes are procedure codes that start with the letter 'J'.

         A. Find the number of claim lines that have J-codes.

         B. How much was paid for J-codes to providers for 'in network' claims?

         C. What are the top five J-codes based on the payment to providers?
   '''

    def question1_answers(self, jcode_claims):
        print("answers for question #1 ")
        print('total number of lines that have Jcodes {}'.format(len(jcode_claims)))
        j_code_provider_payment = sum(
            [float(x[self.header.index('Provider.Payment.Amount')])
             for x in jcode_claims if x[self.header.index('In.Out.Of.Network')].startswith('I')])
        print('total paid for  jcode to provides for "in network" claims {}   '.format(j_code_provider_payment))
        sorted_pay = sorted(jcode_claims, key=lambda x: x[self.header.index('Provider.Payment.Amount')], reverse=True)
        [print("Jcode with max #{} is {}".format(i+1, x[self.header.index('Procedure.Code')])) for i, x in enumerate(sorted_pay[0:5])]

    '''
     2. For the following exercises, determine the number of providers that were paid for at least one J-code.
         Use the J-code claims for these providers to complete the following exercises.
    
            A. Create a scatter plot that displays the number of unpaid claims (lines where the ‘Provider.Payment.Amount’ field is equal to zero) 
            for each provider versus the number of paid claims.
    
            B. What insights can you suggest from the graph?
    
            C. Based on the graph, is the behavior of any of the providers concerning? Explain.
    
    '''
    def question2_answers(self, jcode_claims):
        print("answers for question #2 ")
        claims_dict = dict()
        nj_claims = [x for x in jcode_claims if float(x[self.header.index('Provider.Payment.Amount')]) != 0]
        zero_payment_provider = Counter(
            [x[self.header.index('Provider.ID')] for x in nj_claims if x[self.header.index('Provider.Payment.Amount')].startswith('0')])
        non_zero_payment_provider = Counter(
            [x[self.header.index('Provider.ID')] for x in nj_claims if float(x[self.header.index('Provider.Payment.Amount')]) != 0])
        print('from graph it is clear that provider ID {} has unusually high zero claim amount'.format('FA100005001'))
        print('from graph it is clear that most of the providers have less than 20 zero payment claims except for provider id {}'.format('FA100005001'))

        self.show_plot( zero_payment_provider, non_zero_payment_provider)

    def show_plot(self,zero_payment_provider, non_zero_payment_provider ):
        dd = defaultdict(list)
        fig, ax = plt.subplots()
        for d in (zero_payment_provider, non_zero_payment_provider):
            for k, v in d.items():
                dd[k].append(v)
        z = [x[0] for x in dd.values() if len(x) > 1]
        y = [x[1] for x in dd.values() if len(x) > 1]
        ax.scatter(z, y)
        n = [x for x in dd.keys() if len(dd[x]) > 1]
        for i, txt in enumerate(n):
            ax.annotate(txt, (z[i], y[i]))
        plt.xlabel('Zero payment_count')
        plt.ylabel('Non Zero payment_count')
        plt.show()

    '''
    3. Consider all claim lines with a J-code.

         A. What percentage of J-code claim lines were unpaid?

         B. Create a model to predict when a J-code is unpaid. Explain why you choose the modeling approach.

         C. How accurate is your model at predicting unpaid claims?

          D. What data attributes are predominately influencing the rate of non-payment?
    '''

    def question3_answers(self, jcode_claims):
        print('answers for question # 3')
        y_data = [0 if math.isclose(float(x[self.header.index('Provider.Payment.Amount')]), 0) else 1 for x in jcode_claims]

        '''
         classify y_data which is whether provider was paid any amount or not. We 'll consider imparf of one continuous variable Claim charge amount  
        '''
        x_data = [[float(x[self.header.index('Claim.Charge.Amount')])] for x in jcode_claims]

        '''
             expand x_data by including some categorical variables 
        '''
        enc = OneHotEncoder(drop='first')
        enc_feature_index= []
        enc_feature_idx= []
        categorical_features = np.array(['In.Out.Of.Network', 'Claim.Type', 'Revenue.Code', 'Diagnosis.Code','Claim.Subscriber.Type', 'Denial.Reason.Code','Provider.ID','Place.Of.Service.Code'])
        enc.fit([[x[self.header.index('In.Out.Of.Network')], x[self.header.index('Claim.Type')], x[self.header.index('Revenue.Code')],
                  x[self.header.index('Diagnosis.Code')], x[self.header.index('Claim.Subscriber.Type')], x[self.header.index('Denial.Reason.Code')],
                  x[self.header.index('Provider.ID')] ,x[self.header.index('Place.Of.Service.Code')]] for x in jcode_claims])
        categorical_features_val = enc.transform([[x[self.header.index('In.Out.Of.Network')], x[self.header.index('Claim.Type')], x[self.header.index('Revenue.Code')],
                             x[self.header.index('Diagnosis.Code')], x[self.header.index('Claim.Subscriber.Type')], x[self.header.index('Denial.Reason.Code')],
                             x[self.header.index('Provider.ID')],x[self.header.index('Place.Of.Service.Code')]] for x in jcode_claims])
        x_data = np.concatenate((np.array(categorical_features_val.toarray()), np.array(x_data)), axis=1)
        [enc_feature_index.extend([categorical_features[i]]*len(enc.categories_[i]) for i in range(len(enc.categories_)))]
        [enc_feature_idx.extend(i) for i in enc_feature_index]
        enc_feature_idx.append('Claim.Charge.Amount')
        x_train = x_data[0:30000]
        x_test = x_data[30000:]
        y_train =  y_data[0:30000]
        y_test = y_data[30000:]

        # model logistic regression

        lr = LogisticRegression()
        lr.fit(x_train, y_train)
        score2 = lr.score(x_test, y_test)
        cm = confusion_matrix(y_test, lr.predict(x_test))
        print("accuracy score for logistic regression :{}".format(score2))

        # model Random forest

        clf = RandomForestClassifier(max_depth=50)
        clf.fit(x_train,y_train)
        score3 = clf.score(x_test,y_test )
        cm2= confusion_matrix(y_test, clf.predict(x_test))
        clf.feature_importances_
        feature_importance_dict = dict((i, x) for i, x in enumerate(clf.feature_importances_))
        sorted_feature_importance_dict= {k: v for k, v in sorted(feature_importance_dict.items(), key=lambda item: item[1], reverse=True)}
        print(" as we can see random forest gives as maximum accuracy with accuracy score of {}".format(score3))
        most_imp_features = list(set(sorted_feature_importance_dict.keys()))[0:10]
        print('{} most important features are '.format(len(most_imp_features)))
        [print("  {}".format(enc_feature_idx[i])) for i in most_imp_features]

if __name__ =='__main__':
    jcode_analyser = JCodeDataAnalyser('claim.sample.csv')
    jcode_claims = jcode_analyser.read_jcode_data()
    jcode_analyser.question1_answers(jcode_claims)
    jcode_analyser.question3_answers(jcode_claims)
    jcode_analyser.question2_answers(jcode_claims)