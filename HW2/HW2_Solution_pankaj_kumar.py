'''
A medical claim is denoted by a claim number ('Claim.Number'). Each claim consists of one or more medical lines denoted by a claim line number ('Claim.Line.Number').

1. J-codes are procedure codes that start with the letter 'J'.

     A. Find the number of claim lines that have J-codes.

     B. How much was paid for J-codes to providers for 'in network' claims?

     C. What are the top five J-codes based on the payment to providers?



2. For the following exercises, determine the number of providers that were paid for at least one J-code.
 Use the J-code claims for these providers to complete the following exercises.

    A. Create a scatter plot that displays the number of unpaid claims (lines where the ‘Provider.Payment.Amount’ field is equal to zero) 
    for each provider versus the number of paid claims.

    B. What insights can you suggest from the graph?

    C. Based on the graph, is the behavior of any of the providers concerning? Explain.



3. Consider all claim lines with a J-code.

     A. What percentage of J-code claim lines were unpaid?

     B. Create a model to predict when a J-code is unpaid. Explain why you choose the modeling approach.

     C. How accurate is your model at predicting unpaid claims?

      D. What data attributes are predominately influencing the rate of non-payment?
'''

import matplotlib.pyplot as plt
from collections import Counter, defaultdict


with open('claim.sample.csv', 'r') as f:
     claims = f.readlines()

claims_list =[dict(zip( claims[0].split(','), claim.split(','))) for claim in claims[1:]]
j_claims = list(filter(lambda x: x['Procedure.Code'].startswith('J'),claims_list))
j_code_provider_payment = sum([float(x['Provider.Payment.Amount']) for x in j_claims if x['In.Out.Of.Network'].startswith('I')])
sorted_pay = sorted(j_claims , key= lambda x: x['Provider.Payment.Amount'] , reverse=True)
[print(x['Procedure.Code']) for x in sorted_pay[0:5]]

'''
2. For the following exercises, determine the number of providers that were paid for at least one J-code.
 Use the J-code claims for these providers to complete the following exercises.

    A. Create a scatter plot that displays the number of unpaid claims (lines where the ‘Provider.Payment.Amount’ field is equal to zero) 
    for each provider versus the number of paid claims.

    B. What insights can you suggest from the graph?

    C. Based on the graph, is the behavior of any of the providers concerning? Explain.
'''

claims_dict = dict()
nj_claims = [x for x in j_claims if float(x['Provider.Payment.Amount'])!=0]
zero_payment_provider = Counter([ x['Provider.ID'] for x in nj_claims if x['Provider.Payment.Amount'].startswith('0')])
non_zero_payment_provider = Counter([ x['Provider.ID'] for x in nj_claims if float(x['Provider.Payment.Amount'])!=0])

def show_plot():
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


#show_plot()

'''
3. Consider all claim lines with a J-code.

     A. What percentage of J-code claim lines were unpaid?

     B. Create a model to predict when a J-code is unpaid. Explain why you choose the modeling approach.

     C. How accurate is your model at predicting unpaid claims?

      D. What data attributes are predominately influencing the rate of non-payment?
'''
zero_payment_count = len([ x['Provider.ID'] for x in j_claims if x['Provider.Payment.Amount'].startswith('0')])
non_zero_payment_count = len([ x['Provider.ID'] for x in j_claims if float(x['Provider.Payment.Amount'])!=0])
print(zero_payment_count/non_zero_payment_count)


 