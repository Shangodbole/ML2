from graphviz import Digraph

dot = Digraph(comment='Corona Virus cases Singapore', engine='sfdp')
dot.attr(bgcolor='white')
dot.graph_attr['rankdir'] = 'TD'

for i in range(1, 41):
    dot.node("{}".format(i), 'Case{}'.format(i))
dot.edges([])
dot.node("{}".format(19), 'Case{} Local Singaporean No Travel History\n probably due to Chinese tour group from Guangxi'.format(19))
dot.node("30", "Case30 ATTENDED MEETING AT GRAND HYATT")
dot.node("8", "Case8 Chinese From Wuhan ATTENDED LIFE CHURCH AND MISSIONS PL")
dot.edge('8', '9', constraint='false', label= "Chinese From Wuhan ATTENDED LIFE CHURCH AND MISSIONS PL")
dot.edge('8', '38', constraint='false', label= "ATTENDED LIFE CHURCH AND MISSIONS PL")
dot.edge('8', '31', constraint='false', label= "ATTENDED LIFE CHURCH AND MISSIONS PL")
dot.edge('8', '33', constraint='false', label= "ATTENDED LIFE CHURCH AND MISSIONS PL")
dot.edge('19', '20', constraint='false', label= "Colleague of Case19", len ="2.00")
dot.edge('19', '21', constraint='false', label= "Domestic worker of Case19", len ="2.00")
dot.edge('19', '24', constraint='false',label= "Came to shop employing Case19")
dot.edge('24', '25', constraint='false',label= "husband of Case24")
dot.edge('26', '13', constraint='false', label= "Daughter of case 13")
dot.edge('19', '27',    label= "Husband of case 19", len ="2.00")
dot.edge('19', '28',   label= "Baby of case 19", len = "2.00")
dot.edge('30', '36', constraint='false', label="MEETING AT GRAND HYATT")
dot.edge('30', '39', constraint='false', label="MEETING AT GRAND HYATT")
dot.node("Source : https://www.straitstimes.com/singapore/health/novel-coronavirus-cases-in-singapore")
dot.view()
#dot.render('corona_sg.gv', view=True)