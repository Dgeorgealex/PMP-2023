from pgmpy.models import BayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination
import matplotlib.pyplot as plt
import networkx as nx

model = BayesianNetwork([('C', 'I'), ('C', 'A'), ('I', 'A')])

cpd_c = TabularCPD(variable='C', variable_card=2, values=[[0.9995], [0.0005]])
cpd_i = TabularCPD(variable='I', variable_card=2, values=[[0.99, 0.97], [0.01, 0.03]], evidence=['C'], evidence_card=[2])
cpd_a = TabularCPD(variable='A', variable_card=2, values=[[0.9999, 0.05, 0.98, 0.02],
                                                          [0.0001, 0.95, 0.02, 0.98]],
                   evidence=['C', 'I'],
                   evidence_card=[2, 2])
print(cpd_c)
print(cpd_i)
print(cpd_a)

model.add_cpds(cpd_c, cpd_i, cpd_a)

assert model.check_model()

infer = VariableElimination(model)
p_earthquake_alarm = infer.query(variables=['C'], evidence={'A':1})
print(p_earthquake_alarm)

p_fire_alarm = infer.query(variables=['I'], evidence={'A':0})
print(p_fire_alarm)

pos = nx.circular_layout(model)
nx.draw(model, pos=pos, with_labels=True, node_size=4000, font_weight='bold', node_color='skyblue')
plt.show()