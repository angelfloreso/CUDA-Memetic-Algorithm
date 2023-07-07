import graphviz as gv

g2 = gv.Digraph(format='svg')
g2.node('A')
g2.node('B')
g2.edge('A', 'B')
g2.render('img/g2')


import matplotlib.pyplot as plt
import networkx as nx


secuencia = '''1,14=-0.1	5,1=1.0	6,1=1.0	7,2=0.5	7,3=0.4	8,4=0.2	8,17=-0.2	9,5=1.0	9,6=-0.1	10,7=0.3	11,4=0.4	11,7=-0.2	11,22=0.4	12,23=0.1	13,8=0.6	14,9=1.0	15,10=0.2	16,11=0.5	16,12=-0.2	17,13=0.5	19,14=0.1	20,15=0.7	20,26=0.3	21,16=0.6	22,16=0.5	23,17=0.2	24,15=-0.2	24,18=-0.1	24,19=0.3	25,20=0.4	26,21=-0.2	26,28=0.1	27,24=0.6	27,25=0.3	27,30=-0.2	28,25=0.5	29,26=0.4	30,27=0.6'''

G = nx.DiGraph()
nodos = {}
labels = {}

for item in secuencia.split("	"):
	valores = item.split(",")
	valoresA = valores[1].split("=")
	print valores[0], valoresA[0], valoresA[1]

	if not int(valores[0]) in nodos:
		G.add_node(int(valores[0]))
		nodos[int(valores[0])] = 0
	if not int(valoresA[0]) in nodos:
		G.add_node(int(valoresA[0]))
		nodos[int(valoresA[0])] = 0
	labels[(int(valores[0]), int(valoresA[0]))] = float(valoresA[1])
	G.add_edge(int(valores[0]), int(valoresA[0]), weight = float(valoresA[1]))

print G.nodes()
#G = nx.complete_graph(100)

graph_pos = nx.fruchterman_reingold_layout(G)
#nx.draw(G, pos)

nx.draw_networkx_nodes(G,graph_pos, node_size = 1600, alpha = 0.3, node_color = 'blue')

nx.draw_networkx_edges(G,graph_pos, width = 1, alpha = 0.3, edge_color = 'blue')

nx.draw_networkx_labels(G, graph_pos, font_size = 12, font_family = 'sans-serif')

nx.draw_networkx_edge_labels(G, graph_pos, edge_labels = labels,  label_pos = 0.3)


plt.show()