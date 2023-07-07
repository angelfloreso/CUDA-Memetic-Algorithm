from pylab import *
import numpy  
from scipy import interpolate
from scipy.interpolate import spline
from matplotlib.legend_handler import HandlerLine2D

G, H, alfa, beta, X, aptitud = [], [], [], [], [], 0

entrada = '/Users/lti/Dropbox/Maestria/Tesis/Paralelizacion/Instancias/Tominaga2SSGeneratedData.txt'

def leerEntrada(path):
	h, series, genes, tam, valores, seriesArr = 0, 0, 0, 0, [], []
	archivo = open(path, mode='r')
	for linea in archivo.readlines():
		datos = linea.split(' ')
		if series == 0: 
			genes, series = datos
			valores = [[] for i in range(int(genes))]
		elif tam == 0: tam = int(datos[0])
		elif h == 0: h = float(datos[1])
		elif len(datos) == int(genes)+1:
			for i in range(int(genes)):
				valores[i].append(float(datos[i]))
			if len(valores[0]) == tam:
				seriesArr.append(valores)
				valores = [[] for i in range(int(genes))]

	return float(h), int(series), int(genes), int(tam), seriesArr

h, series, ngen, ntiempos, valores = leerEntrada(entrada)

info = '''
G = [1.222005	1.886825	-0.777418	2.152205	]
H = [2.662681	0.105679	0.516583	3.281134	]
ALPHA = [15.363136	12.327762	]
BETA = [7.792911	14.420581	]
X = [0.700000	0.648556	0.621817	0.622036	0.662612	0.770077	0.948680	1.115478	1.191137	1.196875	1.177224	1.155717	1.140179	1.131499	1.128083	1.127815	1.128920	1.130259	1.131291	1.131886	1.132124	1.132145	1.132072	1.131983	1.131914	1.131874	1.131858	1.131856	1.131861	1.131867	1.131871	1.131874	1.131875	1.131875	1.131875	1.131875	1.131874	1.131874	1.131874	1.131874	1.131874	1.131874	1.131874	1.131874	1.131874	1.131874	1.131874	1.131874	1.131874	1.131874	0.300000	0.347468	0.416206	0.516932	0.657748	0.811352	0.888330	0.865447	0.815522	0.778362	0.757923	0.749348	0.747654	0.749170	0.751558	0.753603	0.754895	0.755489	0.755618	0.755519	0.755358	0.755220	0.755133	0.755093	0.755084	0.755090	0.755101	0.755110	0.755116	0.755118	0.755119	0.755119	0.755118	0.755117	0.755117	0.755117	0.755117	0.755117	0.755117	0.755117	0.755117	0.755117	0.755117	0.755117	0.755117	0.755117	0.755117	0.755117	0.755117	0.755117	]
Aptitud 7.359549
'''

info = info.replace('[', '').replace(']', '')

for item in info.split('\n'):
	value = item.split(' = ')
	if value[0] == 'G': G = [float(item2) for item2 in value[1].split('\t')[:-1]]
	elif value[0] == 'H': H = [float(item2) for item2 in value[1].split('\t')[:-1]]
	elif value[0] == 'ALPHA': alfa = [float(item2) for item2 in value[1].split('\t')[:-1]]
	elif value[0] == 'BETA': beta = [float(item2) for item2 in value[1].split('\t')[:-1]]
	elif value[0] == 'X': X = [float(item2) for item2 in value[1].split('\t')[:-1]]
	elif value[0].split(' ')[0] == 'Aptitud': Aptitud = float(value[0].split(' ')[1])

def SSystem(x):
	evaluation = []
	for i in range(ngen):
		prod1, prod2 = 1.0, 1.0
		for k in range(ngen):
			prod1 *= (x[k] ** G[ngen*i+k]) if x[k]>0 else prod1 * 0
			prod2 *= (x[k] ** H[ngen*i+k]) if x[k]>0 else prod2 * 0

		evaluation.append((alfa[i]*prod1 - beta[i]*prod2)*h)
		if evaluation <= 0: evaluation = 0

	return evaluation

y, xfin, xini = [], ntiempos*h, 0
y.append([valores[0][i][0] for i in range(ngen)])
k1, k2, k3, k4 = [], [], [], []
lista = []

print G, H, alfa, beta

for i in range(ntiempos-1):

	x = [y[i][j] for j in range(ngen)]
	k1 = SSystem(x)
	x = [y[i][j]+k1[j]/2. for j in range(ngen)]
	k2 = SSystem(x)
	x = [y[i][j]+k2[j]/2. for j in range(ngen)]
	k3 = SSystem(x)
	x = [y[i][j]+k3[j] for j in range(ngen)]
	k4 = SSystem(x)
	y.append([y[i][j]+(k1[j]+2*k2[j]+2*k3[j]+k4[j])/6. for j in range(ngen)])

	print i, round(y[-2][0], 6), round(y[-2][1], 6)
















