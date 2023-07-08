from pylab import *
import numpy as np
from scipy import interpolate
from scipy.interpolate import spline
import matplotlib.pyplot as plt


entrada = '/home/af/CUDA/CUDA-Memetic-Algorithm/Instancias/ANN5GeneratedData5Noise.txt'
salida = '/home/af/CUDA/CUDA-Memetic-Algorithm/Salidas/salidaA.txt'

def leerEntrada(path):
	series, genes, tam, valores, seriesArr = 0, 0, 0, [], []
	archivo = open(path, mode='r')
	for linea in archivo.readlines():
		datos = linea.split(' ')
		if series == 0: 
			genes, series = datos
			valores = [[] for i in range(int(genes))]
		elif tam == 0: tam = int(datos[0])
		elif len(datos) == int(genes):
			for i in range(int(genes)):
				valores[i].append(float(datos[i]))
			if len(valores[0]) == tam:
				seriesArr.append(valores)
				valores = [[] for i in range(int(genes))]

	return int(series), int(genes), int(tam), seriesArr

def leerSalida(path):
	series, genes, tam, valores, resultados = 0, 0, 0, [], []
	archivo = open(path, mode='r')
	for linea in archivo.readlines():
		datos = linea.split(' ')
		if datos[0] == 'X':
			resultados.append([float(item) for item in datos[2].split('[')[1].split('\t')[:-1]])

	print len(resultados)
	return resultados

series, genes, tam, valores = leerEntrada(entrada)
resultados = leerSalida(salida)
rXGenes = {}

for item in range(series):
	print "serie", item+1
	rXGenes[item+1] = {}
	i = 1
	for item2 in resultados[(item) * tam:(item+1) * tam]:
		print "corrida", i
		rXGenes[item+1][i] = [[], [], [], [], []]
		for item3 in range(tam):
			rXserie = item2[(item3)*genes:(item3+1)*genes]
			[rXGenes[item+1][i][item4].append(rXserie[item4]) for item4 in range(genes)]
		i+=1

x = np.arange(0, tam, 1);
y = []
j = 1
for item in valores:
	i = 1
	for item2 in item:
		y = item2
		plt.figure(str(i))
		titulo = "Gen "+str(i)#+" Serie "+ str(j)
		plt.title(titulo, fontsize = 20)
		plt.xlabel("Tiempo", fontsize = 20)
		plt.ylabel("Valor", fontsize = 20)
		plt.plot(x, y)
		print "Serie", j, i
		plt.plot(x, rXGenes[j][1][i-1])
		i+=1
	print "Salto de serie"
	j+=1
plt.legend(handler_map={j: HandlerLine2D(numpoints=4)})
plt.show()