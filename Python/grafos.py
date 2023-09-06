import graphviz as gv
import numpy as np

archivo = 'DE-PSO-256'
nGenes = 2
path = 'Z:/'

salida = path + 'Salidas/'+archivo+'.txt'
valoresOrd = {}
calculados = {}

def leerSalida(path):
    series, genes, tam, valores, resultados = 0, 0, 0, [], []
    G, H, alfa, beta, X, aptitud = [], [], [], [], [], 0

    archivo = open(path, mode='r')
    for linea in archivo.readlines():
        datos = linea.split(' ')
        if datos[0] == 'G':
            G = [float (item) for item in datos[2].split('[')[1].split('\t')[:-1]]
        if datos[0] == 'H':
            H = [float (item) for item in datos[2].split('[')[1].split('\t')[:-1]]
        if datos[0] == 'ALPHA':
            alfa = [float (item) for item in datos[2].split('[')[1].split('\t')[:-1]]
        if datos[0] == 'BETA':
            beta = [float (item) for item in datos[2].split('[')[1].split('\t')[:-1]]
        if datos[0] == 'X':
            valores.append([float (item) for item in datos[2].split('[')[1].split('\t')[:-1]])

        if datos[0] == 'Aptitud':
            valoresOrd[float(datos[1])] = [valores[-1], G, H, alfa, beta, float(datos[1])]

    keylist = [item for item in valoresOrd.keys()]
    keylist.sort()
    valores = [valoresOrd[key] for key in keylist]
    print(valores)
    return valores




#[g2.node(str(item)) for item in range(nGenes)]
cntRender = 0
for item in leerSalida(salida):
    cnt = 0
    g2 = gv.Digraph(format='svg')
    for i in range(nGenes):
        for j in range(nGenes):
            #print i+1, j+1, item[1][i*nGenes+j], item[2][i*nGenes+j]

            if not abs(item[1][i*nGenes+j]) == 0: 
                g2.edge(str(i+1), str(j+1), style = 'dashed')#, label = str(item[1][i*nGenes+j]), labelfontsize='8.0'
            #if not item[2][i*nGenes+j] <=-1 and not item[2][i*nGenes+j] >=1: 
                #g2.edge(str(i+1), str(j+1))

    cntRender+=1
    print ("Generando", str(cntRender), item[5])
    g2.render('img/g'+str(cntRender))

        
