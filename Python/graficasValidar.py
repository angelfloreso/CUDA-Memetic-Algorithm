from pylab import *
import numpy  
from matplotlib.legend_handler import HandlerLine2D

archivo = 'DE-PSO-256'
graficar = 3

path = 'Z:/'

entrada = path + 'Instancias/Tominaga2SSGeneratedData.txt'
salida = path + 'Salidas/'+archivo+'.txt'
corridas = 20
valoresOrd = {}
calculados = {}
boxplot = []

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
        elif len(datos) - 1 == int(genes):
            for i in range(int(genes)):
                valores[i].append(float(datos[i]))
            if len(valores[0]) == tam:
                seriesArr.append(valores)
                valores = [[] for i in range(int(genes))]
    return float(h), int(series), int(genes), int(tam), seriesArr

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

        if datos[0] == 'MSE':
            #print (datos[1])
            boxplot.append()

    keylist = list(valoresOrd.keys())
    sort(keylist)
    valores = [valoresOrd[key] for key in keylist]
    return valores

def procesar():
    corrida = 1
    serie = 1
    for item in valoresSalida:
        item = item[0]
        a, b = 0, ntiempos
        for item2 in range(ngen):
            #print (item[a:b], a, b, "gen", item2+1, "corrida", corrida, "serie", serie)
            if not item2+1 in calculados: calculados[item2+1]={}
            if not corrida in calculados[item2+1]: calculados[item2+1][corrida]={}
            if not serie in calculados[item2+1][corrida]: calculados[item2+1][corrida][serie]={}
            calculados[item2+1][corrida][serie] = item[a:b]
            a+=ntiempos
            b+=ntiempos
            if b==100+ntiempos:
                #print (a, b, corrida)
                a=0
                b=ntiempos
                #corrida+=1
                if corrida == corridas+1:
                    serie+=1
                    corrida = 1

        corrida+=1

def generaGrafica():
    x = np.arange(0, ntiempos, 1);
    y = []
    j = 1
    for item in valores:
        i = 1
        for item2 in item:
            y = item2
            plt.figure("Gen "+str(i)+" Serie "+ str(j)+' '+ archivo)
            titulo = "Gen "+str(i)+" Serie "+ str(j)
            plt.title(titulo, fontsize = 20)
            plt.xlabel("Tiempo", fontsize = 20)
            plt.ylabel("Valor", fontsize = 20)
            plt.plot(x, y, label="Original")
            #print "Serie", y, len(y)
            matrix = []
            for item3 in range(graficar):
                try:
                    plt.plot(x, calculados[i][item3+1][j], label="C{0}".format(item3+1))
                    matrix.append(calculados[i][item3+1][j])
                except Exception as e:
                    print (e)
            plt.plot(x, np.squeeze(np.asarray(np.asmatrix(matrix).mean(0))), linestyle="--", label="Media")
            plt.legend(handler_map={j: HandlerLine2D(numpoints=4)})
            i+=1
        #print ("Salto de serie")
        j+=1

    plt.show()


h, series, ngen, ntiempos, valores = leerEntrada(entrada)
valoresSalida = leerSalida(salida)

X = valoresSalida[0][0]
G = valoresSalida[0][1]
H = valoresSalida[0][2]
alfa = valoresSalida[0][3]
beta = valoresSalida[0][4]
errorGPU = valoresSalida[0][5]

xGenes = []
for item in valores[0]: xGenes += [item2 for item2 in item]

def RK():
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
    error = 0
    MSE = 0
    RMSE = 0
    for i in range(ntiempos):

        x = [y[i][j] for j in range(ngen)]
        k1 = SSystem(x)
        x = [y[i][j]+k1[j]/2. for j in range(ngen)]
        k2 = SSystem(x)
        x = [y[i][j]+k2[j]/2. for j in range(ngen)]
        k3 = SSystem(x)
        x = [y[i][j]+k3[j] for j in range(ngen)]
        k4 = SSystem(x)
        y.append([y[i][j]+(k1[j]+2*k2[j]+2*k3[j]+k4[j])/6. for j in range(ngen)])
        #print (i, "Gen 1 CPU", round(y[-2][0], 6), "GPU", round(X[i], 6), "Real",round(xGenes[i], 6))
        #print (i, "Gen 2 CPU", round(y[-2][1], 6), "GPU", round(X[i+ntiempos], 6), "Real", round(xGenes[i+ntiempos], 6))
        error += ((y[-2][0]-xGenes[i])/xGenes[i])**2
        error += ((y[-2][1]-xGenes[i+ntiempos])/xGenes[i+ntiempos])**2
        MSE += ((y[-2][0]-xGenes[i]))**2
        MSE += ((y[-2][1]-xGenes[i+ntiempos]))**2

    MSE = MSE/(ntiempos*ngen)
    RMSE = sqrt(MSE)

    print ("\nError Cuadratico Medio", error, "Calculado GPU", errorGPU, "MSE", MSE, "RMSE", RMSE)

procesar()
generaGrafica()
#RK()





