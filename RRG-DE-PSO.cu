/**
    Inferencia en Redes Reguladoras de Genes 
    RRGDiver.cu
	Compilar con las banderas: nvcc --default-stream per-thread -lcurand -O2 -o test RRGDiver.cu

    @author AF
    @version 1.3 06/09/16 
*/

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cmath>
#include <cfloat>
#include <cuda.h>
#include <curand.h>
#include <curand_kernel.h>
#include <pthread.h>

#define POPSIZE 256 								// Tamano de la poblacion
#define GMAX	1000								// Numero de generaciones
#define PSOGEN 	500							// Numero de iteraciones PSO
#define FPARAM  0.04f								// Parametro F para la ED
#define MAXRAND 50       							// Tamano del vector de valores random

#define HILOS_X 16									// Hilos y Bloques para Kernels
#define HILOS_Y 16
#define BLOQUES_X 16
#define BLOQUES_Y 16

#define MAXGij  4.0									// Rangos de exploracion
#define MINGij -4.0
#define MAXHij  4.0
#define MINHij -4.0
#define MAXALPHA 20.0
#define MINALPHA 0.0
#define MAXBETA 20.0
#define MINBETA 0.0

/**
    Variables de rendimiento
*/
#define CORRIDAS 20									// Numero de corridas
#define STREAMS 2									// Procesos simultaneos
#define GPUS 0 										// Maximos de GPUs (0 = Sin limite)

/**
    Diversidad
*/
#define DIVERSIDAD 50 								// Cantidad de generaciones sin mejorar fitness
#define REMPDIV 0.7 								// Porcentaje de remplazo en diversidad

/**
    PSO
*/

#define WPSO 0.7
#define C1 1.4
#define C2 1.4

/**
    Prioridad a G y H ceros
*/
#define PCOEF 0.01									// Coeficiente de penalización
#define PROF 2										// Profundidad de busqueda
#define UMBR 0.05									// Umbral de skeletalizing

#define DEBUG false									// Debuguer

typedef struct ThreadParams {						// Parametros para la funcion pthreads
	int serie;										// Serie que se esta evaluando
	int corrida;									// Corrida que se esta ejecutando
	int device;										// id del GPU que esta haciendo la evaluacion
	int control;									// id del hilo en el GPU que ejecuta la evaluacion
	float resultado;								// Mejor fitness obtenido
} ThreadParams;

pthread_mutex_t mutexsum;
pthread_t *threads;									// Arreglo para el manejo de hilos en CPU
ThreadParams **args;
cudaStream_t **streamEval;							// Arreglo de Streams para procesos simultaneos GPU
cudaStream_t **streamCalc;
cudaEvent_t *start;
cudaEvent_t *stop;
curandGenerator_t **randomGenerator;				// Generador para CURAND

FILE *fIn;											// Archivo de entrada
FILE *fOut;											// Archivo de salida
int numeroGenes,nseries,numeroTiempos;				// Variables de control para la instancia

int individuoSize_H,solucionSize_H;					// Variables de tamanos para parametros
int GSize_H,HSize_H,AlphaSize_H,BetaSize_H;

float h;											// Denota el tamaño de paso para el metodo Runge - Kutta
float MValor;										// Maxima distancia euclidiana posible calculada con los limites de G, H alfa y beta

float ***W_D;										// Variables de calculo para evolucion diferencial
float ***K_D;										// Se cambiaron todas las variables globales en arreglos de diferentes dimenciones
float **tiempos;									// para diferenciar las ejecuciones en diferentes device, hilos o corrida.
float ***auxiliar;
float *datosReales;
float ***fitness_H;
float ***fitness_D;
float ***goodness_D;
float ***minGoodness;
float ***poblacion_D;
float ***datosReales_D;
float ***mejorIndividuo;
float ***nuevoFitness_D;
float ***nuevaPoblacion_D;
float ***randomFloatValues_D;

int ***numBSF;
int ***numBSF_D;
int ***idBSF;
int ***idBSF_D;
float ***bestSoFar;
float ***bestSoFar_D;
float ***MSE;
float ***MSE_D;
float ***RMSE;
float ***RMSE_D;
int ***mustChange_D;
int ***randomIndices_D;
int nswarm, ndim;

dim3 bloque(HILOS_X,HILOS_Y);						// Dimenciones de bloques e hilos
dim3 malla(BLOQUES_X,BLOQUES_Y);

/**
    MACRO para corroborar si una llamada a CURAND fue exitosa.

    @param x funcion de curand.
*/
#define CURAND_CALL(x) if((x)!=CURAND_STATUS_SUCCESS) { \
    printf("CURAND - Error en %s:%d, Codigo: %d\n",__FILE__,__LINE__, (x));\
    cudaDeviceReset();\
    }

/**
    MACRO para corroborar si una llamada a CUDA fue exitosa.

    @param x funcion de cuda.
*/
#define CUDA_CALL(x) if((x) != cudaSuccess) { \
    printf("CUDA - Error en %s:%d, %s\n",__FILE__,__LINE__, cudaGetErrorString(x));     \
    cudaDeviceReset();\
    }

/**
    Device Realiza el calculo de la ecuacion diferencial.

    @param X valor en tiempo t de cada gen.
    @param i gen que se esta evaluando.
    @param ngen numero de genes.
    @param G valores G.
    @param H valores H.
    @param alpha valores alpha.
    @param beta valores beta.

    @return valor en el tiempo.
*/
__device__ float SSystem(float *X, int i, int ngen, float * G, float * H, float * alpha, float * beta){
	float evaluation,prod1,prod2;
	int j;	

	for(j = 0,prod1 = 1,prod2 = 1; j < ngen; j++){
		prod1=prod1*pow(X[j],G[i*ngen+j]);
		prod2=prod2*pow(X[j],H[i*ngen+j]);
	}
	evaluation = alpha[i]*prod1-beta[i]*prod2;
	if(isnan(evaluation) || isinf(evaluation))
		evaluation=0;

	return evaluation;
}

/**
    Device Metodo Runge - Kutta.

    @param G valores G.
    @param H valores H.
    @param alph valores alpha.
    @param beta valores beta.
    @param X auxiliar para calculo del valor de salida.
    @param B datos reales.
    @param w auxiliar para calculo del sistema s.
    @param k auxiliar para calculo del sistema s.
    @param numeroTiempos numero de tiempos.
    @param numeroGenes numero de genes.
    @param h parametro de paso para RK.
    @param i tiempo que se esta evaluando

    @change Se modifico el metodo para que se evaluaran todos los tiempos en un hilo diferente.

*/
__device__ void rungeKutta(float *G, float * H, float *alph, float * beta, float *X, float *B, float *w, float * k, int numeroTiempos, int numeroGenes, float h, int i){
	int j;
		
	for(j = 0; j < numeroGenes; j++)
		w[j] = X[j*numeroTiempos+i];
	for(j = 0; j < numeroGenes; j++)
		k[j] = h*SSystem( w,j, numeroGenes, G,H, alph, beta);
	for(j = 0; j < numeroGenes; j++)
		w[j] = X[j*numeroTiempos+i]+k[0+j]/2;
	for(j = 0; j < numeroGenes; j++)
		k[1*numeroGenes+j] = h*SSystem(  w,j, numeroGenes, G,H, alph, beta);
	for(j = 0; j < numeroGenes; j++)
		w[j] = X[j*numeroTiempos+i]+k[1*numeroGenes+j]/2;
	for(j = 0; j < numeroGenes; j++)
		k[2*numeroGenes+j] = h*SSystem(  w,j, numeroGenes, G,H, alph, beta);
	for(j = 0; j < numeroGenes; j++)
		w[j] = X[j*numeroTiempos+i]+k[2*numeroGenes+j];
	for(j = 0; j < numeroGenes; j++)
		k[3*numeroGenes+j] = h*SSystem( w,j, numeroGenes, G,H, alph, beta);
	for(j = 0; j < numeroGenes; j++){
		X[j*numeroTiempos+i+1] = X[j*numeroTiempos+i]+(k[0+j]+2*k[1*numeroGenes+j]+2*k[2*numeroGenes+j]+k[3*numeroGenes+j])/6;
		if(X[j*numeroTiempos+i+1] < 0)
			X[j*numeroTiempos+i+1] = 0;
	}

}

/**
    Device Calcula el error cuadratico medio del arreglo de todos los valores calculados de X.

    @param X_i valores calculados para el individuo.
    @param realData valores reales.
    @param aptitud fitness del individuo.
    @param numeroGenes numero de genes.
    @param numeroTiempos numero de tiempos.
    @param ini individuo que se esta evaluando.

*/
__device__ void errorCuadraticoMedio(float *X_i, float *realData, float *aptitud, int numeroGenes, int numeroTiempos, int ini, float *G, float *H, float *auxiliar){
	float sum = 0, sumProm = 0;
	int n = numeroGenes*numeroTiempos;
	float gAbs = 0, hAbs = 0;
	int i, j, k;
	float selected;

	for(k = 0; k < numeroGenes; k++){
		for(j = 0; j < numeroGenes; j++){
			auxiliar[j] = abs(G[k*numeroGenes+j]);
			auxiliar[numeroGenes+j] = abs(H[k*numeroGenes+j]);
		}
		for(i = 0; i < numeroGenes; i++){
			selected = auxiliar[i];
			j = i - 1;
			while ((j >= 0) && (selected < auxiliar[j])) {
				auxiliar[j+1] = auxiliar[j];
				j--;
			}
			auxiliar[j+1] = selected;
			selected = auxiliar[numeroGenes+i];
			j = i - 1;
			while ((j >= 0) && (selected < auxiliar[numeroGenes+j])) {
				auxiliar[numeroGenes+j+1] = auxiliar[numeroGenes+j];
				j--;
			}
			auxiliar[numeroGenes+j+1] = selected;
		}
		for(j = 0; j < PROF; j++){
			gAbs += auxiliar[j];
			hAbs += auxiliar[numeroGenes+j];
			//printf("g %f, h %f, sumg %f, sumh %f ", auxiliar[j], auxiliar[numeroGenes+j], gAbs, hAbs);
		}
		//printf("\n");
		__syncthreads();
	}

	for(int i = 0; i < n; i++){			
		sum += powf((X_i[i]-realData[i])/(realData[i]),2);
		sumProm += powf((X_i[i]-realData[i]),2);
		__syncthreads();
	}

	//printf("sum %f, gAbs %f, hAbs %f, sumTotal %f \n", sum, gAbs, hAbs, sum + (PCOEF * (gAbs + hAbs)));
	
	sum += PCOEF * (gAbs + hAbs);
	
	if(isinf(sum))
		sum = FLT_MAX;

	aptitud[ini] = sum;
	aptitud[ini + POPSIZE] = (sumProm/n);
	aptitud[ini + (2*POPSIZE)] = sqrt(sumProm/n);
}
 
/**
    Device Evalua a los miembros de poblacion, el indice global del thread toma el apuntador 
    al individuo correspondiente.

    @param poblacion individuos de la poblacion.
    @param realData valores reales.
    @param w auxiliar para calculo del sistema s.
    @param k auxiliar para calculo del sistema s.
    @param numeroTiempos numero de tiempos.
    @param numeroGenes numero de genes.
    @param h parametro de paso para RK.
    @param indSize tamano de cada individuo.
    @param aptitud fitness de la poblacion.

    @change se agrego un desplazamiento a las variables W y K, ya que se traslapaba la informacion 
    y no se hacia correctamente la evaluacion
*/
__global__ void evaluaPoblacion(float *poblacion,float *realData, float *W, float *K, int numeroTiempos,int numeroGenes,float h,int indSize,float *aptitud, float *auxiliar){
	int i = threadIdx.x + (blockIdx.x * blockDim.x);	
	int j;
	int k = 0;

	float *individuo,*G_i,*H_i,*A_i,*B_i,*X_i;

	while(i < POPSIZE){			
		individuo = poblacion+i*indSize; //i*indSize es el offset hasta el siguiente individuo
		G_i = individuo;
		H_i = individuo + numeroGenes*numeroGenes;
		A_i = individuo + 2*numeroGenes*numeroGenes;
		B_i = individuo + 2*numeroGenes*numeroGenes + numeroGenes;
		X_i = individuo + 2*numeroGenes*numeroGenes + 2*numeroGenes;

		W += i*numeroGenes;
		K += i*4*numeroGenes;
		
		/*Resolvemos el sistema de ecuaciones diferenciales para obtener las predicciones X_i*/
		
		for(j = 0; j < numeroGenes; j++)
			X_i[numeroTiempos*j] = realData[numeroTiempos*j];
		
		k = 0;
		while(k < numeroTiempos-1){
			rungeKutta(G_i,H_i,A_i,B_i,X_i,realData,W,K,numeroTiempos,numeroGenes,h, k);
			__syncthreads();
			k += 1;
		}
				
		/*Error cuadratico medio*/
		errorCuadraticoMedio(X_i, realData, aptitud, numeroGenes, numeroTiempos, i, G_i, H_i, auxiliar);
		i += blockDim.x * gridDim.x;

	}
}

__global__ void iniSwarm(float *swarm, float *pbest, int indSize, float *vMatrix, float *randomValues){
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int j = threadIdx.y + blockIdx.y * blockDim.y;

	while(i < POPSIZE){

		while( j < indSize){
			pbest[j+i*indSize] = swarm[j+i*indSize];

			vMatrix[j+i*indSize] = randomValues[j+i*indSize];

			j += blockDim.y * gridDim.y;
		}
		i += blockDim.x * gridDim.x;
	}
	
}

__global__ void swarmPSO(float *swarm, float *gbest, float *pbest, float *vMatrix,int indSize,int solucionSize, float *randomValues, int GSize, int HSize, int aSize, int bSize){
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int j = threadIdx.y + blockIdx.y * blockDim.y;
	int k = 0;	
	float upperBound,lowerBound;

	while(i < POPSIZE){

		while( j < indSize-solucionSize){

			//pbest[j+i*indSize] = pbest[j+i*indSize] - swarm[j+i*indSize];
			//swarm[j+i*indSize] = gbest[j] - swarm[j+i*indSize];

			vMatrix[j+i*indSize] = WPSO * vMatrix[j+i*indSize] + C1 * (randomValues[k+i*indSize] * pbest[j+i*indSize] - swarm[j+i*indSize]) + C2 * (randomValues[(k+1)+i*indSize] * gbest[j] - swarm[j+i*indSize]);


			if(j < GSize){
				upperBound = MAXGij;
				lowerBound = MINGij;
			}else if( j < GSize+HSize){
				upperBound = MAXHij;
				lowerBound = MINHij;
			}else if(j < GSize+HSize+aSize){
				upperBound = MAXALPHA;
				lowerBound = MINALPHA;		
			}else if(j < GSize+HSize+aSize+bSize){
				upperBound = MAXBETA;
				lowerBound = MINBETA;		
			}

			swarm[j+i*indSize] += vMatrix[j+i*indSize];

			//printf("%f, %f, %f, %d, %d\n", swarm[j+i*indSize], upperBound, lowerBound, swarm[j+i*indSize] > upperBound, swarm[j+i*indSize] < lowerBound);

			if (swarm[j+i*indSize] > upperBound || swarm[j+i*indSize] < lowerBound ){

				//printf("cambiando valor %f, %f, %f\n", swarm[j+i*indSize], upperBound, lowerBound);

				swarm[j+i*indSize] = lowerBound + randomValues[k+i*indSize]*(upperBound-lowerBound);

				//printf("nuevo valor %f, %f, %f\n", swarm[j+i*indSize], upperBound, lowerBound);
			}

			//else{

				//printf("no cambio valor %f, %f, %f\n", swarm[j+i*indSize], upperBound, lowerBound);

				
			//}

			//veridicar que se encuentre dentro de los limites
			
			//printf("i %d, j %d, %f, %f\n",i, j, swarm[j+i*indSize], gbest[j]);
			k += 2;
			j += blockDim.y * gridDim.y;
		}
		i += blockDim.x * gridDim.x;
	}
	
}

/**
    Device Obtiene la distancia de un individuo con la poblacion
    y calcula el Goodness con g(ini) = f(s) + e^be / menor.

    @param poblacion individuos de la poblacion.
    @param realData valores reales.
    @param w auxiliar para calculo del sistema s.
    @param k auxiliar para calculo del sistema s.
    @param numeroTiempos numero de tiempos.
    @param numeroGenes numero de genes.
    @param h parametro de paso para RK.
    @param indSize tamano de cada individuo.
    @param aptitud fitness de la poblacion.

    @return valores del goodness del individuo.
*/
__device__ void obtenerGoodness(float *dist, int ini, float *salida, float *aptitud, int numeroGenes, float MValor, float *bestSoFar){
	float menor = FLT_MAX;
	float valor = 0;
	float be = 0.08;

	
	for(int i = ini*POPSIZE; i < ini*POPSIZE+POPSIZE; i++){
		valor = sqrt(dist[i]);
		if (valor < menor)
			menor = valor;
	}
	be*=MValor;
	if (aptitud[ini] <= *bestSoFar) {
		salida[ini] = 0;
	}
	else salida[ini] = aptitud[ini] + exp(be/menor);
	//printf("aptitud %f, salida %f, menor %f, %f\n", aptitud[ini], salida[ini], menor, MValor);
	
}

__global__ void obtenerMetricas(float *bestSoFar, float *MSE, float *RMSE, float *aptitud, int *numBSF, int *idBSF){
	float menor = FLT_MAX;
	float valor = 0;
	int idMenor = 0;

	for(int i = 0; i < POPSIZE; i++){
		__syncthreads();
		valor = aptitud[i];
		if (valor <= menor){
			menor = valor;
			idMenor = i;
		}
	}

	if (menor < *bestSoFar){
		*bestSoFar = aptitud[idMenor];
		*MSE = aptitud[idMenor + POPSIZE];
		*RMSE = aptitud[idMenor + (2*POPSIZE)];
		*idBSF = idMenor;
		*numBSF = 0;
	}
	else{
		*numBSF+=1;
	}
}

/**
    Device Calcula la distancia euclidiana de cada individuo con los demas de la poblacion.

    @param poblacion individuos de la poblacion.
    @param numeroGenes numero de genes.
    @param indSize tamano de cada individuo.
    @param aptitud fitness de la poblacion.
    @param distancia auxiliar para calculo del goodness.
    @param sum auxiliar para calculo de la minima distancia.
    @param MValor valor maximo de distancia.
*/
__global__ void calculaDistancia(float *poblacion, int numeroGenes,int indSize,float *aptitud,float *distancia,float *sum, float MValor, float *bestSoFar){
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int j = threadIdx.y + blockIdx.y * blockDim.y;
	int k = 0;

	float *individuoi, *individuoj;

	while(i < POPSIZE){
		individuoi = poblacion+i*indSize;

		while(j < POPSIZE){			
			individuoj = poblacion+j*indSize;
			sum[POPSIZE*i+j] = 0;
			while(k < 2*numeroGenes*numeroGenes + 2*numeroGenes){

				if (i != j){
					sum[POPSIZE*i+j] += powf(individuoi[k]-individuoj[k],2);
					//printf("%f contra %f en i %d, j %d y k %d = %f = %f \n", individuoi[k], individuoj[k], i, j, k, powf(individuoi[k]-individuoj[k],2), sum[POPSIZE*i+j]);
				}
				else{
					sum[POPSIZE*i+j] = nan("");
				}
				k++;

			}
			//printf("en i %d, j %d = %f = %f \n", i, j, sum[POPSIZE*i+j], sqrt(sum[POPSIZE*i+j]));
			j += blockDim.y * gridDim.y;
			
		}
		obtenerGoodness(sum, i, distancia, aptitud, numeroGenes, MValor, bestSoFar);
		i += blockDim.x * gridDim.x;
	}
}

/**
    Device Calcula el valor a partir del cual se reemplazaran los individuos para conservar diversidad, 
	Ordena los individuos del menor al mayor, a partir del porcentaje de reemplazo elige al primer individuo 
	que no sera elejido y guarda su goodness, cualquier individuo con un valor menor sera reemplazado.

    @param array auxiliar para ordenamiento.
    @param goodness goodness de la poblacion.
    @param minGoodness valor de reemplazo.
*/
__global__ void buscaRemplazo(float *array, float *goodness, float *minGoodness) {
	int i, j, k;
	float selected;
	int cambio = (int )POPSIZE-POPSIZE*REMPDIV;

	for(k = 0; k < POPSIZE; k++)
		array[k] = goodness[k];

	for (i = 1; i < POPSIZE; i++){
		selected = array[i];
		j = i - 1;
		while ((j >= 0) && (selected < array[j])) {
			array[j+1] = array[j];
			j--;
		}
		array[j+1] = selected;
	}
	//printf("reemplazo en %d = %f\n", cambio, array[cambio]);
	*minGoodness = array[cambio];
}

/**
    Device Crea diversidad en la poblacion modificandlos valores de g, h, alfa y beta.
    Cada parametro tiene la probabilidad pMutacion de ser modificado.

    @param array auxiliar para ordenamiento.
    @param goodness goodness de la poblacion.
    @param minGoodness valor de reemplazo.
    @param minGoodness valor de reemplazo.
    @param minGoodness valor de reemplazo.
    @param minGoodness valor de reemplazo.
    @param minGoodness valor de reemplazo.
    @param minGoodness valor de reemplazo.

*/
__global__ void diversifica(float *poblacion, float *aptitud, int indSize, int GSize, int HSize, int aSize, int bSize, int solucionSize, float *randomValues, float *minGoodness, int randControl){
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int j = threadIdx.y + blockIdx.y * blockDim.y;
	int k;
	bool lower = false;
	float upperBound,lowerBound;
	float pMutacion = 0;

	while(i < POPSIZE-1){
		if (aptitud[i] >= *minGoodness){
			while( j < indSize-solucionSize){
				k = i*indSize+j;
				if(j < GSize){
					upperBound = MAXGij;
					lowerBound = MINGij;
					lower = true;
				}else if( j < GSize+HSize){
					upperBound = MAXHij;
					lowerBound = MINHij;	
					lower = true;	
				}else if(j < GSize+HSize+aSize){
					upperBound = MAXALPHA;
					lowerBound = MINALPHA;		
				}else if(j < GSize+HSize+aSize+bSize){
					upperBound = MAXBETA;
					lowerBound = MINBETA;		
				}
				if(randomValues[(i*(indSize-solucionSize)+j)+randControl] > pMutacion){
					poblacion[k] = lowerBound + randomValues[(i*(indSize-solucionSize)+j)+randControl]*(upperBound-lowerBound);
					if (lower && abs(poblacion[k]) < UMBR) poblacion[k] = 0;
				}
				j += blockDim.y * gridDim.y;
			}
		}
		i += blockDim.x * gridDim.x;
	}

}
	  
/*Libera recursos*/
void liberaMemoria(){
	free(fitness_H);
	free(mejorIndividuo);	
	free(datosReales);
} 

void liberaMemoriaDevice(int device, int control){
	CUDA_CALL(cudaFree(poblacion_D[device][control]));
	CUDA_CALL(cudaFree(fitness_D[device][control]));
	CUDA_CALL(cudaFree(goodness_D[device][control]));
	CUDA_CALL(cudaFree(nuevaPoblacion_D[device][control]));
	CUDA_CALL(cudaFree(nuevoFitness_D[device][control]));
	CUDA_CALL(cudaFree(randomIndices_D[device][control]));
	CUDA_CALL(cudaFree(randomFloatValues_D[device][control]));
	CUDA_CALL(cudaFree(mustChange_D[device][control]));
	CUDA_CALL(cudaFree(W_D[device][control]));
	CUDA_CALL(cudaFree(K_D[device][control]));
	CUDA_CALL(cudaFree(auxiliar[device][control]));
	CUDA_CALL(cudaFree(minGoodness[device][control]));
	CUDA_CALL(cudaFreeHost(numBSF[device][control]));
	CUDA_CALL(cudaFreeHost(idBSF[device][control]));
	CUDA_CALL(cudaFreeHost(bestSoFar[device][control]));

	curandDestroyGenerator(randomGenerator[device][control]);
	//CUDA_CALL( cudaStreamDestroy(streamEval[device][control]) );
	//CUDA_CALL( cudaStreamDestroy(streamCalc[device][control]) );
	free(fitness_H[device][control]);
} 

/*Inicializa la poblacion. Dado que todos los parametros estan codificados secuencialmente en el arreglo, recibe los tamanos de los parametros. Podria recibir los limites en su lugar. */
__global__ void inicializaPoblacion(float *poblacion,int indSize, int GSize, int HSize, int aSize, int bSize, int solucionSize,float *randomValues, float *bestSoFar, int *numBSF){    
    int i = threadIdx.x + blockIdx.x * blockDim.x;
	int j = threadIdx.y + blockIdx.y * blockDim.y;
	int k;
	bool lower = false;
	float upperBound,lowerBound;
	*bestSoFar = FLT_MAX;
	*numBSF = 0;

	while(i < POPSIZE){		
		while( j < indSize-solucionSize){
			k = i*indSize+j;
			/*Limites para cada parametro, el ultimo caso (la solucion al sistema de ecuaciones diferenciales) es inicializacion en 0*/
			if(j < GSize){
				upperBound = MAXGij;
				lowerBound = MINGij;
				lower = true;
			}else if( j < GSize+HSize){
				upperBound = MAXHij;
				lowerBound = MINHij;
				lower = true;		
			}else if(j < GSize+HSize+aSize){
				upperBound = MAXALPHA;
				lowerBound = MINALPHA;		
			}else if(j < GSize+HSize+aSize+bSize){
				upperBound = MAXBETA;
				lowerBound = MINBETA;		
			}else{
				upperBound = lowerBound = 0;
			}
			poblacion[k] = lowerBound + randomValues[i*(indSize-solucionSize)+j]*(upperBound-lowerBound);

			if (lower && abs(poblacion[k]) < UMBR) poblacion[k] = 0;
					
			j += blockDim.y * gridDim.y;
		}
		i += blockDim.x * gridDim.x;
	}
}
	  
/* Pide memoria */
void init(int numGpus){	
	GSize_H = numeroGenes*numeroGenes;					/*Memoria para G*/  
	HSize_H = numeroGenes*numeroGenes;			 		/*Memoria para H*/  
	AlphaSize_H = numeroGenes; 							/*Memoria para alpha*/
	BetaSize_H = numeroGenes;							/*Memoria para beta*/
	solucionSize_H = numeroGenes*numeroTiempos;			/*Memoria para la solucion a la ecuacion diferencial*/
	individuoSize_H = GSize_H + HSize_H + AlphaSize_H + BetaSize_H + solucionSize_H;
	MValor = sqrt( (numeroGenes*powf(MINGij-MAXGij,2)) + (numeroGenes*powf(MINHij-MAXHij,2)) + (numeroGenes*numeroGenes*powf(MINALPHA-MAXALPHA,2)) + (numeroGenes*numeroGenes*powf(MINBETA-MAXBETA,2)) );
	
	nswarm = POPSIZE;
	ndim = individuoSize_H;

	datosReales = 															(float*) malloc(sizeof(float)*numeroGenes*numeroTiempos);
	poblacion_D = 															(float***) malloc(sizeof(float**) * numGpus);
	fitness_D = 															(float***) malloc(sizeof(float**) * numGpus);
	goodness_D = 															(float***) malloc(sizeof(float**) * numGpus);
	nuevaPoblacion_D = 														(float***) malloc(sizeof(float**) * numGpus);
	nuevoFitness_D = 														(float***) malloc(sizeof(float**) * numGpus);
	randomFloatValues_D = 													(float***) malloc(sizeof(float**) * numGpus);
	fitness_H = 															(float***) malloc(sizeof(float**) * numGpus);
	mejorIndividuo = 														(float***) malloc(sizeof(float**) * numGpus);
	W_D = 																	(float***) malloc(sizeof(float**) * numGpus);
	K_D = 																	(float***) malloc(sizeof(float**) * numGpus);
	randomIndices_D = 														(int***) malloc(sizeof(int**) * numGpus);
	mustChange_D = 															(int***) malloc(sizeof(int**) * numGpus);
	randomGenerator = 														(curandGenerator_t**) malloc(sizeof(curandGenerator_t*) * numGpus * numGpus);

	auxiliar = 																(float***) malloc(sizeof(float**) * numGpus);
	minGoodness = 															(float***) malloc(sizeof(float**) * numGpus);

	numBSF = 																(int***) malloc(sizeof(int**) * numGpus);
	numBSF_D = 																(int***) malloc(sizeof(int**) * numGpus);

	idBSF = 																(int***) malloc(sizeof(int**) * numGpus);
	idBSF_D = 																(int***) malloc(sizeof(int**) * numGpus);

	bestSoFar = 															(float***) malloc(sizeof(float**) * numGpus);
	bestSoFar_D = 															(float***) malloc(sizeof(float**) * numGpus);
	MSE = 																	(float***) malloc(sizeof(float**) * numGpus);
	MSE_D = 																(float***) malloc(sizeof(float**) * numGpus);
	RMSE = 																	(float***) malloc(sizeof(float**) * numGpus);
	RMSE_D = 																(float***) malloc(sizeof(float**) * numGpus);

}

/* Inicializa los arreglos que se utilizarán en determinado GPU*/
void initDevice(int deviceId){
	poblacion_D[deviceId] = 												(float**) malloc(sizeof(float*)*CORRIDAS);
	fitness_D[deviceId] = 													(float**) malloc(sizeof(float*)*CORRIDAS);
	goodness_D[deviceId] = 													(float**) malloc(sizeof(float*)*CORRIDAS);
	nuevaPoblacion_D[deviceId] = 											(float**) malloc(sizeof(float*)*CORRIDAS);
	nuevoFitness_D[deviceId] = 												(float**) malloc(sizeof(float*)*CORRIDAS);
	randomFloatValues_D[deviceId] = 										(float**) malloc(sizeof(float*)*CORRIDAS);
	fitness_H[deviceId] = 													(float**) malloc(sizeof(float*)*CORRIDAS);
	W_D[deviceId] = 														(float**) malloc(sizeof(float*)*CORRIDAS);
	K_D[deviceId] = 														(float**) malloc(sizeof(float*)*CORRIDAS);
	mejorIndividuo[deviceId] = 												(float**) malloc(sizeof(float*)*CORRIDAS);
	mustChange_D[deviceId] = 												(int**) malloc(sizeof(int*)*CORRIDAS);
	randomIndices_D[deviceId] = 											(int**) malloc(sizeof(int*)*CORRIDAS);
	randomGenerator[deviceId] = 											(curandGenerator_t*) malloc(sizeof(curandGenerator_t) * CORRIDAS);
	streamEval[deviceId] = 													(cudaStream_t*) malloc(sizeof(cudaStream_t) * CORRIDAS);
	streamCalc[deviceId] = 													(cudaStream_t*) malloc(sizeof(cudaStream_t) * CORRIDAS);
	auxiliar[deviceId] = 													(float**) malloc(sizeof(float*)*CORRIDAS);
	minGoodness[deviceId] = 												(float**) malloc(sizeof(float*)*CORRIDAS);
	bestSoFar[deviceId] = 													(float**) malloc(sizeof(float*)*CORRIDAS);
	bestSoFar_D[deviceId] = 												(float**) malloc(sizeof(float*)*CORRIDAS);
	MSE[deviceId] = 														(float**) malloc(sizeof(float*)*CORRIDAS);
	MSE_D[deviceId] = 														(float**) malloc(sizeof(float*)*CORRIDAS);
	RMSE[deviceId] = 														(float**) malloc(sizeof(float*)*CORRIDAS);
	RMSE_D[deviceId] = 														(float**) malloc(sizeof(float*)*CORRIDAS);
	numBSF[deviceId] = 														(int**) malloc(sizeof(int*)*CORRIDAS);
	numBSF_D[deviceId] = 													(int**) malloc(sizeof(int*)*CORRIDAS);
	idBSF[deviceId] = 														(int**) malloc(sizeof(int*)*CORRIDAS);
	idBSF_D[deviceId] = 													(int**) malloc(sizeof(int*)*CORRIDAS);

}

/* Inicializa los arreglos y crea los Streams y el generador de numeros aleatorios en el device correspondiente */
void initDeviceControl(int deviceId, int control){
	mejorIndividuo[deviceId][control] = 									(float*) malloc(sizeof(float)*individuoSize_H);
	poblacion_D[deviceId][control] = 										(float*) malloc(sizeof(float)*individuoSize_H*POPSIZE);
	fitness_H[deviceId][control] = 											(float*) malloc(sizeof(float)*POPSIZE);
	randomFloatValues_D[deviceId][control] = 								(float*) malloc(sizeof(float)*(individuoSize_H-solucionSize_H)*POPSIZE*CORRIDAS);
	
	fitness_D[deviceId][control] = 											(float*) malloc(sizeof(float)*POPSIZE*3);
	nuevoFitness_D[deviceId][control] = 									(float*) malloc(sizeof(float)*POPSIZE*3);
	
	goodness_D[deviceId][control] = 										(float*) malloc(sizeof(float)*POPSIZE);
	nuevaPoblacion_D[deviceId][control] = 									(float*) malloc(sizeof(float)*individuoSize_H*POPSIZE);
	mustChange_D[deviceId][control] = 										(int*) malloc(sizeof(int)*POPSIZE);
	randomIndices_D[deviceId][control] = 									(int*) malloc(sizeof(int)*POPSIZE*3);
	W_D[deviceId][control] = 												(float*) malloc(sizeof(float)*numeroGenes*POPSIZE*numeroTiempos*2);
	K_D[deviceId][control] = 												(float*) malloc(sizeof(float)*4*numeroGenes*POPSIZE*numeroTiempos*2);
	auxiliar[deviceId][control] = 											(float*) malloc(sizeof(float)*POPSIZE*POPSIZE);
	minGoodness[deviceId][control] = 										(float*) malloc(sizeof(float));
	numBSF[deviceId][control] = 											(int*) malloc(sizeof(int));
	numBSF_D[deviceId][control] = 											(int*) malloc(sizeof(int));
	idBSF[deviceId][control] = 												(int*) malloc(sizeof(int));
	idBSF_D[deviceId][control] = 											(int*) malloc(sizeof(int));
	bestSoFar[deviceId][control] = 											(float*) malloc(sizeof(float));
	bestSoFar_D[deviceId][control] = 										(float*) malloc(sizeof(float));
	MSE[deviceId][control] = 												(float*) malloc(sizeof(float));
	MSE_D[deviceId][control] = 												(float*) malloc(sizeof(float));
	RMSE[deviceId][control] = 												(float*) malloc(sizeof(float));
	RMSE_D[deviceId][control] = 											(float*) malloc(sizeof(float));

	CUDA_CALL( cudaSetDevice(deviceId) );
	CUDA_CALL(cudaMalloc(&poblacion_D[deviceId][control],					sizeof(float)*individuoSize_H*POPSIZE));
	CUDA_CALL(cudaMalloc(&nuevaPoblacion_D[deviceId][control],				sizeof(float)*individuoSize_H*POPSIZE));
	CUDA_CALL(cudaMalloc((void **)&randomFloatValues_D[deviceId][control],	sizeof(float)*(individuoSize_H-solucionSize_H)*POPSIZE*MAXRAND) );
	CUDA_CALL(cudaMalloc(&fitness_D[deviceId][control],						sizeof(float)*POPSIZE*3));
	CUDA_CALL(cudaMalloc(&nuevoFitness_D[deviceId][control],				sizeof(float)*POPSIZE*3));	
	CUDA_CALL(cudaMalloc(&goodness_D[deviceId][control],					sizeof(float)*POPSIZE));
	CUDA_CALL(cudaMalloc(&randomIndices_D[deviceId][control] ,				3*POPSIZE*sizeof(int) ) );
	CUDA_CALL(cudaMalloc(&mustChange_D[deviceId][control] ,					POPSIZE*sizeof(int) ) );
	CUDA_CALL(cudaMalloc(&W_D[deviceId][control],						 	sizeof(float)*numeroGenes*POPSIZE*numeroTiempos*2));
	CUDA_CALL(cudaMalloc(&K_D[deviceId][control],						 	sizeof(float)*4*numeroGenes*POPSIZE*numeroTiempos*2));
	
	CUDA_CALL(cudaMalloc(&auxiliar[deviceId][control],						sizeof(float) *POPSIZE*POPSIZE*individuoSize_H));
	CUDA_CALL(cudaMalloc(&minGoodness[deviceId][control],					sizeof(float)));

	CUDA_CALL( cudaHostAlloc((void**) &numBSF[deviceId][control],			sizeof(int), cudaHostAllocMapped) );
 	CUDA_CALL( cudaHostGetDevicePointer(&numBSF_D[deviceId][control],		numBSF[deviceId][control], 0) );
 	CUDA_CALL( cudaHostAlloc((void**) &idBSF[deviceId][control],			sizeof(int), cudaHostAllocMapped) );
 	CUDA_CALL( cudaHostGetDevicePointer(&idBSF_D[deviceId][control],		idBSF[deviceId][control], 0) );
 	CUDA_CALL( cudaHostAlloc((void**) &bestSoFar[deviceId][control],		sizeof(float), cudaHostAllocMapped) );
 	CUDA_CALL( cudaHostGetDevicePointer(&bestSoFar_D[deviceId][control],	bestSoFar[deviceId][control], 0) );
 	CUDA_CALL( cudaHostAlloc((void**) &MSE[deviceId][control],				sizeof(float), cudaHostAllocMapped) );
 	CUDA_CALL( cudaHostGetDevicePointer(&MSE_D[deviceId][control],			MSE[deviceId][control], 0) );
 	CUDA_CALL( cudaHostAlloc((void**) &RMSE[deviceId][control],				sizeof(float), cudaHostAllocMapped) );
 	CUDA_CALL( cudaHostGetDevicePointer(&RMSE_D[deviceId][control],			RMSE[deviceId][control], 0) );

    srand(time(NULL));
	/*Estableciendo la semilla aleatoria para CURAND*/
    CURAND_CALL(curandCreateGenerator(&randomGenerator[deviceId][control],CURAND_RNG_PSEUDO_DEFAULT));
    CURAND_CALL(curandSetPseudoRandomGeneratorSeed(randomGenerator[deviceId][control], rand()));
	/*Pidiendo memoria en device*/
	/*Para resolver la ecuacion diferencial*/
	//CUDA_CALL( cudaStreamCreate(&streamEval[deviceId][control]) );
	//CUDA_CALL( cudaStreamCreateWithFlags(&streamCalc[deviceId][control],cudaStreamNonBlocking) );
}

/*Compara las aptitudes de la poblacion original (contenida en fitnessPoblacionOriginal) y la poblacion creada (contenida en fitnessPoblacionNueva)
	Si el individuo en la poblacion original tiene valor de fitness mas bajo no se reemplaza, en caso contrario si*/
__global__ void comparaPoblacion(float *fitnessPoblacionOriginal,float *fitnessPoblacionNueva, int *mustChange){
    int i = threadIdx.x + blockIdx.x * blockDim.x;

	while( i < POPSIZE){
		if(fitnessPoblacionOriginal[i] > fitnessPoblacionNueva[i])
			mustChange[i] = 1;
		else 
			mustChange[i] = 0;
		i += blockDim.x * gridDim.x;
	}    
}

/*Cada uno de estos kernels reemplaza indSize elementos de original por nuevo si mustChange esta en true*/
__global__ void seleccionaPoblacion(float *original, float *nuevo, int *mustChange, int indSize){
    int i = threadIdx.x + blockIdx.x * blockDim.x;
	int j = threadIdx.y + blockIdx.y * blockDim.y;
	int k;	

	while(i < POPSIZE){		
		while( j < indSize && mustChange[i] == 1){
			k = i*indSize+j;		
			original[k] = nuevo[k];
			j += blockDim.y * gridDim.y;
		}
		i += blockDim.x * gridDim.x;
	}
	
}

/*Cada uno de estos kernels reemplaza indSize elementos de original por nuevo si mustChange esta en true*/
__global__ void seleccionaFitness(float *original, float *nuevo, int *mustChange, int indSize){
    int i = threadIdx.x + blockIdx.x * blockDim.x;
	int j = threadIdx.y + blockIdx.y * blockDim.y;
	int k;	

	while(i < POPSIZE){		
		while( j < indSize && mustChange[i] == 1){
			k = i*indSize+j;		
			original[k] = nuevo[k];
			original[k + POPSIZE] = nuevo[k + POPSIZE];
			original[k + (2*POPSIZE)] = nuevo[k + (2*POPSIZE)];
			j += blockDim.y * gridDim.y;
		}
		i += blockDim.x * gridDim.x;
	}
	
}

/*Introduce en el arreglo indices 3 elementos aleatorios diferentes del id del hilo*/
__global__ void calculaIndicesAleatorios(int *indices,int seed){
    int i = threadIdx.x + blockIdx.x * blockDim.x;
	
	while( i < POPSIZE){
		curandState state;
		curand_init(seed,i,0,&state);

		indices[3*i] = curand( &state ) % POPSIZE;
		indices[3*i+1] = curand( &state ) % POPSIZE;
		indices[3*i+2] = curand( &state ) % POPSIZE;

		while( (i == indices[3*i]) || (i == indices[3*i+1]) || (i == indices[3*i+2])
			|| (indices[3*i] == indices[3*i+1]) || (indices[3*i] == indices[3*i+2])
			|| (indices[3*i+1] == indices[3*i+2]) ){
			indices[3*i] = curand( &state ) % POPSIZE;
			indices[3*i+1] = curand( &state ) % POPSIZE;
			indices[3*i+2] = curand( &state ) % POPSIZE;
		}
		i += blockDim.x * gridDim.x;
	}
}

/*Crea la poblacion Siguiente, los randomValues dan una probablidad de mutacion en funcion de Cr. La mutacion esta en funcion de los indices aleatorios*/
__global__ void cruzaPoblacion(float *poblacion,float *poblacionSiguiente,int indSize,int solucionSize,float Cr,float *randomValues,int *randomIndices,int generaciones){
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int j = threadIdx.y + blockIdx.y * blockDim.y;
	int k,r1,r2,r3;
		
	while(i < POPSIZE){		
		while( j < indSize-solucionSize){
			k = i*indSize+j;
			if( randomValues[(i*(indSize-solucionSize)+j)+generaciones] < Cr){
			/*Mutar*/
				/*Obtenemos la posicion j de los individuos r1,r2,r3*/
			    r1 = randomIndices[3*i]*indSize+j;
				r2 = randomIndices[3*i+1]*indSize+j;
				r3 = randomIndices[3*i+2]*indSize+j;			
				/*TODO:Tomar en cuenta los limites*/
				/*Differential Evolution, Noman2005*/
				poblacionSiguiente[k] = poblacion[r1] + FPARAM*( poblacion[r2] - poblacion[r3] );

				if (abs(poblacionSiguiente[k]) < UMBR) poblacionSiguiente[k] = 0;

			}else{
			/*Normal*/
				poblacionSiguiente[k] = poblacion[k];
			}
			j += blockDim.y * gridDim.y;
		}
		i += blockDim.x * gridDim.x;
	}

}

__global__ void PSOSearch(float *mejorIndividuo, int indSize, float fitness){

	for (int i = 0; i < indSize; ++i) {
		printf("%f, ", mejorIndividuo[i]);
	}
	printf("fitness %f\n\n", fitness);
	
	printf("\n");
	printf("\n");
}

void imprime(float *poblacion, int indSize, int salto, float *fitnesscopia, int *change){
    
	for (int i = 0; i < POPSIZE; ++i)
	{
		for (int j = 0; j < salto; ++j)
		{
			printf("%f, ", poblacion[i*indSize+j]);
		}
		printf("fitness %f, change %d \n\n", fitnesscopia[i], change[i]);
	}
	printf("\n");
	printf("\n************************************************");
}

void imprimeBest(float *mejorIndividuo, int indSize, float fitness, int index){
    
	
	for (int i = 0; i < indSize; ++i) {
		printf("%f, ", mejorIndividuo[i]);
	}
	printf("fitness %f, index %d\n\n", fitness, index);
	
	printf("\n");
	printf("\n");
}

/* 
	Se cambio la funcion con la sintaxis para poder utilizarla con la librería pthreads

	recibe como parametro una estructura tipo ThreadParams que contiene los valores de las variables
	que se utilizarán y el valor de la salida calculada.

	Se agregó a cada funcion del device el parametro del stream en el que se ejecutará.

	Se cambio la copia de memoria para que se haga de manera asincrona.

*/
void *evolucionDiferencial(void *arg){

	int generaciones, best, nDiversidad = 0, randControl = 0;
	float Cr = 0.8f, bestF, mse, rmse;

	ThreadParams *args=(ThreadParams *)arg;

	int serie = args -> serie;
	int corrida = args -> corrida;
	int device = args -> device;
	int control = args -> control;

	float timeC;

	initDeviceControl(device, control);
	cudaEventCreate(&start[corrida]);
	cudaEventCreate(&stop[corrida]);
	cudaEventRecord(start[corrida], 0);

	/*Creo valores aleatorios entre 0 y 1*/
	CURAND_CALL( curandGenerateUniform(randomGenerator[device][control], randomFloatValues_D[device][control], (individuoSize_H-solucionSize_H)*POPSIZE*MAXRAND) );
	inicializaPoblacion<<<malla,bloque>>>(poblacion_D[device][control],individuoSize_H,GSize_H,HSize_H,AlphaSize_H,BetaSize_H,solucionSize_H,randomFloatValues_D[device][control], bestSoFar_D[device][control], numBSF_D[device][control]);
	evaluaPoblacion<<<1,bloque>>>(poblacion_D[device][control],datosReales_D[device][serie],W_D[device][control],K_D[device][control],numeroTiempos,numeroGenes,h,individuoSize_H,fitness_D[device][control], auxiliar[device][control]);				

	float *poblacionActual;
	float *fitnesscopia;
	int *changeInd;

	for(generaciones = 0 ; generaciones < GMAX; generaciones++){
		randControl+=4;
		// Calcula los indices aleatorios para la cruza y mutacion
		calculaIndicesAleatorios<<<1,bloque>>>(randomIndices_D[device][control],rand());
		// Crea la poblacion siguiente
		cruzaPoblacion<<<malla,bloque>>>(poblacion_D[device][control],nuevaPoblacion_D[device][control],individuoSize_H,solucionSize_H,Cr,randomFloatValues_D[device][control],randomIndices_D[device][control], randControl);
		// Evalua la poblacion siguiente

		//cudaStream_t streams[numeroGenes];

		evaluaPoblacion<<<1,bloque>>>(nuevaPoblacion_D[device][control],datosReales_D[device][serie],W_D[device][control],K_D[device][control],numeroTiempos,numeroGenes,h,individuoSize_H,nuevoFitness_D[device][control], auxiliar[device][control]);			
		
		//CUDA_CALL( cudaThreadSynchronize() );
		// Calculo los individuos que deben cambiarse
		comparaPoblacion<<<1,1>>>(fitness_D[device][control],nuevoFitness_D[device][control],mustChange_D[device][control]);
		// Cambio las variables de los individuos que fueron peores, la bandera que indica esto esta en mustChange_D
		seleccionaPoblacion<<<malla,bloque>>>(poblacion_D[device][control],nuevaPoblacion_D[device][control],mustChange_D[device][control],individuoSize_H);		
		// Cambio los valores de aptitud de los individuos que fueron peores, la bandera que indica esto esta en mustChange_D
		seleccionaFitness<<<malla,bloque>>>(fitness_D[device][control],nuevoFitness_D[device][control],mustChange_D[device][control],1);		
		
		// Decrementa el parametro
		// Cr -= (0.002);

		// Se calcula el BFS y el numero de repeticiones del mismo
		//CUDA_CALL( cudaThreadSynchronize() );
		obtenerMetricas<<<1,1>>>(bestSoFar_D[device][control], MSE_D[device][control], RMSE_D[device][control], fitness_D[device][control], numBSF_D[device][control], idBSF[device][control]);
		
		if (generaciones % 50 == 0 && generaciones != 0){
		 	printf("\t\tED - Device %d, Thread %d, Corrida %d, Generacion %d, BestSoFar %f, MSE %f, RMSE %f\n", device, control, corrida, generaciones, *bestSoFar[device][control], *MSE[device][control], *RMSE[device][control]);
		}
		

		if (*numBSF[device][control] >= DIVERSIDAD && DIVERSIDAD > 0 ){ //|| generaciones+1 == GMAX) {
			randControl++;
			// Se calcula la distancia de cada individuo con la poblacion
			calculaDistancia<<<malla,bloque>>>(poblacion_D[device][control],numeroGenes,individuoSize_H,fitness_D[device][control],goodness_D[device][control], auxiliar[device][control], MValor, bestSoFar_D[device][control]);				
			// Obtiene el valor del Goodness minimo para permanecer en la poblacion
			buscaRemplazo<<<1,1>>>(auxiliar[device][control], goodness_D[device][control], minGoodness[device][control]);
			// Crea diversidad en la poblacion reemplazando los individuos con un goodness mayor al calculado previamente
			diversifica<<<malla,bloque>>>(poblacion_D[device][control], goodness_D[device][control],individuoSize_H,GSize_H,HSize_H,AlphaSize_H,BetaSize_H, solucionSize_H,randomFloatValues_D[device][control], minGoodness[device][control], randControl);
			// Evalua la nueva poblacion
			evaluaPoblacion<<<1,bloque>>>(poblacion_D[device][control],datosReales_D[device][serie],W_D[device][control],K_D[device][control],numeroTiempos,numeroGenes,h,individuoSize_H,fitness_D[device][control], auxiliar[device][control]);				
			// Reinicia el contador de BSF para permitirle evolucionar
			*numBSF[device][control] = 0;
			nDiversidad++;
			// Se calcula el BFS y el numero de repeticiones del mismo
			CUDA_CALL( cudaThreadSynchronize() );
			obtenerMetricas<<<1,1>>>(bestSoFar_D[device][control], MSE_D[device][control], RMSE_D[device][control], fitness_D[device][control], numBSF_D[device][control], idBSF[device][control]);
		
			printf("\t\tDiversidad - Device %d, Thread %d, Corrida %d, Generacion %d, BestSoFar %f, MSE %f, RMSE %f\n", device, control, corrida, generaciones, *bestSoFar[device][control], *MSE[device][control], *RMSE[device][control]);
		
		}

		iniSwarm<<<malla,bloque>>>(poblacion_D[device][control], nuevaPoblacion_D[device][control], individuoSize_H, auxiliar[device][control], randomFloatValues_D[device][control]);
		swarmPSO<<<malla,bloque>>>(nuevaPoblacion_D[device][control], (poblacion_D[device][control]+*idBSF[device][control]*individuoSize_H), poblacion_D[device][control], auxiliar[device][control], individuoSize_H, solucionSize_H, randomFloatValues_D[device][control], GSize_H, HSize_H, AlphaSize_H, BetaSize_H);
		evaluaPoblacion<<<1,bloque>>>(nuevaPoblacion_D[device][control], datosReales_D[device][serie], W_D[device][control], K_D[device][control], numeroTiempos, numeroGenes, h, individuoSize_H, nuevoFitness_D[device][control], auxiliar[device][control]);				
		comparaPoblacion<<<1,bloque>>>(fitness_D[device][control],nuevoFitness_D[device][control],mustChange_D[device][control]);
		seleccionaPoblacion<<<malla,bloque>>>(poblacion_D[device][control], nuevaPoblacion_D[device][control], mustChange_D[device][control], individuoSize_H);		
		seleccionaFitness<<<malla,bloque>>>(fitness_D[device][control], nuevoFitness_D[device][control], mustChange_D[device][control], 1);		
		
		if (generaciones % 50 == 0 && generaciones != 0){
			printf("\t\tPSO - Device %d, Thread %d, Corrida %d, Iteracion %d, BestSoFar %f, MSE %f, RMSE %f\n", device, control, corrida, generaciones, *bestSoFar[device][control], *MSE[device][control], *RMSE[device][control]);
		}

		if (randControl == MAXRAND-5){
			CURAND_CALL( curandGenerateUniform(randomGenerator[device][control], randomFloatValues_D[device][control], (individuoSize_H-solucionSize_H)*POPSIZE*MAXRAND) );
			randControl = -4;
			//printf("\t\t*** %d ***\n", generaciones);
		}
		//CUDA_CALL( cudaStreamSynchronize(0) );
	}

	CUDA_CALL( cudaThreadSynchronize() );
	best = *idBSF[device][control];
	bestF = *bestSoFar_D[device][control];
	mse = *MSE_D[device][control];
	rmse = *RMSE_D[device][control];

	printf("\t\tED - Corrida %d, Device %d, Thread %d, bestF %f, MSE %f, RMSE %f, nDiversidad %d\n",corrida ,device ,control, bestF, mse, rmse, nDiversidad);
	

	CUDA_CALL( cudaMemcpy(mejorIndividuo[device][control],(poblacion_D[device][control]+best*individuoSize_H),sizeof(float)*individuoSize_H,cudaMemcpyDeviceToHost) );
	args -> resultado = bestF;


	liberaMemoriaDevice(device, control);
	cudaEventRecord(stop[corrida], 0);
	cudaEventSynchronize(stop[corrida]);
	cudaEventElapsedTime(&timeC, start[corrida], stop[corrida]);

	tiempos[serie][corrida] = timeC;
	pthread_exit(NULL);
}

/* 
	Esta función fue separada para poder llamar los calculos de manera asincrona y luego
	escribirlos en orden

*/
void reportaSalida(int serie, int corrida, int control, int device, float bestF){
	int i;
	/*Reportando la salida*/
	fprintf(fOut,"Mejor solucion de la corrida %d, serie %d \n",corrida+1,serie+1);
	fprintf(fOut,"G = [");
	for(i = 0 ; i < GSize_H ;i++){
		fprintf(fOut,"%f\t",mejorIndividuo[device][control][i]);
	}
	fprintf(fOut,"]\n");
	fprintf(fOut,"H = [");
	for(; i < GSize_H+HSize_H ;i++){
		fprintf(fOut,"%f\t",mejorIndividuo[device][control][i]);
	}
	fprintf(fOut,"]\n");
	fprintf(fOut,"ALPHA = [");
	for(; i < GSize_H+HSize_H+AlphaSize_H ;i++){
		fprintf(fOut,"%f\t",mejorIndividuo[device][control][i]);
	}
	fprintf(fOut,"]\n");
	fprintf(fOut,"BETA = [");
	for(; i < GSize_H+HSize_H+AlphaSize_H+BetaSize_H ;i++){
		fprintf(fOut,"%f\t",mejorIndividuo[device][control][i]);
	}
	fprintf(fOut,"]\n");
	fprintf(fOut,"X = [");
	for(; i < GSize_H+HSize_H+AlphaSize_H+BetaSize_H+solucionSize_H;i++){
		fprintf(fOut,"%f\t",mejorIndividuo[device][control][i]);
	}
	fprintf(fOut,"]\n");	
	fprintf(fOut,"Aptitud %f\n",bestF);
	fprintf(fOut,"Tiempo %f\n",tiempos[serie][corrida]);
	
	printf("\t\tbestF, iter %f, %d\n",bestF,corrida);
}

float max(float *d, int size){
	float f = FLT_MIN;
	for(int i = 0 ; i < size ; i++)
		if( d[i] > f )
			f = d[i];
	
	return f;
}

float min(float *d,int size){
	float f = FLT_MAX;
	for(int i = 0 ; i < size ; i++)
		if( d[i] < f )
			f = d[i];
	
	return f;
}

float promedio(float *d,int size){
	float sum = 0;
	for(int i = 0 ; i < size ; i++)		
		sum += d[i];
	
	return sum/size;
}

float desvEstandar(float *d,int size){
	float dB,sum;
	int i;
	dB = promedio(d,size);
	
	for( i = 0 , sum = 0.0 ; i < size;i++)
		sum += pow(d[i] - dB,2);
	
	sum = sum/size;
	
	return sqrt(sum);
}

/* Obtiene el número de GPUs y algunas caracteristicas de interes para configuracion de la ejecución */
int getGPUDevices(){
	int num_gpus = 0;
	size_t mem_tot_0 = 0;
	size_t mem_free_0 = 0;
	printf("-------------------------------------------------------------------------------------------------------------------------------\n");
	CUDA_CALL( cudaGetDeviceCount(&num_gpus) );
	
	for(int i = 0; i < num_gpus; i++){
		cudaDeviceProp dprop;
		CUDA_CALL( cudaGetDeviceProperties(&dprop, i) );
		cudaMemGetInfo  (&mem_free_0, &mem_tot_0);
		printf("\tid %d: %s, WarpSize %d, Memory free %f MBytes, Total %f MBytes\n", i, dprop.name, dprop.warpSize,(float)mem_free_0/1000000, (float)mem_tot_0/1000000);
	}
	return num_gpus;
}

/*Lee los datos reales de archivo*/	  
void RRG(char *inFile, char *outFile){
	int numGpus = 1;

	if (GPUS == 0){
		numGpus = getGPUDevices();
	}
	else {
		numGpus = GPUS;
	}

	int actualDevice = 0;

	float resultadosCorridas[CORRIDAS];
	fIn = fopen (inFile,"r+");
	fOut = fopen (outFile,"w+");
	
	/*Numero de genes*/
	fscanf(fIn,"%d",&numeroGenes);	

	if (PROF > numeroGenes) {
		printf("-------------------------------------------------------------------------------------------------------------------------------\n");
		printf("\tError: La profundidad excede al numero de genes, (PROF) %d, Numero de genes %d\n", PROF, numeroGenes);
		printf("-------------------------------------------------------------------------------------------------------------------------------\n");
		cudaDeviceReset();
		exit(0);

	}

	/*Series de datos*/
	fscanf(fIn,"%d",&nseries);		
	
	/*
	*Los instantes de tiempo pueden cambiar entre serie y serie, con motivo de la eficiencia se deja afuera.
	*La razon es que todos los archivos tienen los mismos instantes de tiempo
	*/	
	fscanf(fIn,"%d",&numeroTiempos);			
	/*Estos valores quedan perdidos*/
	for(int i = 0; i<numeroTiempos; i++)
		fscanf(fIn,"%f",&h);					
		
	//Pide memoria y demas
	init(numGpus);
	printf("-------------------------------------------------------------------------------------------------------------------------------\n");
	printf("\tEntrada %s   | Salida %s\n",inFile, outFile);
	printf("-------------------------------------------------------------------------------------------------------------------------------\n");
	printf("\tPoblacion  %d | Generaciones %d | Diversidad %d | PDiversidad %f | Umbral %f\n", POPSIZE, GMAX, DIVERSIDAD, REMPDIV, UMBR);
	printf("-------------------------------------------------------------------------------------------------------------------------------\n");
	printf("\tW  %f | C1 %f | C2 %f | Iteraciones PSO %d\n", WPSO, C1, C2, PSOGEN);
	printf("-------------------------------------------------------------------------------------------------------------------------------\n");
	printf("\tCorridas   %d | STREAMS      %d | GPUs       %d | Profundidad %d | Coef Penalización %f\n",CORRIDAS, STREAMS, GPUS, PROF, PCOEF);
	// Initialisa y configura el thread joinable
	pthread_attr_t attr;
	void **status;
	pthread_mutex_init(&mutexsum, NULL);
	pthread_attr_init(&attr);
	pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_JOINABLE);

	
	printf("-------------------------------------------------------------------------------------------------------------------------------\n");

	ThreadParams** args = (struct ThreadParams**) malloc(CORRIDAS * nseries * sizeof(struct ThreadParams*));
	tiempos = (float**) malloc(sizeof(float*)*CORRIDAS*nseries);
	streamEval = (cudaStream_t**) malloc(sizeof(cudaStream_t*) * CORRIDAS * numGpus);
	streamCalc = (cudaStream_t**) malloc(sizeof(cudaStream_t*) * CORRIDAS * numGpus);
	datosReales_D = (float***) malloc(sizeof(float**)*numGpus*nseries);

	for(int i = 0; i<numGpus; i++){
		CUDA_CALL( cudaSetDevice(i) );
		CUDA_CALL( cudaDeviceReset() );
		initDevice(i);
		datosReales_D[i] = (float**) malloc(sizeof(float*)*nseries);
	}
	for(int k = 0 ; k < nseries ; k++){
		printf("\tSerie %d\n",k);
		/*Leyendo los datos de la series*/
		for(int j = 0; j < numeroTiempos; j++)
			for(int i = 0; i < numeroGenes; i++)
				fscanf(fIn,"%f",&datosReales[i*numeroTiempos+j]);
	
		/*Copio los datos de la serie a device*/
		for(int i = 0; i <numGpus; i++){
			datosReales_D[i][k] = (float*) malloc(sizeof(float)*numeroGenes*numeroTiempos);
			CUDA_CALL( cudaSetDevice(i) );
			CUDA_CALL( cudaDeviceReset() );
			CUDA_CALL( cudaMalloc(&datosReales_D[i][k],sizeof(float)*numeroGenes*numeroTiempos) );
			CUDA_CALL( cudaMemcpy(datosReales_D[i][k], datosReales, sizeof(float)*numeroGenes*numeroTiempos,cudaMemcpyHostToDevice) );
		
		}	
		
		start = (cudaEvent_t*) malloc(sizeof(cudaEvent_t) * CORRIDAS);
		stop = (cudaEvent_t*) malloc(sizeof(cudaEvent_t) * CORRIDAS);
		tiempos[k] = (float*) malloc(sizeof(float)*CORRIDAS);

		args[k] = (ThreadParams*) malloc(sizeof(ThreadParams) * CORRIDAS);
		threads = (pthread_t*) malloc(sizeof(pthread_t) * CORRIDAS);
		status = (void**) malloc(sizeof(void*) * CORRIDAS);
		

		int threadControl = 0;
		int hilosControl = 0;
		int corridasControl = 0;

		for( int i = 0 ; i < CORRIDAS ; i++){
			
			args[k][i].serie = k;
			args[k][i].corrida = i;
			args[k][i].device = actualDevice;
			args[k][i].control = threadControl;

			//printf("\t\tCorrida %d, Device %d, Thread %d\n",i ,actualDevice ,threadControl);
			if (pthread_create(&threads[i], &attr, evolucionDiferencial, (void *)&args[k][i] )) exit(-1);
			actualDevice++;
			if (actualDevice >= numGpus){
				actualDevice = 0;
				threadControl++;
				hilosControl++;
			}

			// Espera que la cantidad de hilos lanzados termine para ejecutar los siguientes
			if (hilosControl == STREAMS || i == CORRIDAS-1){
				for( int j = i - corridasControl; j <= i ; j++){
					//printf("\t\tSincronizando %d, de %d\n", j, i);
					pthread_join(threads[j], &status[j]);
				}
				hilosControl = 0;
				corridasControl = 0;
			}
			corridasControl++;
		}
		
		actualDevice = 0;
		threadControl = 0;

		for( int i = 0 ; i < CORRIDAS ; i++){
			printf("\t\tsalida %d, Device %d, Thread %d\n",i ,actualDevice ,threadControl);
			resultadosCorridas[i] = args[k][i].resultado;
			reportaSalida(k, i, threadControl, actualDevice, resultadosCorridas[i]);
			actualDevice++;

			if (actualDevice >= numGpus){
				actualDevice = 0;
				threadControl++;
			}	
		}

		fprintf(fOut,"\n**************************************************");		
		fprintf(fOut,"\nEstadisticas sobre el error cuadratico medio de las corridas\n");
		fprintf(fOut,"Maximo %f\n",max(resultadosCorridas,CORRIDAS));
		fprintf(fOut,"Minimo %f\n",min(resultadosCorridas,CORRIDAS));
		fprintf(fOut,"Promedio %f\n",promedio(resultadosCorridas,CORRIDAS));
		fprintf(fOut,"Desviacion Estandar %f\n",desvEstandar(resultadosCorridas,CORRIDAS));
		fprintf(fOut,"Promedio tiempos %f\n",promedio(tiempos[k],CORRIDAS));
		fprintf(fOut,"**************************************************\n");
		fprintf(fOut,"Archivo %s, Poblacion %d, Generaciones %d, Diversidad %d, GPUs %d\n",inFile, POPSIZE, GMAX, DIVERSIDAD, numGpus);
		fprintf(fOut,"**************************************************\n");

		printf("\n**************************************************");		
		printf("\nEstadisticas sobre el error cuadratico medio de las corridas\n");
		printf("Maximo %f\n",max(resultadosCorridas,CORRIDAS));
		printf("Minimo %f\n",min(resultadosCorridas,CORRIDAS));
		printf("Promedio %f\n",promedio(resultadosCorridas,CORRIDAS));
		printf("Desviacion Estandar %f\n",desvEstandar(resultadosCorridas,CORRIDAS));
		printf("Promedio tiempos %f\n",promedio(tiempos[k],CORRIDAS));
		printf("**************************************************\n");
		printf("Archivo %s, Poblacion %d, Generaciones %d, Diversidad %d, GPUs %d\n",inFile, POPSIZE, GMAX, DIVERSIDAD, numGpus);
		printf("**************************************************\n");
	
		
		/*Estas lecturas quedan sin uso, solo para no afectar el formato del archivo de entrada*/
		if ( k != nseries -1){
			fscanf(fIn,"%d",&numeroTiempos);			
			/*Estos valores quedan perdidos*/
			for(int i = 0; i<numeroTiempos; i++)
				fscanf(fIn,"%f",&h);					
		}
		for(int i = 0; i <numGpus; i++){
			CUDA_CALL( cudaSetDevice(i) );
			CUDA_CALL(cudaFree(datosReales_D[i][k]));
		}
	
	}

	for(int i = 0; i <numGpus; i++){
		CUDA_CALL( cudaSetDevice(i) );
		CUDA_CALL( cudaDeviceReset() );
	}

	liberaMemoria();
	fclose(fIn);
	fclose(fOut);
	pthread_mutex_destroy(&mutexsum);
}

int main(int argc, char** argv){
	RRG(argv[1],argv[2]);	
}
