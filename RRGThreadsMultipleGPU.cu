/* AF 040516

Compilar con las banderas

nvcc --default-stream per-thread -lcurand -O2 -o test RRGThreadsMultipleGPU.cu

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

#define POPSIZE 128
#define GMAX	1000
#define F       0.04f

#define HILOS_X 16
#define HILOS_Y 16
#define BLOQUES_X 16
#define BLOQUES_Y 16

#define MAXGij  3.0
#define MINGij -3.0
#define MAXHij  3.0
#define MINHij -3.0
#define MAXALPHA 20.0
#define MINALPHA 0.0
#define MAXBETA 20.0
#define MINBETA 0.0

#define CORRIDAS 20

/* Define el número maximo de procesos que se pueden ejecutar al mismo tiempo en GPU */
#define STREAMS 1

/* Define el número maximo de GPUs que se pueden utilizar (0 = Todos los GPUs disponibles) */
#define GPUS 0

/* Estructura creada para pasar parametros a la funcion que se ejecuta con pthreads */
typedef struct ThreadParams {
	int serie;
	int corrida;
	int device;
	int control;
	float resultado;
} ThreadParams;

/*
	Se cambiaron todas las variables globales en arreglos de diferentes dimenciones
	para diferenciar las ejecuciones en diferentes device, hilos o corrida.
 */

pthread_t *threads;
ThreadParams **args;
cudaStream_t **streamEval;
cudaStream_t **streamCalc;
cudaEvent_t *start;
cudaEvent_t *stop;
curandGenerator_t **randomGenerator;

FILE *fIn;										/*Archivo de entrada*/
FILE *fOut;										/*Archivo de salida*/
int numeroGenes,nseries,numeroTiempos;
int GSize_H,HSize_H,AlphaSize_H,BetaSize_H,individuoSize_H,solucionSize_H;

/* h denota el tamaño de paso para el metodo Runge - Kutta*/
float h; 				

/*Datos reales obtenidos del experimento*/
float ***datosReales_D;

float *datosReales;
float ***mejorIndividuo;
float ***fitness_H;
float ***randomFloatValues_D;
float ***poblacion_D;
float ***fitness_D;
float ***nuevaPoblacion_D;
float ***nuevoFitness_D;
int ***randomIndices_D;
int ***mustChange_D;
float ***W_D;
float ***K_D;
float **tiempos;

dim3 bloque(HILOS_X,HILOS_Y);
dim3 malla(BLOQUES_X,BLOQUES_Y);

/*MACRO para corroborar si una llamada a CURAND fue exitosa*/
#define CURAND_CALL(x) if((x)!=CURAND_STATUS_SUCCESS) { \
    printf("CURAND - Error en %s:%d\n",__FILE__,__LINE__);\
    }

/*MACRO para corroborar si una llamada CUDA fue exitosa*/
#define CUDA_CALL(x) if((x) != cudaSuccess) { \
      printf("CUDA - Error en %s:%d\n",__FILE__,__LINE__);     \
      }
 
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


/*al parecer cada instante de tiempo es independiente del siguiente o anterior, solo esta limitado por el tiempo original*/
__device__ void rungeKutta(float *G, float * H, float *alph, float * beta, float *X, float *B, float *w, float * k, int numeroTiempos, int numeroGenes, float h){
	int i,j;
	
	for(i = 0; i < numeroGenes; i++)
		X[numeroTiempos*i]=B[numeroTiempos*i];
	
	for(i = 0; i < numeroTiempos-1; i++){
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

}
 
 /*Evalua a los miembros de poblacion, el indice global del thread toma el apuntador al individuo correspondiente*/
__global__ void evaluaPoblacion(float *poblacion,float *realData, float *W, float *K, int numeroTiempos,int numeroGenes,float h,int indSize,float *aptitud){
	int i = threadIdx.x + (blockIdx.x * blockDim.x);	
	int j;

	float *individuo,*G_i,*H_i,*A_i,*B_i,*X_i,sum;

	/*__shared__ float *realData;

	realData = realDataD;*/


	while(i < POPSIZE){			
		individuo = poblacion+i*indSize; //i*indSize es el offset hasta el siguiente individuo
		G_i = individuo;
		H_i = individuo + numeroGenes*numeroGenes;
		A_i = individuo + 2*numeroGenes*numeroGenes;
		B_i = individuo + 2*numeroGenes*numeroGenes + numeroGenes;
		X_i = individuo + 2*numeroGenes*numeroGenes + 2*numeroGenes;				
		/*Resolvemos el sistema de ecuaciones diferenciales para obtener las predicciones X_i*/
		rungeKutta(G_i,H_i,A_i,B_i,X_i,realData,W,K,numeroTiempos,numeroGenes,h);		
		/*Error cuadratico medio*/
		for(j = 0,sum = 0; j < numeroGenes*numeroTiempos; j++)			
				sum += powf((X_i[j]-realData[j])/(realData[j]),2);		
		if(isinf(sum))
			sum = FLT_MAX;		
		aptitud[i] = sum;
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
	CUDA_CALL(cudaFree(nuevaPoblacion_D[device][control]));
	CUDA_CALL(cudaFree(nuevoFitness_D[device][control]));
	CUDA_CALL(cudaFree(randomIndices_D[device][control]));
	CUDA_CALL(cudaFree(randomFloatValues_D[device][control]));
	CUDA_CALL(cudaFree(mustChange_D[device][control]));
	CUDA_CALL(cudaFree(W_D[device][control]));
	CUDA_CALL(cudaFree(K_D[device][control]));
	curandDestroyGenerator(randomGenerator[device][control]);
	CUDA_CALL( cudaStreamDestroy(streamEval[device][control]) );
	CUDA_CALL( cudaStreamDestroy(streamCalc[device][control]) );
	free(fitness_H[device][control]);
} 

/*Inicializa la poblacion. Dado que todos los parametros estan codificados secuencialmente en el arreglo, recibe los tamanos de los parametros. Podria recibir los limites en su lugar. */
__global__ void inicializaPoblacion(float *poblacion,int indSize, int GSize, int HSize, int aSize, int bSize, int solucionSize,float *randomValues){    
    int i = threadIdx.x + blockIdx.x * blockDim.x;
	int j = threadIdx.y + blockIdx.y * blockDim.y;
	int k;
	float upperBound,lowerBound;

	while(i < POPSIZE){		
		while( j < indSize){
			k = i*indSize+j;
			/*Limites para cada parametro, el ultimo caso (la solucion al sistema de ecuaciones diferenciales) es inicializacion en 0*/
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
			}else{
				upperBound = lowerBound = 0;
			}
			poblacion[k] = lowerBound + randomValues[k]*(upperBound-lowerBound);						
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
	individuoSize_H = GSize_H+HSize_H+AlphaSize_H+BetaSize_H+solucionSize_H;
	datosReales = (float*) malloc(sizeof(float)*numeroGenes*numeroTiempos);
	poblacion_D = (float***) malloc(CORRIDAS * nseries * sizeof(float**));
	fitness_D = (float***) malloc(CORRIDAS * nseries * sizeof(float**));
	nuevaPoblacion_D = (float***) malloc(CORRIDAS * nseries * sizeof(float**));
	nuevoFitness_D = (float***) malloc(CORRIDAS * nseries * sizeof(float**));
	randomIndices_D = (int***) malloc(CORRIDAS * nseries * sizeof(int**));
	randomFloatValues_D = (float***) malloc(CORRIDAS * nseries * sizeof(float**));
	fitness_H = (float***) malloc(CORRIDAS * nseries * sizeof(float**));
	mejorIndividuo = (float***) malloc(CORRIDAS * nseries * sizeof(float**));
	mustChange_D = (int***) malloc(CORRIDAS * nseries * sizeof(int**));
	randomGenerator = (curandGenerator_t**) malloc(numGpus*CORRIDAS * nseries * sizeof(curandGenerator_t*));
	W_D = (float***) malloc(CORRIDAS * nseries * sizeof(float**));
	K_D = (float***) malloc(CORRIDAS * nseries * sizeof(float**));
}

/* Inicializa los arreglos que se utilizarán en determinado GPU*/
void initDevice(int deviceId){
	poblacion_D[deviceId] = (float**) malloc(individuoSize_H*POPSIZE*sizeof(float*)*CORRIDAS);
	fitness_D[deviceId] = (float**) malloc(sizeof(float*)*POPSIZE*CORRIDAS);
	nuevaPoblacion_D[deviceId] = (float**) malloc(individuoSize_H*POPSIZE*sizeof(float*)*CORRIDAS);
	nuevoFitness_D[deviceId] = (float**) malloc(sizeof(float*)*POPSIZE*CORRIDAS);
	randomIndices_D[deviceId] = (int**) malloc(sizeof(int*)*POPSIZE*3*CORRIDAS);
	randomFloatValues_D[deviceId] = (float**) malloc(individuoSize_H*POPSIZE*sizeof(float*)*CORRIDAS);
	fitness_H[deviceId] = (float**) malloc(POPSIZE*sizeof(float*)*CORRIDAS);
	mustChange_D[deviceId] = (int**) malloc(POPSIZE*sizeof(int*)*CORRIDAS);
	W_D[deviceId] = (float**) malloc(sizeof(float*)*numeroGenes*POPSIZE*CORRIDAS);
	K_D[deviceId] = (float**) malloc(sizeof(float*)*4*numeroGenes*POPSIZE*CORRIDAS);
	mejorIndividuo[deviceId] = (float**) malloc(individuoSize_H*sizeof(float*)*CORRIDAS);
	randomGenerator[deviceId] = (curandGenerator_t*) malloc(CORRIDAS * nseries * sizeof(curandGenerator_t));
	streamEval[deviceId] = (cudaStream_t*) malloc(sizeof(cudaStream_t) * CORRIDAS);
	streamCalc[deviceId] = (cudaStream_t*) malloc(sizeof(cudaStream_t) * CORRIDAS);
}

/* Inicializa los arreglos y crea los Streams y el generador de numeros aleatorios en el device correspondiente */
void initDeviceControl(int deviceId, int control){
	mejorIndividuo[deviceId][control] = (float*) malloc(individuoSize_H*sizeof(float));
	poblacion_D[deviceId][control] = (float*) malloc(individuoSize_H*POPSIZE*sizeof(float));
	fitness_H[deviceId][control] = (float*) malloc(POPSIZE*sizeof(float));
	randomFloatValues_D[deviceId][control] = (float*) malloc(individuoSize_H*POPSIZE*sizeof(float));
	fitness_D[deviceId][control] = (float*) malloc(sizeof(float)*POPSIZE);
	nuevaPoblacion_D[deviceId][control] = (float*) malloc(individuoSize_H*POPSIZE*sizeof(float));
	nuevoFitness_D[deviceId][control] = (float*) malloc(sizeof(float)*POPSIZE);
	mustChange_D[deviceId][control] = (int*) malloc(POPSIZE*sizeof(int));
	randomIndices_D[deviceId][control] = (int*) malloc(sizeof(int)*POPSIZE*3);
	W_D[deviceId][control] = (float*) malloc(sizeof(float)*numeroGenes*POPSIZE);
	K_D[deviceId][control] = (float*) malloc(sizeof(float)*4*numeroGenes*POPSIZE);
	CUDA_CALL( cudaSetDevice(deviceId) );
	CUDA_CALL(cudaMalloc(&poblacion_D[deviceId][control],individuoSize_H*POPSIZE*sizeof(float)));
	CUDA_CALL(cudaMalloc((void **)&randomFloatValues_D[deviceId][control],individuoSize_H*POPSIZE*sizeof(float)) );
	CUDA_CALL(cudaMalloc(&fitness_D[deviceId][control],sizeof(float)*POPSIZE));			
	CUDA_CALL(cudaMalloc(&nuevaPoblacion_D[deviceId][control],individuoSize_H*POPSIZE*sizeof(float)));
	CUDA_CALL(cudaMalloc(&nuevoFitness_D[deviceId][control],sizeof(float)*POPSIZE));			
	CUDA_CALL(cudaMalloc(&randomIndices_D[deviceId][control] , 3*POPSIZE*sizeof(int) ) );
	CUDA_CALL(cudaMalloc(&mustChange_D[deviceId][control] , POPSIZE*sizeof(int) ) );
	CUDA_CALL(cudaMalloc(&W_D[deviceId][control], sizeof(float)*numeroGenes*POPSIZE));
	CUDA_CALL(cudaMalloc(&K_D[deviceId][control], sizeof(float)*4*numeroGenes*POPSIZE));
    srand(time(NULL));
	/*Estableciendo la semilla aleatoria para CURAND*/
    CURAND_CALL(curandCreateGenerator(&randomGenerator[deviceId][control],CURAND_RNG_PSEUDO_DEFAULT));
    CURAND_CALL(curandSetPseudoRandomGeneratorSeed(randomGenerator[deviceId][control], rand()));
	/*Pidiendo memoria en device*/
	/*Para resolver la ecuacion diferencial*/
	CUDA_CALL( cudaStreamCreate(&streamEval[deviceId][control]) );
	CUDA_CALL( cudaStreamCreateWithFlags(&streamCalc[deviceId][control],cudaStreamNonBlocking) );
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
__global__ void seleccionaPoblacion(float *original,float *nuevo,int *mustChange,int indSize){
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
__global__ void cruzaPoblacion(float *poblacion,float *poblacionSiguiente,int indSize,float Cr,float *randomValues,int *randomIndices){
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int j = threadIdx.y + blockIdx.y * blockDim.y;
	int k,r1,r2,r3;
		
	while(i < POPSIZE){		
		while( j < indSize){
			k = i*indSize+j;
			if( randomValues[k] < Cr){
			/*Mutar*/
				/*Obtenemos la posicion j de los individuos r1,r2,r3*/
			    r1 = randomIndices[3*i]*indSize+j;
				r2 = randomIndices[3*i+1]*indSize+j;
				r3 = randomIndices[3*i+2]*indSize+j;			
				/*TODO:Tomar en cuenta los limites*/
				poblacionSiguiente[k] = poblacion[r1] + F*( poblacion[r2] - poblacion[r3] );
			}else{
			/*Normal*/
				poblacionSiguiente[k] = poblacion[k];
			}
			j += blockDim.y * gridDim.y;
		}
		i += blockDim.x * gridDim.x;
	}

}

/* 
	Se cambio la funcion con la sintaxis para poder utilizarla con la librería pthreads

	recibe como parametro una estructura tipo ThreadParams que contiene los valores de las variables
	que se utilizarán y el valor de la salida calculada.

	Se agregó a cada funcion del device el parametro del stream en el que se ejecutará.

	Se cambio la copia de memoria para que se haga de manera asincrona.

*/
void *evolucionDiferencial(void *arg){

	int generaciones,best,i;
	float Cr = 0.8f,bestF;

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
	CURAND_CALL( curandSetStream (randomGenerator[device][control], streamCalc[device][control]) );
	CURAND_CALL( curandGenerateUniform(randomGenerator[device][control], randomFloatValues_D[device][control], individuoSize_H*POPSIZE) );
	inicializaPoblacion<<<malla,bloque, 0, streamCalc[device][control]>>>(poblacion_D[device][control],individuoSize_H,GSize_H,HSize_H,AlphaSize_H,BetaSize_H,solucionSize_H,randomFloatValues_D[device][control]);
	evaluaPoblacion<<<1,bloque, 0, streamCalc[device][control]>>>(poblacion_D[device][control],datosReales_D[device][serie],W_D[device][control],K_D[device][control],numeroTiempos,numeroGenes,h,individuoSize_H,fitness_D[device][control]);				

	for(generaciones = 0 ; generaciones < GMAX; generaciones++){
		//Calcula los indices aleatorios para la cruza y mutacion
		calculaIndicesAleatorios<<<1,bloque, 0, streamCalc[device][control]>>>(randomIndices_D[device][control],rand());
		//Probabilidades de mutacion para cada variable de cada individuo
		CURAND_CALL( curandSetStream (randomGenerator[device][control], streamCalc[device][control]) );
		CURAND_CALL( curandGenerateUniform(randomGenerator[device][control], randomFloatValues_D[device][control], individuoSize_H*POPSIZE) );
		//Crea la poblacion siguiente
		cruzaPoblacion<<<malla,bloque, 0, streamCalc[device][control]>>>(poblacion_D[device][control],nuevaPoblacion_D[device][control],individuoSize_H,Cr,randomFloatValues_D[device][control],randomIndices_D[device][control]);
		//Evalua la poblacion siguiente		
		evaluaPoblacion<<<1,bloque, 0, streamCalc[device][control]>>>(nuevaPoblacion_D[device][control],datosReales_D[device][serie],W_D[device][control],K_D[device][control],numeroTiempos,numeroGenes,h,individuoSize_H,nuevoFitness_D[device][control]);			
		//Calculo los individuos que deben cambiarse
		comparaPoblacion<<<1,bloque, 0, streamCalc[device][control]>>>(fitness_D[device][control],nuevoFitness_D[device][control],mustChange_D[device][control]);
		//Cambio las variables de los individuos que fueron peores, la bandera que indica esto esta en mustChange_D
		seleccionaPoblacion<<<malla,bloque, 0, streamCalc[device][control]>>>(poblacion_D[device][control],nuevaPoblacion_D[device][control],mustChange_D[device][control],individuoSize_H);		
		//Cambio los valores de aptitud de los individuos que fueron peores, la bandera que indica esto esta en mustChange_D
		seleccionaPoblacion<<<malla,bloque, 0, streamCalc[device][control]>>>(fitness_D[device][control],nuevoFitness_D[device][control],mustChange_D[device][control],1);		
		//Decrementa el parametro
		//Cr -= (0.002);
	}
	/*Saco las aptitudes finales*/
	CUDA_CALL( cudaMemcpyAsync(fitness_H[device][control],fitness_D[device][control],sizeof(float)*POPSIZE,cudaMemcpyDeviceToHost, streamCalc[device][control]) );
	/*Secuencialmente busco el mejor individuo*/
	for(i = 0,best = -1,bestF = FLT_MAX ; i < POPSIZE;i++){
		if( fitness_H[device][control][i] < bestF ){
			bestF = fitness_H[device][control][i];
			best = i;
		}
	}
	CUDA_CALL( cudaMemcpyAsync(mejorIndividuo[device][control],(poblacion_D[device][control]+best*individuoSize_H),sizeof(float)*individuoSize_H,cudaMemcpyDeviceToHost, streamCalc[device][control]) );
	args -> resultado = bestF;
	liberaMemoriaDevice(device, control);

	cudaEventRecord(stop[corrida], 0);
	//cudaEventSynchronize(stop[corrida]);
	cudaEventElapsedTime(&timeC, start[corrida], stop[corrida]);

	tiempos[serie][corrida] = timeC;

	printf("\t\t\tbestF,device,control %f, %d, %d\n",bestF,device,control);

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
	printf("---------------------------\n");
	CUDA_CALL( cudaGetDeviceCount(&num_gpus) );
	
	for(int i = 0; i < num_gpus; i++){
		cudaDeviceProp dprop;
		CUDA_CALL( cudaGetDeviceProperties(&dprop, i) );
		cudaMemGetInfo  (&mem_free_0, &mem_tot_0);
		printf("   %d: name %s, warpSize %d, Memory free %f MBytes, total %f MBytes\n", i, dprop.name, dprop.warpSize,(float)mem_free_0/1000000, (float)mem_tot_0/1000000);
	}
	printf("---------------------------\n");
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

	printf("CUDA devices:\t%d\n", numGpus);

	float resultadosCorridas[CORRIDAS];
	fIn = fopen (inFile,"r+");
	fOut = fopen (outFile,"w+");
	
	/*Numero de genes*/
	fscanf(fIn,"%d",&numeroGenes);	
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

	printf("Archivo%s\n",inFile);

	// Initialisa y configura el thread joinable
	pthread_attr_t attr;
	void **status;
	pthread_attr_init(&attr);
	pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_JOINABLE);

	
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

			printf("\t\tCorrida %d, Device %d, threadControl %d\n",i ,actualDevice ,threadControl);
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
					printf("\t\tSincronizando %d, de %d\n", j, i);
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
			printf("\t\tsalida %d, Device %d, threadControl %d\n",i ,actualDevice ,threadControl);
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
}

int main(int argc, char** argv){
	RRG(argv[1],argv[2]);	
}
