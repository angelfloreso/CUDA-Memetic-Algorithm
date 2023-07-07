from numpy  import *
import pylab
import matplotlib.pylab
def f(s):
    return ((s**2).sum(axis=1))

nswarm = 5
ndim = 1
max_iter = 10
lb = -10.0
ub = 10.0
W = 0.7
C1 = 1.4
C2 = 1.4

def pso():
    swarm = random.random((nswarm,ndim))*(ub-lb)+lb
    v = random.random((nswarm,ndim))
    fx = f(swarm)
    pbest = swarm.copy()
    fx_pbest = fx.copy()
    index = argmin(fx_pbest)
    fx_gbest = fx_pbest[index]
    gbest = pbest[index]
    i=0
    convergence = []

    while(i < max_iter):

        #print "best", index

        print "pbest", list(pbest)
        #print "swarm", list(swarm)


        v = W*v + C1 *random.random((nswarm,ndim))*(pbest-swarm) +  C2*random.random()*(gbest-swarm)

        #print "suarmp antes", list(swarm)
        #print "v", list(v)

        swarm += v

        #print "suarmp despues", list(swarm)
        
        l =  swarm < lb
        u = swarm > ub



        swarm[l+u] = random.random((nswarm,ndim))[l+u]*(ub-lb)+lb

        fx = f(swarm)
        index = fx < fx_pbest
        fx_pbest[index] = fx[index]
        pbest[index] = swarm[index]
        index = argmin(fx_pbest)
        fx_gbest = fx_pbest[index]
        gbest = pbest[index]
        convergence.append(fx_gbest)

        #print "suarmp salida", list(swarm)

        print "suarmp salida", list(pbest)

        print "**********************************************++"
        i += 1
    return [fx_gbest, gbest, convergence]

[fx_gbest, gbest, convergence] = pso()
print fx_gbest, gbest
pylab.scatter(arange(max_iter),array(convergence))
pylab.show()
