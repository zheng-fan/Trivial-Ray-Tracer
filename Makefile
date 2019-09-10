default: simple

simple:
	icpc -O3 -std=c++11 -I. -o proj.o -c main.cpp
	icpc -O3 -std=c++11 -L./FreeImage -o proj proj.o -lfreeimage

runsimple:
	./proj

cluster:
	mpiicpc -std=c++11 -O3 -I. -o proj.o -DMPIRUN -DSAMPLE -DMULTILIGHT -c main.cpp
	#mpiicpc -O3 -I. -o proj.o -DMPIRUN -c main.cpp
    # -DMonteCarloSample
    # -DFZGLM
	mpiicpc -std=c++11 -O3 -L./FreeImage -o proj proj.o -lfreeimage

runcluster:
	qsub < proj.qsub
