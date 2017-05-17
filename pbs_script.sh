#!/bin/sh

# Example PBS script for Jinx

# allocate 4 sixcore nodes for up to 5 minutes
#PBS -q class
#PBS -l nodes=4:sixcore
#PBS -l walltime=2:00:00
#PBS -N cse6220-radixsort-schoi332

# change to our project directory
cd $HOME/hw2

# hardcode MPI path
MPIRUN=/usr/lib64/openmpi/bin/mpirun

# set size of the problem instance (must be a multiple of `p`)
N=$(expr 12 \* 1024 \* 1024)

# TODO: experiment with different k

# loop over number of processors (our 4 nodes job can run up to 48)
echo "MyStruct"
for p in 1 2 4 6 12 24 48
do
    for i in 1 2 3 4 5
    do
        $MPIRUN -np $p --hostfile $PBS_NODEFILE --map-by ppr:12:node --map-by ppr:1:core ./sort-mystruct -t -n $N
    done
done

echo "MyStruct Optimized"
for p in 1 2 4 6 12 24 48
do
    for i in 1 2 3 4 5
    do
        $MPIRUN -np $p --hostfile $PBS_NODEFILE --map-by ppr:12:node --map-by ppr:1:core ./sort-mystruct-opt -t -n $N
    done
done
