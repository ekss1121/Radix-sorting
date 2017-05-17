# Makefile for HPC 6220 Programming Assignment 3
CXX=mpic++
CCFLAGS=-Wall -g 
# activate for compiler optimizations:
#CCFLAGS=-Wall -O3
LDFLAGS=

# set up google test
GTEST_DIR = ./gtest
CCFLAGS += -I. #-I$(GTEST_DIR)

all: sort-mystruct sort-mystruct-opt tests

test: tests
	for p in 3 4 5; do \
	echo "### TESTING WITH $$p PROCESSES ###"; mpirun -np $$p ./radix-tests ;\
	done

tests: radix-tests

sort-mystruct: main.o mystruct.o
	$(CXX) $(LDFLAGS) -o $@ $^

sort-mystruct-opt: main-opt.o mystruct_opt.o
	$(CXX) $(LDFLAGS) -o $@ $^

main-opt.o: main.cpp
	$(CXX) $(CCFLAGS) -DUSE_MYSTRUCT_OPT -c main.cpp -o main-opt.o

radix-tests: mpi_tests.o mystruct.o mpi_gtest.o gtest-all.o
	$(CXX) $(LDFLAGS) -o $@ $^

gtest-all.o : $(GTEST_DIR)/gtest-all.cc $(GTEST_DIR)/gtest.h
	$(CXX) $(CCFLAGS) -c $(GTEST_DIR)/gtest-all.cc

%.o: %.cpp %.h
	$(CXX) $(CCFLAGS) -c $<

%.o: %.cpp
	$(CXX) $(CCFLAGS) -c $<

clean:
	rm -f *.o radix-sort radix-tests sort-mystruct sort-mystruct-opt
