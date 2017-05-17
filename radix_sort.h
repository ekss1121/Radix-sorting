/**
 * @file    radix_sort.h
 * @author  Patrick Flick <patrick.flick@gmail.com>
 *
 * Copyright (c) 2016 Georgia Institute of Technology. All Rights Reserved.
 */

/*
 * TODO: implement your radix sort solution in this file
 */

#include <mpi.h>
#include <vector>
#include <new>
#include <iostream>
// returns the value of the digit starting at offset `offset` and containing `k` bits
#define GET_DIGIT(key, k, offset) ((key) >> (offset)) & ((1 << (k)) - 1)


/**
 * @brief   Parallel distributed radix sort.
 *
 * This function sorts the distributed input range [begin, end)
 * via lowest-significant-byte-first radix sort.
 *
 * This function will sort elements of type `T`, by the key of type `unsigned int`
 * which is returned by the key-access function `key_func`.
 *
 * The MPI datatype for the templated (general) type `T` has to be passed
 * via the `dt` parameter.
 *
 * @param begin         A pointer to the first element in the local range to be sorted.
 * @param end           A pointer to the end of the range to be sorted. This
 *                      pointer points one past the last element, such that the
 *                      total number of elements is given by `end - begin`.
 * @param key_func      A function with signature: `unsigned int (const T&)`.
 *                      This function returns the key of each element, which is
 *                      used for sorting.
 * @param dt            The MPI_Datatype which represents the type `T`. This
 *                      is used whenever elements of type `T` are communicated
 *                      via MPI.
 * @param comm          The communicator on which the sorting happens.
 *                      NOTE: this is not necessarily MPI_COMM_WORLD. Call
 *                            all MPI functions with this communicator and
 *                            NOT with MPI_COMM_WORLD.
 */
template <typename T>
void radix_sort(T* begin, T* end, unsigned int (*key_func)(const T&), MPI_Datatype dt, MPI_Comm comm, unsigned int k = 16) {
    // get comm rank and size
    int rank, p;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &p);
    // The number of elements per processor: n/p
    size_t np = end - begin;
    // Allocate memmorry for sorted array
    std::vector<T> sorted(np);
    // Allocate memmory for unsorted array
    std::vector<T> unsorted(np);
    // the number of histogram buckets = 2^k
    unsigned int num_buckets = 1 << k;
    std::vector< int> hisgm(num_buckets);
    for (unsigned int d = 0; d < 8*sizeof(unsigned int); d += k) {
    	// 1.) create histogram and sort via bucketing (~ counting sort)
	for (size_t i =0; i<np; i++) unsorted[i] = *(begin+i);
	// Clear hisgm for every digit
	for (unsigned int i=0; i<num_buckets; i++) hisgm[i] = 0;
	// Calculate the histogram of key frequencies.
	for (size_t i = 0; i < np; i++){
	    hisgm[GET_DIGIT(key_func(unsorted[i]), k, d)] += 1;
	}
	// Calculate the starting index of each key(local prefix sum)
	std::vector<int> prefix_sum(num_buckets);
	std::vector<int> prefix_sum_local(num_buckets);
	prefix_sum[0] = 0;
	prefix_sum_local[0] =0;
	for (unsigned int i = 1; i < num_buckets; i++){
	    prefix_sum[i] = prefix_sum[i-1] + hisgm[i-1];
	    prefix_sum_local[i] = prefix_sum[i];
	}
	// Copy to sorted array
	for ( size_t i = 0; i<np; i++){
	    sorted[prefix_sum_local[GET_DIGIT(key_func(unsorted[i]), k, d)]] =unsorted[i];
	    prefix_sum_local[GET_DIGIT(key_func(unsorted[i]), k, d)] += 1;
	}
	//+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
	// 2.) get global histograms (P, G) via MPI_Exscan/MPI_Allreduce,...
	// declare P, G and L
	//unsigned int p_local[np], p_global[np];
	std::vector< int> p_global(np,0);
	//unsigned int g_local[np], g_global[np];
	std::vector< int> g_global(np,0);
	//unsigned int l_local[np];
	std::vector< int> l_local(np,0);
	// Calculate global histogram and prefixsum.
	std::vector< int> hisgm_global(num_buckets);
	std::vector< int> prefix_sum_global(num_buckets);
	MPI_Exscan(&hisgm[0], &hisgm_global[0], num_buckets, MPI_INT, MPI_SUM, comm);
	MPI_Allreduce(&prefix_sum[0], &prefix_sum_global[0], num_buckets, MPI_INT, MPI_SUM, comm);
	// Calculate P, G and L
	for ( size_t i = 0; i < np; i++){
	    unsigned int cur_dgt = GET_DIGIT(key_func(sorted[i]), k, d);
	    p_global[i] = hisgm_global[cur_dgt];
	    g_global[i] = prefix_sum_global[cur_dgt];
	    for (size_t j = 0; j < i; j++){
		if ((GET_DIGIT(key_func(sorted[j]), k, d) )== cur_dgt) l_local[i] ++;
	    }
	}
	// 3.) calculate send_counts
	std::vector< int> send_counts(p, 0);
	for (size_t i =0; i< np; i++) {
	    send_counts[(p_global[i] + g_global[i] +l_local[i])/np] ++;
	}
	// 4.) communicate send_counts to get recv_counts
	std::vector< int> recv_counts(p, 0);
	MPI_Alltoall(&send_counts[0], 1, MPI_INT, &recv_counts[0], 1, MPI_INT, comm);
	// 4.) calculate displacements
	std::vector< int> sdispls(p);
	std::vector< int> rdispls(p);
	sdispls[0] = 0;
	rdispls[0] = 0;
	for (size_t i=1; i< p; i++) {
	    // Calculate send_displs
	    sdispls[i] = sdispls[i-1] + send_counts[i-1];
	    // Calculate recev_displs
	    rdispls[i] = rdispls[i-1] + recv_counts[i-1];
	}
    	// 6.) MPI_Alltoallv
	MPI_Alltoallv(&sorted[0], &send_counts[0], &sdispls[0], dt,
		      &unsorted[0], &recv_counts[0], &rdispls[0], dt, comm);
	
	// 7.) local sorting via bucketing (~ counting sort)
	// Clear hisgm for local sorting
	for (unsigned int i=0; i<num_buckets; i++) hisgm[i] = 0;
	for (size_t i = 0; i < np; i++){
	    hisgm[GET_DIGIT(key_func(unsorted[i]), k, d)] += 1;
	}
	// Calculate the starting index of each key(local prefix sum)
	prefix_sum[0] = 0;
	for (unsigned int i = 1; i < num_buckets; i++){
	    prefix_sum[i] = prefix_sum[i-1] + hisgm[i-1];
	}
	// Copy to sorted array
	for ( size_t i = 0; i< np; i++){
	    sorted[prefix_sum[GET_DIGIT(key_func(unsorted[i]), k, d)]] = unsorted[i];
	    prefix_sum[GET_DIGIT(key_func(unsorted[i]), k, d)] += 1;
	}
    // copy to input array
    for (size_t i=0;i<np;i++) *(begin+i) = sorted[i];
    }
    
}
