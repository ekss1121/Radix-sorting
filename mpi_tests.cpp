/**
 * @file    mpi_tests.cpp
 * @author  Patrick Flick <patrick.flick@gmail.com>
 * @brief   GTest Unit Tests for the parallel MPI code.
 *
 * Copyright (c) 2014 Georgia Institute of Technology. All Rights Reserved.
 */
/*
 * Add your own test cases here. We will test your final submission using
 * a more extensive tests suite. Make sure your code works for many different
 * input cases.
 *
 * Note:
 * The google test framework is configured, such that
 * only errors from the processor with rank = 0 are shown.
 */

#include <mpi.h>
#include <gtest/gtest.h>

#include <vector>
#include <algorithm>

#include "radix_sort.h"
#include "utils.h"
#include "mystruct.h"

/*********************************************************************
 *                   Add your own test cases here.                   *
 *********************************************************************/
// Other test cases can include:
// - all elements are equal
// - elements are randomly picked
// - elements are already sorted (or sorted inversely)


/* testing radix sorting of unsigned integers */
unsigned int nokey_func(const unsigned int& x) {
    return x;
}

template <typename T>
void test_sort_global(std::vector<T>& global_in, unsigned int (*key_func)(const T&), MPI_Datatype dt, unsigned int k) {
    int rank, p;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &p);

    ASSERT_TRUE(global_in.size() % p == 0) << "input size has to be a mutliple of p";

    // scatter data from rank 0 to all processors
    std::vector<T> local_x = scatter(global_in, dt, MPI_COMM_WORLD);

    // radix sort the input
    radix_sort(&local_x[0], &local_x[0]+local_x.size(), key_func, dt, MPI_COMM_WORLD, k);

    // gather sorted datat back for verification
    std::vector<T> global_result = gather(local_x, dt, MPI_COMM_WORLD);

    // verify results on root process
    if (rank == 0) {
        std::stable_sort(global_in.begin(), global_in.end());
        EXPECT_EQ(global_in, global_result);
    }
}


void test_rand_ints(int n) {
    std::vector<unsigned int> x;
    int rank, p;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &p);
    if (rank == 0) {
        // sort with 10 elements per process
        x.resize(n*p);
        for (size_t i = 0; i < x.size(); ++i)
            x[i] = rand() % n;
    }
    test_sort_global(x, &nokey_func, MPI_UNSIGNED, 8);
}

TEST(MpiTest, Sort10rand) {
    test_rand_ints(10);
}

TEST(MpiTest, Sort1000rand) {
    test_rand_ints(1000);
}

void test_rand_mystruct(int n) {
    std::vector<MyStruct> x;
    int rank, p;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &p);
    if (rank == 0) {
        // sort with 10 elements per process
        x.resize(n*p);
        std::generate(x.begin(), x.end(), mystruct_rand);
    }
    MPI_Datatype dt = mystruct_get_mpi_type();
    test_sort_global(x, &mystruct_key_access, dt, 8);
}

TEST(MpiTest, SortMyStruct100rand) {
    test_rand_mystruct(100);
}

TEST(MpiTest, SortMyStruct100000) {
    test_rand_mystruct(100000);
}

// equal struct
void test_equal_mystruct(int n){
    std::vector<MyStruct> x;
    int rank, p;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &p);
    if (rank == 0){
        x.resize(n*p);
        std::generate(x.begin(), x.end(), mystruct_rand);
        for (size_t i =0; i< x.size(); i++)
            x[i].key = n;
    }
    MPI_Datatype dt = mystruct_get_mpi_type();
    test_sort_global(x, &mystruct_key_access, dt, 8);
}

TEST(MpiTest, SortMystruct1000equal){
    test_equal_mystruct(1000);
}
// Reversed struct
void test_revs_mystruct(int n){
    std::vector<MyStruct> x;
    int rank, p;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &p);
    if (rank == 0){
        x.resize(n*p);
        std::generate(x.begin(), x.end(), mystruct_rand);
        for (size_t i =0; i< x.size(); i++)
            x[i].key = n*p - i;
    }
    MPI_Datatype dt = mystruct_get_mpi_type();
    test_sort_global(x, &mystruct_key_access, dt, 8);
}

TEST(MpiTest, SortMystruct1000revs){
    test_revs_mystruct(1000);
}

// equal elements test cases
void test_equal_ints(int n) {
    std::vector<unsigned int> x;
    int rank, p;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &p);
    if (rank == 0){
        // sort with n elements per processor
        x.resize(n*p);
        for (size_t i =0; i< x.size(); ++i)
            x[i] = n;

    }
    test_sort_global(x, &nokey_func, MPI_UNSIGNED, 8);
}

TEST(MpiTest, Sort1000euqal){
    test_equal_ints(1000);
}

// Reversed elements test cases
void test_revs_ints(int n) {
    std::vector<unsigned int> x;
    int rank, p;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &p);
    if (rank == 0){
        // sort with n elements per processor
        x.resize(n*p);
        for (size_t i =0; i< x.size(); ++i)
            x[i] = n*p - i;

    }
    test_sort_global(x, &nokey_func, MPI_UNSIGNED, 8);
}

TEST(MpiTest, Sort1000revc){
    test_revs_ints(1000);
}