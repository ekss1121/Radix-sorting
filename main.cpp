/**
 * @file    main.cpp
 * @author  Patrick Flick <patrick.flick@gmail.com>
 * @brief   Implements functionality for benchmarking your sorting implementations.
 *
 * Copyright (c) 2016 Georgia Institute of Technology. All Rights Reserved.
 */

/*********************************************************************
 *                  !!  DO NOT CHANGE THIS FILE  !!                  *
 *********************************************************************/

#include <time.h> // for clock_gettime()

// for in/output
#include <iostream>
#include <vector>
#include <string>
#include <algorithm>

// for MPI
#include <mpi.h>

// own includes
#include "utils.h"
#include "radix_sort.h"
#ifdef USE_MYSTRUCT_OPT
#include "mystruct_opt.h"
#else
#include "mystruct.h"
#endif

// the executable name
std::string exe_name;

/**
 * Prints the usage of the program.
 */
void print_usage()
{
    std::cerr << "Usage: " << exe_name << " [options]" << std::endl;
    std::cerr << "      Options:" << std::endl;
    std::cerr << "          -n <n>       Sets the total number of input elements. This number must be a multiple of the number of MPI processes." << std::endl;
    std::cerr << "        OR" << std::endl;
    std::cerr << "          -m <m>       Sets the number of input elements per process." << std::endl;
    std::cerr << "      Optional:" << std::endl;
    std::cerr << "          -t           Print timing results in tablular format: \"n\tp\tk\ttime\"" << std::endl;
    std::cerr << "          -k           The number of bits to sort by in each iteration of radix sort." << std::endl;
    std::cerr << "      Example:" << std::endl;
    std::cerr << "          " << exe_name << " -m 1000" << std::endl;
    std::cerr << "                  Generates 1000 random elements on each process and then sorts them using parallel radix sort." << std::endl;
}



// enable time measurements on MAC OS
#ifdef __MACH__
#include <mach/clock.h>
#include <mach/mach.h>
#endif


// timing function that works for both Linux and MAC OS
void my_gettime(struct timespec *ts) {
#ifdef __MACH__ // OS X does not have clock_gettime, use clock_get_time
  clock_serv_t cclock;
  mach_timespec_t mts;
  host_get_clock_service(mach_host_self(), SYSTEM_CLOCK, &cclock);
  clock_get_time(cclock, &mts);
  mach_port_deallocate(mach_task_self(), cclock);
  ts->tv_sec = mts.tv_sec;
  ts->tv_nsec = mts.tv_nsec;
#else
  clock_gettime(CLOCK_MONOTONIC, ts);
#endif
}

int main(int argc, char *argv[])
{
    // Init MPI
    MPI_Init(&argc, &argv);

    // set up MPI
    int rank, p;
    // get total size of processors and current rank
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &p);

    int n = -1;
    int m = -1;
    int k = -1;
    bool tabular = false;

    /***************************
     *  Parse input arguments  *
     ***************************/
    if (rank == 0) {
        exe_name = argv[0];

        // forget about first argument (which is the executable's name)
        argc--;
        argv++;


        // parse optional parameters
        while(argc > 0 && argv[0][0] == '-')
        {
            char option = argv[0][1];
            switch (option)
            {
                case 'n':
                    // the next argument must be the number
                    argv++;
                    argc--;
                    n = atoi(argv[0]);
                    break;
                case 'k':
                    // the next argument must be the number
                    argv++;
                    argc--;
                    k = atoi(argv[0]);
                    break;
                case 't':
                    tabular = true;
                    break;
                case 'm':
                    // the next argument must be the number
                    argv++;
                    argc--;
                    m = atoi(argv[0]);
                    break;
                default:
                    print_usage();
                    MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
            }
            // iterate to next argument
            argv++;
            argc--;
        }

        if (m < 1) {
          if (n < 1) {
            print_usage();
            MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
          } else if (n < p || n % p != 0) {
            std::cerr << "[ERROR] n must be a multiple of p" << std::endl;
            MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
          }
        }

        if (m < 1) {
            m = n / p;
        } else {
            n = m*p;
        }

        if (k < 1) {
            k = 8;
        }
    }

    MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&m, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&k, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // seed the random generator
    // NOTE: This is for random input generation.
    srand(1337*rank);

    /****************************
     *  Generate input locally  *
     ****************************/
    std::vector<MyStruct> input(m);
    std::generate(input.begin(), input.end(), mystruct_rand);

    // get input datatype
    MPI_Datatype mystruct_dt = mystruct_get_mpi_type();

    /*****************************************
     *  Parallel, distirbuted radix sorting  *
     *****************************************/
    // Timing only on master node (and barrierized)
    MPI_Barrier (MPI_COMM_WORLD);
    // start timer
    struct timespec t_start, t_end;
    my_gettime(&t_start);

    // actually sort
    radix_sort(&input[0], &input[0] + m, &mystruct_key_access, mystruct_dt, MPI_COMM_WORLD, k);

    MPI_Barrier (MPI_COMM_WORLD);
    // get elapsed time in seconds
    my_gettime(&t_end);
    double time_ms = (t_end.tv_sec - t_start.tv_sec) * 1e+3
        + (double) (t_end.tv_nsec - t_start.tv_nsec) * 1e-6;
    // Timing only on master node (and barrierized)

    // output time
    if (rank == 0) {
        if (tabular) {
            std::cout << n << "\t" << p << "\t" << k << "\t" << time_ms << std::endl;
        } else {
            std::cout << "Time for sorting " << n << " elements: " << time_ms << " ms" << std::endl;
        }
    }

    // finish up
    MPI_Finalize();
    return 0;
}


