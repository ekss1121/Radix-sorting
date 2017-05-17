/**
 * @file    utils.h
 * @author  Patrick Flick <patrick.flick@gmail.com>
 *
 * Utility functions.
 *
 * Copyright (c) 2016 Georgia Institute of Technology. All Rights Reserved.
 */

/*********************************************************************
 *                  !!  DO NOT CHANGE THIS FILE  !!                  *
 *********************************************************************/

#ifndef UTILS_H
#define UTILS_H

#include <mpi.h>
#include <vector>
#include <iostream>

// output a std::vector
template <typename T>
std::ostream& operator <<(std::ostream& stream, const std::vector<T>& vec) {
    stream << "[";
    for (size_t i = 0; i < vec.size(); ++i) {
        if (i > 0)
            stream << ", ";
        stream << vec[i];
    }
    stream << "]";
    return stream;
}

template <typename T>
std::vector<T> gather(const std::vector<T>& in, MPI_Datatype dt, MPI_Comm comm) {
    int p;
    MPI_Comm_size(comm, &p);
    std::vector<T> result(in.size()*p);
    MPI_Gather((T*)&in[0], in.size(), dt, &result[0], in.size(), dt, 0, comm);
    return result;
}

template <typename T>
std::vector<T> scatter(const std::vector<T>& in, MPI_Datatype dt, MPI_Comm comm) {
    int p, rank;
    MPI_Comm_size(comm, &p);
    MPI_Comm_rank(comm, &rank);
    if (rank != 0 && in.size() > 0) {
        std::cerr << "std::vector scatter only scatters from rank 0" << std::endl;
        MPI_Abort(comm, -1);
    }
    if (rank == 0 && in.size() % p != 0) {
        std::cerr << "std::vector scatter assumes the number of elements is divisable by the number of processes." << std::endl;
        MPI_Abort(comm, -1);
    }
    int scatter_size = in.size() / p;
    MPI_Bcast(&scatter_size, 1, MPI_INT, 0, comm);
    std::vector<T> result(scatter_size);
    MPI_Scatter((void*)&in[0], scatter_size, dt,
                (void*)&result[0], scatter_size, dt, 0, comm);
    return result;
}

#endif
