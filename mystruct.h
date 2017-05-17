/**
 * @file    mystruct.h
 * @author  Patrick Flick <patrick.flick@gmail.com>
 *
 * Copyright (c) 2016 Georgia Institute of Technology. All Rights Reserved.
 */

#ifndef MYSTRUCT_H
#define MYSTRUCT_H

/*********************************************************************
 *                  !!  DO NOT CHANGE THIS FILE  !!                  *
 *********************************************************************/


#include <mpi.h>
#include <iostream>

// The structure we will perform radix sort on
struct MyStruct {
    unsigned int key;
    double d;
    char e[4];
};

/**
 * Returns the MPI_Datatype for `MyStruct`.
 *
 * You have to implement this function in mystruct.cpp
 */
MPI_Datatype mystruct_get_mpi_type();


/**
 * Returns a random instance of `MyStruct`.
 */
MyStruct mystruct_rand();

/**
 * Returns the value of the key of `MyStruct`.
 */
unsigned int mystruct_key_access(const MyStruct& s);


/**
 * Further utility functions that you don't have to worry about.
 */

// operator used for comparison based sorting
bool operator<(const MyStruct& lhs, const MyStruct& rhs);
// operator for comparing if two MyStruct's are equal
bool operator==(const MyStruct& lhs, const MyStruct& rhs);
// output format for MyStruct
std::ostream& operator <<(std::ostream& stream, const MyStruct& s);


#endif
