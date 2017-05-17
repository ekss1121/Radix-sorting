/**
 * @file    mystruct_opt.cpp
 * @author  Patrick Flick <patrick.flick@gmail.com>
 *
 * Copyright (c) 2016 Georgia Institute of Technology. All Rights Reserved.
 */
#include "mystruct_opt.h"
#include <iostream>
#include <cstdlib>

/**
 * Returns the MPI_Datatype for `MyStruct`.
 *
 * TODO: You have to implement this function here:
 */
MPI_Datatype mystruct_get_mpi_type() {
    // use MPI commands to create a custom data type for MyStruct
    // TODO: create the MPI datatype for MyStruct
    MPI_Datatype type;
    MyStruct a; // define MyStruct type variable a
    int blens[3] = {1, 4, 1};
    MPI_Datatype types[3] = {MPI_DOUBLE, MPI_CHAR, MPI_UNSIGNED};
    MPI_Datatype mpi_temp_t;
    MPI_Aint base, adr_key, adr_d, adr_e;
    // Calculating offset independent of archiecture
    MPI_Get_address(&a, &base);
    MPI_Get_address(&a.key, &adr_key);
    MPI_Get_address(&a.d, &adr_d);
    MPI_Get_address(&a.e[0], &adr_e);
    MPI_Aint disps[3] = {adr_d - base, adr_e - base, adr_key - base};
    MPI_Aint extend = sizeof(a);
    // TODO: create the MPI datatype for MyStruct
    MPI_Type_create_struct(3, blens, disps, types, &mpi_temp_t);
    MPI_Type_create_resized(mpi_temp_t, 0, extend, &type);
    MPI_Type_commit(&type);
    return type;

    MPI_Type_commit(&type);
    return type;
}


/*********************************************************************
 *              Don't change anything beyond this point              *
 *********************************************************************/

/**
 * Returns a random instance of `MyStruct`.
 */
MyStruct mystruct_rand() {
    MyStruct result;
    result.key = rand();
    result.d = rand()*1.0/RAND_MAX;
    result.e[0] = 'a' + rand() % 26;
    result.e[1] = 'a' + rand() % 26;
    result.e[2] = 'a' + rand() % 26;
    result.e[3] = 0;
    return result;
}

/**
 * Returns the value of the key of `MyStruct`.
 */
unsigned int mystruct_key_access(const MyStruct& s) {
    return s.key;
}

// output format for MyStruct
std::ostream& operator <<(std::ostream& stream, const MyStruct& s) {
    stream << "{key=" << s.key << ",d=" << s.d << ",e=\"" << s.e << "\"}";
    return stream;
}

// operator used for comparison based sorting
bool operator<(const MyStruct& lhs, const MyStruct& rhs) {
    return lhs.key < rhs.key;
}
// operator for comparing if two MyStruct's are equal
bool operator==(const MyStruct& lhs, const MyStruct& rhs) {
    return (lhs.key == rhs.key) && (lhs.d == rhs.d) && (std::string(lhs.e) == std::string(rhs.e));
}
