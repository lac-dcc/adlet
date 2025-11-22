#pragma once
#include <bitset>

#ifndef SIZE_MACRO
#define SIZE_MACRO 4096
#endif

constexpr int SIZE = SIZE_MACRO;
using SparsityVector = std::bitset<SIZE>;

bool sparsityVectorAny(SparsityVector sv, int size);
