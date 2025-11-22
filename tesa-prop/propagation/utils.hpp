#pragma once
#include <bitset>

constexpr int SIZE = 4096;
using SparsityVector = std::bitset<SIZE>;

bool sparsityVectorAny(SparsityVector sv, int size);
