#include "utils.hpp"

bool sparsityVectorAny(SparsityVector sv, int size) {
    bool res = false;
    for (int i = 0; i < size; ++i) {
        if (sv[i])
            return true;
    }
    return res;
}
