#include "propagate.hpp"
#include "TeSA.hpp"

void propagateMatmul(TeSA &in1, TeSA &in2, TeSA &out) {
  // Forward
  TeSA propOut{in1 * in2};
  out = out & propOut;
  // Backward
  TeSA propIn1{out * in2.transpose()};
  in1 = in1 & propIn1;
  TeSA propIn2{out * in2.transpose()};
  in2 = in2 & propIn2;
}
