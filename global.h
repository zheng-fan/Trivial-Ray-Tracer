#ifdef MPIRUN
#include "mpi.h"
#endif
#include <algorithm>
#include <bitset>
#include <cassert>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <deque>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <list>
#include <map>
#include <numeric>
#include <queue>
#include <set>
#include <sstream>
#include <stack>
#include <string>
#include <vector>
#define debug puts("-----")
#define inf (1 << 30)
#define INF (1ll << 62)
#define finf ((float)(1e30))
#define FINF (1e300)
//using namespace std;
const double PI = 3.14159265358979323846264338327950288;
const double eps = 1e-4;
//#define FZGLM
#ifdef FZGLM
#include <fzglm.h>
#else
#include <glm/ext.hpp>
#include <glm/glm.hpp>
#endif
//using namespace glm;
//std::ostream &operator<< (std::ostream &out, const glm::dmat3 &v)
//{
//    out<<glm::to_string(v);
//    return out;
//}
std::ostream &operator<<(std::ostream &out, const glm::dvec3 &v) {
    out << glm::to_string(v);
    return out;
}
std::ostream &operator<<(std::ostream &out, const glm::dvec2 &v) {
    out << glm::to_string(v);
    return out;
}
//#define sqr(x) ((x)*(x))
template <typename T>
double norm(const T &x) {
    return sqrt(dot(x, x));
}
template <typename T>
double norm2(const T &x) {
    return dot(x, x);
}

namespace fz {
#define RAND48_SEED_0 (0x330e)
#define RAND48_SEED_1 (0xabcd)
#define RAND48_SEED_2 (0x1234)
#define RAND48_MULT_0 (0xe66d)
#define RAND48_MULT_1 (0xdeec)
#define RAND48_MULT_2 (0x0005)
#define RAND48_ADD (0x000b)

unsigned short _rand48_mult[3] =
    {
        RAND48_MULT_0,
        RAND48_MULT_1,
        RAND48_MULT_2};
unsigned short _rand48_add = RAND48_ADD;

void _dorand48(unsigned short xseed[3]) {
    unsigned long accu;
    unsigned short temp[2];

    accu = (unsigned long)_rand48_mult[0] * (unsigned long)xseed[0] +
           (unsigned long)_rand48_add;
    temp[0] = (unsigned short)accu; /* lower 16 bits */
    accu >>= sizeof(unsigned short) * 8;
    accu += (unsigned long)_rand48_mult[0] * (unsigned long)xseed[1] +
            (unsigned long)_rand48_mult[1] * (unsigned long)xseed[0];
    temp[1] = (unsigned short)accu; /* middle 16 bits */
    accu >>= sizeof(unsigned short) * 8;
    accu += _rand48_mult[0] * xseed[2] + _rand48_mult[1] * xseed[1] + _rand48_mult[2] * xseed[0];
    xseed[0] = temp[0];
    xseed[1] = temp[1];
    xseed[2] = (unsigned short)accu;
}

double erand48(unsigned short xseed[3]) {
    _dorand48(xseed);
    return ldexp((double)xseed[0], -48) +
           ldexp((double)xseed[1], -32) +
           ldexp((double)xseed[2], -16);
}

} // namespace fz

// 组合数打表
const int CN = 60;
double c[CN][CN] = {};
void cinit() {
    for (int i = 0; i < CN; i++) {
        c[i][0] = c[i][i] = 1;
        for (int j = 1; j < i; j++)
            c[i][j] = c[i - 1][j] + c[i - 1][j - 1];
    }
}
