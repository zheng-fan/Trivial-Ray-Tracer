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
namespace glm {
const double PI = 3.14159265358979323846264338327950288;
typedef unsigned char byte;
double radians(double deg) {
    return deg / 180 * PI;
}
struct dvec3 {
    union {
        double x, r;
    };
    union {
        double y, g;
    };
    union {
        double z, b;
    };
    dvec3(const double _x, const double _y, const double _z) : x(_x), y(_y), z(_z) {}
    dvec3() : x(0), y(0), z(0) {}
    dvec3(double a) : x(a), y(a), z(a) {}
    dvec3 operator+(const dvec3 &b) const {
        return dvec3(x + b.x, y + b.y, z + b.z);
    }
    dvec3 &operator+=(const dvec3 &b) {
        x += b.x;
        y += b.y;
        z += b.z;
        return *this;
    }
    dvec3 operator-(const dvec3 &b) const {
        return dvec3(x - b.x, y - b.y, z - b.z);
    }
    dvec3 operator*(double r) const {
        return dvec3(x * r, y * r, z * r);
    }
    dvec3 operator*(const dvec3 &b) const {
        return dvec3(x * b.x, y * b.y, z * b.z);
    }
    dvec3 operator/(double r) const {
        return dvec3(x / r, y / r, z / r);
    }
    friend inline dvec3 operator*(double r, const dvec3 &a) {
        return dvec3(a.x * r, a.y * r, a.z * r);
    }
    friend inline dvec3 operator-(const dvec3 &a) {
        return dvec3(-a.x, -a.y, -a.z);
    }
};
double dot(const dvec3 &a, const dvec3 &b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}
dvec3 cross(const dvec3 &a, const dvec3 &b) {
    return dvec3(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x);
}
dvec3 normalize(const dvec3 &a) {
    return a * (1 / sqrt(dot(a, a)));
}
dvec3 triangleNormal(const dvec3 &p1, const dvec3 &p2, const dvec3 &p3) {
    return cross(p3 - p1, p2 - p1);
}
std::string to_string(const dvec3 &v) {
    char s[100];
    sprintf(s, "dvec3(%.6f, %.6f, %.6f)", v.x, v.y, v.z);
    return std::string(s);
}
} // namespace glm
