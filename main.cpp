#include "FreeImage/FreeImage.h"
#include "global.h"
using glm::byte;
using glm::dmat3;
using glm::dvec2;
using glm::dvec3;
using glm::normalize;
using glm::radians;
using std::cout;
using std::endl;
using std::max;
using std::min;
struct Image {
    dvec3 *d;
    int width, height;
    byte *img;
    Image(int width = 1, int height = 1) : width(width), height(height) {
        d = new dvec3[width * height];
        img = new byte[width * height * 3];
        //        img=new byte[width*height*3*2]; // *2是为了防止MPI_Gather时访问到不可用内存
    }
    dvec3 *operator[](int k) {
        return d + k * width;
    }
    byte *build(int st, int ed) {
        for (int i = st; i < ed; i++) {
            img[i * 3] = d[i].b * 255;
            img[i * 3 + 1] = d[i].g * 255;
            img[i * 3 + 2] = d[i].r * 255;
        }
        return img;
    }
    dvec3 *buildback(int st, int ed) {
        for (int i = st; i < ed; i++) {
            d[i].b = img[i * 3] / 255.0;
            d[i].g = img[i * 3 + 1] / 255.0;
            d[i].r = img[i * 3 + 2] / 255.0;
        }
        return d;
    }
    ~Image() {
        delete[] d;
        delete[] img;
    }
    void Debug() {
        for (int id = 0; id < width * height; id++) {
            int i = id / width;
            int j = id % width;
            std::cout << i << ' ' << j << ' ' << (*this)[i][j] << std::endl;
        }
    }
};
struct Object {
    // 系数：环境光ambient，漫反射diffuse，镜面反射specular。（用于直接光源）
    // 环境光系数Ka代替了光强Ia
    dvec3 Ka, Kd, Ks;
    double eta;       // 折射率
    double shininess; // 光泽常数
    // 系数：透射系数，反射系数。（用于递归）
    dvec3 KtP, KsP;
    //    // 物体发出的光，即发射光emission
    //    dvec3 Ie;
    Object(const dvec3 &Ka, const dvec3 &Kd, const dvec3 &Ks, const double &eta, const double &shininess, const dvec3 &KtP, const dvec3 &KsP) : Ka(Ka), Kd(Kd), Ks(Ks), eta(eta), shininess(shininess), KtP(KtP), KsP(KsP) {}
    virtual dvec3 normdir(const dvec3 &pos) const = 0;
    virtual std::pair<int, dvec3> IntersectPoint(const dvec3 &o, const dvec3 &dir) = 0;
    virtual dvec3 GetTexure(const dvec3 &pos) const = 0;
};
struct Sphere : public Object {
    dvec3 center;
    double radius;
    dvec3 texure;
    Sphere(const dvec3 &Ka, const dvec3 &Kd, const dvec3 &Ks, const double &eta, const double &shininess, const dvec3 &KtP, const dvec3 &KsP, const dvec3 &center, const double &radius, const dvec3 &texure) : Object(Ka, Kd, Ks, eta, shininess, KtP, KsP), center(center), radius(radius), texure(texure) {}
    std::pair<int, dvec3> IntersectPoint(const dvec3 &o, const dvec3 &dir) {
        const dvec3 &P0 = o;
        const dvec3 &P1 = dir;
        dvec3 P0_center = P0 - center;
        double a = dot(P1, P1);
        double b = 2 * dot(P1, P0_center);
        double c = dot(P0_center, P0_center) - radius * radius;
        double delta = b * b - 4 * a * c;
        if (delta <= 0)
            return std::make_pair(0, dvec3());
        double t1 = (-b - sqrt(delta)) / (2 * a);
        double t2 = (-b + sqrt(delta)) / (2 * a);
        if (t1 < 0) {
            if (t2 < 0)
                return std::make_pair(0, dvec3());
            return std::make_pair(1, P0 + t2 * P1);
        }
        return std::make_pair(1, P0 + t1 * P1);
    }
    dvec3 normdir(const dvec3 &pos) const {
        return normalize(pos - center);
    }
    dvec3 GetTexure(const dvec3 &pos) const {
        return texure;
    }
};
struct Triangle : public Object {
    dvec3 p0, p1, p2;
    dvec3 pn;    // 用一个点表示法向在哪一边
    int textype; // 0代表直接使用利用texure指定的颜色，1代表黑白相间格，2代表贴图
    dvec3 n;     // 法向
    dvec2 vt0, vt1, vt2;
    dvec3 texure;
    Image *img;
    Triangle(const dvec3 &Ka, const dvec3 &Kd, const dvec3 &Ks, const double &eta, const double &shininess, const dvec3 &KtP, const dvec3 &KsP, const dvec3 &p0, const dvec3 &p1, const dvec3 &p2, const dvec3 &pn, const int textype, const dvec2 &vt0, const dvec2 &vt1, const dvec2 &vt2, const dvec3 &texure, Image *img) : Object(Ka, Kd, Ks, eta, shininess, KtP, KsP), p0(p0), p1(p1), p2(p2), pn(pn), textype(textype), vt0(vt0), vt1(vt1), vt2(vt2), texure(texure), img(img) {
        n = normalize(triangleNormal(p0, p1, p2));
        if (dot(n, pn - p0) < 0)
            n = -n;
    }
    std::pair<int, dvec3> IntersectPoint(const dvec3 &o, const dvec3 &dir) {
        dvec3 e1 = p1 - p0;
        dvec3 e2 = p2 - p0;
        dvec3 q = cross(dir, e2);
        double a = dot(e1, q);
        if (a > -eps && a < eps)
            return std::make_pair(0, dvec3());
        double f = 1 / a;
        dvec3 s = o - p0;
        double u = f * dot(s, q);
        if (u < 0)
            return std::make_pair(0, dvec3());
        dvec3 r = cross(s, e1);
        double v = f * dot(dir, r);
        if (v < 0 || u + v > 1)
            return std::make_pair(0, dvec3());
        double t = f * dot(e2, r);
        if (t < 0)
            return std::make_pair(0, dvec3()); // 在相反的方向
        return std::make_pair(1, o + t * dir);
    }
    dvec3 normdir(const dvec3 &pos) const {
        return n;
    }
    dvec3 GetTexure(const dvec3 &pos) const {
        const double L = 20.0;
        if (textype == 0)
            return texure;
        if (textype == 1) {
            double Lx = dot(pos - p0, normalize(p1 - p0));
            double Ly = dot(pos - p1, normalize(p2 - p1));
            int nx = (int)(Lx / L);
            int ny = (int)(Ly / L);
            return dvec3((nx & 1) ^ (ny & 1));
        }
        if (textype == 2) {
            double u = norm(cross(p1 - pos, p2 - pos)) / 2;
            double v = norm(cross(p0 - pos, p2 - pos)) / 2;
            double w = norm(cross(p0 - pos, p1 - pos)) / 2;
            double l = u + v + w;
            u /= l, v /= l, w /= l;
            dvec2 L = u * vt0 + v * vt1 + w * vt2;
            int nx = L.x * img->height;
            int ny = L.y * img->width;
            return (*img)[nx][ny];
        }
        return texure;
    }
};
struct Rectangle : public Object {
    dvec3 p0, p1, p2;
    dvec3 pn;
    int textype; // 0代表直接使用利用texure指定的颜色，1代表黑白相间格，2代表贴图
    dvec3 n;
    dvec3 texure;
    Image *img;
    Rectangle(const dvec3 &Ka, const dvec3 &Kd, const dvec3 &Ks, const double &eta, const double &shininess, const dvec3 &KtP, const dvec3 &KsP, const dvec3 &p0, const dvec3 &p1, const dvec3 &p2, const dvec3 &pn, const int textype, const dvec3 &texure, Image *img) : Object(Ka, Kd, Ks, eta, shininess, KtP, KsP), p0(p0), p1(p1), p2(p2), pn(pn), textype(textype), texure(texure), img(img) {
        n = normalize(triangleNormal(p0, p1, p2));
        if (dot(n, pn - p0) < 0)
            n = -n;
    }
    std::pair<int, dvec3> IntersectPoint(const dvec3 &o, const dvec3 &dir) {
        Triangle t1(Ka, Kd, Ks, eta, shininess, KtP, KsP, p0, p1, p2, pn, 0, dvec2(), dvec2(), dvec2(), texure, NULL);
        std::pair<int, dvec3> res = t1.IntersectPoint(o, dir);
        if (res.first)
            return res;
        Triangle t2(Ka, Kd, Ks, eta, shininess, KtP, KsP, p2, p0 + p2 - p1, p0, pn, 0, dvec2(), dvec2(), dvec2(), texure, NULL);
        return t2.IntersectPoint(o, dir);
    }
    dvec3 normdir(const dvec3 &pos) const {
        return n;
    }
    dvec3 GetTexure(const dvec3 &pos) const {
        const double L = 20.0;
        if (textype == 0)
            return texure;
        if (textype == 1) {
            double Lx = dot(pos - p0, normalize(p1 - p0));
            double Ly = dot(pos - p1, normalize(p2 - p1));
            int nx = (int)(Lx / L);
            int ny = (int)(Ly / L);
            return dvec3((nx & 1) ^ (ny & 1));
        }
        if (textype == 2) {
            double Lx = dot(pos - p0, normalize(p1 - p0)) / norm(p1 - p0) * img->height;
            double Ly = dot(pos - p1, normalize(p2 - p1)) / norm(p2 - p1) * img->width;
            int nx = Lx;
            int ny = Ly;
            return (*img)[nx][ny];
        }
        return texure;
    }
};
typedef std::vector<std::vector<dvec3>> CtrlPoints;
typedef std::vector<Object *> Scene;
void DrawCuboid(Scene &scene, const dvec3 &Ka, const dvec3 &Kd, const dvec3 &Ks, const double &eta, const double &shininess, const dvec3 &KtP, const dvec3 &KsP, const dvec3 &p, const double L, const double W, const double H, const int textype, const dvec3 &texure) {
    dvec3 dx = dvec3(L / 2, 0, 0);
    dvec3 dy = dvec3(0, W / 2, 0);
    dvec3 dz = dvec3(0, 0, H / 2);
    scene.push_back(new Rectangle(Ka, Kd, Ks, eta, shininess, KtP, KsP, p - dx - dy - dz, p - dx - dy + dz, p + dx - dy + dz, p - 2.0 * dy, textype, texure, NULL)); // 下
    scene.push_back(new Rectangle(Ka, Kd, Ks, eta, shininess, KtP, KsP, p - dx + dy - dz, p - dx + dy + dz, p + dx + dy + dz, p + 2.0 * dy, textype, texure, NULL)); // 上
    scene.push_back(new Rectangle(Ka, Kd, Ks, eta, shininess, KtP, KsP, p - dx - dy - dz, p - dx - dy + dz, p - dx + dy + dz, p - 2.0 * dx, textype, texure, NULL)); // 左
    scene.push_back(new Rectangle(Ka, Kd, Ks, eta, shininess, KtP, KsP, p + dx - dy - dz, p + dx - dy + dz, p + dx + dy + dz, p + 2.0 * dx, textype, texure, NULL)); // 右
    scene.push_back(new Rectangle(Ka, Kd, Ks, eta, shininess, KtP, KsP, p - dx - dy - dz, p + dx - dy - dz, p + dx + dy - dz, p - 2.0 * dz, textype, texure, NULL)); // 前
    scene.push_back(new Rectangle(Ka, Kd, Ks, eta, shininess, KtP, KsP, p - dx - dy + dz, p + dx - dy + dz, p + dx + dy + dz, p + 2.0 * dz, textype, texure, NULL)); // 后
}
struct BezierSurface : public Object {
    CtrlPoints P;
    dvec3 BoxO;
    double L, W, H;
    dvec3 texure;
    Scene BoundingBox;
    int m, n;
    double *Bim, *Bjn, *dBimdu, *dBjndv;
    dvec3 bezierN; // 法向
    BezierSurface(const dvec3 &Ka, const dvec3 &Kd, const dvec3 &Ks, const double &eta, const double &shininess, const dvec3 &KtP, const dvec3 &KsP, const CtrlPoints &P, const dvec3 &BoxO, const double L, const double W, const double H, const dvec3 &texure) : Object(Ka, Kd, Ks, eta, shininess, KtP, KsP), P(P), BoxO(BoxO), L(L), W(W), H(H), texure(texure) {
        DrawCuboid(BoundingBox, dvec3(), dvec3(), dvec3(), 0, 0, dvec3(), dvec3(), BoxO, L, W, H, 0, dvec3());
        m = P.size() - 1;    // 范围是0-m
        n = P[0].size() - 1; // 范围是0-n
        Bim = new double[m + 1];
        dBimdu = new double[m + 1];
        Bjn = new double[n + 1];
        dBjndv = new double[n + 1];
    }
    dvec3 BoundingBoxPoint;
    int JudgeBoundingBox(const dvec3 &o, const dvec3 &dir) {
        for (Scene::const_iterator i = BoundingBox.begin(); i != BoundingBox.end(); i++) {
            Object *obj = *i;
            std::pair<int, dvec3> ret = obj->IntersectPoint(o, dir);
            if (ret.first) {
                BoundingBoxPoint = ret.second;
                return 1;
            }
        }
        return 0;
    }
    /*
    // 使用书上P107的做法
    void calcB(double u, double v) {
        for (int i = 0; i <= m; i++) {
            Bim[i] = c[m][i] * pow(u, i) * pow(1 - u, m - i);
            // c[m][i]*(i*pow(u,i-1)*pow(1-u,m-i)+(m-i)*pow(u,i)*pow(1-u,m-i-1));
            dBimdu[i] = Bim[i] * (i / u + (m - i) / (1 - u));
        }
        for (int j = 0; j <= n; j++) {
            Bjn[j] = c[n][j] * pow(v, j) * pow(1 - v, n - j);
            // c[n][j]*(j*pow(v,j-1)*pow(1-v,n-j)+(n-j)*pow(v,j)*pow(1-v,n-j-1));
            dBjndv[j] = Bjn[j] * (j / v + (n - j) / (1 - v));
        }
    }
    std::pair<int, dvec3> IntersectPoint(const dvec3 &o, const dvec3 &dir) {
        if (JudgeBoundingBox(o, dir) == 0)
            return std::make_pair(0, dvec3());
        int iternum = 10;
        double ti = 0.01, ui = 0.01, vi = 0.01;
        // 计算初始值
        calcB(0, 0);
        dvec3 Siminus1 = dvec3(0);
        for (int i = 0; i <= m; i++)
            for (int j = 0; j <= n; j++)
                Siminus1 += Bim[i] * Bjn[j] * P[i][j];
        dvec3 Ciminus1 = o;
        calcB(ui, vi);
        dvec3 Si = dvec3(0);
        for (int i = 0; i <= m; i++)
            for (int j = 0; j <= n; j++)
                Si += Bim[i] * Bjn[j] * P[i][j];
        dvec3 Ci = o + ti * dir;
        while (iternum--) {
            //            cout<<ti<<' '<<ui<<' '<<vi,debug;
            dvec3 pSpu = dvec3(0);
            for (int i = 0; i <= m; i++)
                for (int j = 0; j <= n; j++)
                    pSpu += dBimdu[i] * Bjn[j] * P[i][j];
            dvec3 pSpv = dvec3(0);
            for (int i = 0; i <= m; i++)
                for (int j = 0; j <= n; j++)
                    pSpv += Bim[i] * dBjndv[j] * P[i][j];
            //            cout<<pSpu<<' '<<pSpv,debug;
            dvec3 dCdt = dir;
            double D = dot(dCdt, cross(pSpu, pSpv));
            //            cout<<cross(pSpu,pSpv)<<' '<<D,debug;
            dvec3 df = (Ci - Ciminus1) - (Si - Siminus1);
            double dt = dot(pSpu, cross(pSpv, df)) / D;
            double du = dot(dCdt, cross(pSpv, df)) / D;
            double dv = -dot(dCdt, cross(pSpu, df)) / D;
            ti += dt;
            ui += du;
            vi += dv;
            calcB(ui, vi);
            dvec3 Siplus1 = dvec3(0);
            for (int i = 0; i <= m; i++)
                for (int j = 0; j <= n; j++)
                    Siplus1 += Bim[i] * Bjn[j] * P[i][j];
            dvec3 Ciplus1 = o + ti * dir;
            //            cout<<Ciplus1<<' '<<Siplus1,debug;
            //            cout<<dt<<' '<<du<<' '<<dv,debug;
            if (norm(Ciplus1 - Siplus1) < eps) {
                debug;
                dvec3 newpSpu = dvec3(0);
                for (int i = 0; i <= m; i++)
                    for (int j = 0; j <= n; j++)
                        newpSpu += dBimdu[i] * Bjn[j] * P[i][j];
                dvec3 newpSpv = dvec3(0);
                for (int i = 0; i <= m; i++)
                    for (int j = 0; j <= n; j++)
                        newpSpv += Bim[i] * dBjndv[j] * P[i][j];
                bezierN = normalize(cross(newpSpu, newpSpv));
                return std::make_pair(1, Ciplus1);
            }
            // 进行下一次迭代
            Siminus1 = Si;
            Ciminus1 = Ci;
            Si = Siplus1;
            Ci = Ciplus1;
        }
        return std::make_pair(0, dvec3());
    }
    */
    // 使用Jacobian矩阵的方法进行牛顿迭代
    void calcB(double u, double v) {
        for (int i = 0; i <= m; i++) {
            Bim[i] = c[m][i] * pow(u, i) * pow(1 - u, m - i);
            // c[m][i]*(i*pow(u,i-1)*pow(1-u,m-i)+(m-i)*pow(u,i)*pow(1-u,m-i-1));
            dBimdu[i] = Bim[i] * (i / u + (m - i) / (1 - u));
        }
        for (int j = 0; j <= n; j++) {
            Bjn[j] = c[n][j] * pow(v, j) * pow(1 - v, n - j);
            // c[n][j]*(j*pow(v,j-1)*pow(1-v,n-j)+(n-j)*pow(v,j)*pow(1-v,n-j-1));
            dBjndv[j] = Bjn[j] * (j / v + (n - j) / (1 - v));
        }
    }
    std::pair<int, dvec3> IntersectPoint(const dvec3 &o, const dvec3 &dir) {
        if (JudgeBoundingBox(o, dir) == 0)
            return std::make_pair(0, dvec3());
        int iternum = 10;
        dvec3 xi = dvec3(0.01, 0.01, 0.01);
        while (iternum--) {
            calcB(xi.y, xi.z); // ui, vi
            dvec3 pSpu = dvec3(0);
            for (int i = 0; i <= m; i++)
                for (int j = 0; j <= n; j++)
                    pSpu += dBimdu[i] * Bjn[j] * P[i][j];
            dvec3 pSpv = dvec3(0);
            for (int i = 0; i <= m; i++)
                for (int j = 0; j <= n; j++)
                    pSpv += Bim[i] * dBjndv[j] * P[i][j];
            dvec3 dCdt = dir;
            // for Jacobian矩阵
            dvec3 pFpt = dCdt;
            dvec3 pFpu = -pSpu;
            dvec3 pFpv = -pSpv;
            dmat3 FPxi = dmat3(pFpt.x, pFpt.y, pFpt.z, pFpu.x, pFpu.y, pFpu.z, pFpv.x, pFpv.y, pFpv.z);
            dvec3 Si = dvec3(0);
            for (int i = 0; i <= m; i++)
                for (int j = 0; j <= n; j++)
                    Si += Bim[i] * Bjn[j] * P[i][j];
            dvec3 Ci = o + xi.x * dir; // xi.x就是ti
            dvec3 Fxi = Ci - Si;
            dvec3 xiplus1 = xi - inverse(FPxi) * Fxi;
            if (norm(Fxi) < eps) {
                debug;
                bezierN = normalize(cross(pSpu, pSpv));
                return std::make_pair(1, o + xiplus1.x * dir);
            }
            // 进行下一次迭代
            xi = xiplus1;
        }
        return std::make_pair(0, dvec3());
    }
    dvec3 normdir(const dvec3 &pos) const {
        return bezierN; // 这个法向量必须求交之后才能用
    }
    dvec3 GetTexure(const dvec3 &pos) const {
        return texure;
    }
    ~BezierSurface() {
        delete[] Bim;
        delete[] dBimdu;
        delete[] Bjn;
        delete[] dBjndv;
    }
};
int intersectFnum;
struct Grid : public Object {
    int textype;
    dvec3 texure;
    Image *img;
    std::vector<Triangle> p;
    Scene BoundingBox;
    dvec3 BoundingBoxPoint;
    Grid(const dvec3 &Ka, const dvec3 &Kd, const dvec3 &Ks, const double &eta, const double &shininess, const dvec3 &KtP, const dvec3 &KsP, const std::vector<dvec3> &v, const std::vector<dvec2> &vt, const std::vector<dvec3> &vn, int *f, int num, const dvec3 &BoxO, const double L, const double W, const double H, const int textype, const dvec3 &texure, Image *img) : Object(Ka, Kd, Ks, eta, shininess, KtP, KsP), textype(textype), texure(texure), img(img) {
        DrawCuboid(BoundingBox, dvec3(), dvec3(), dvec3(), 0, 0, dvec3(), dvec3(), BoxO, L, W, H, 0, dvec3());
        for (int i = 0; i < num; i++) {
            p.push_back(Triangle(Ka, Kd, Ks, eta, shininess, KtP, KsP, v[f[0]], v[f[3]], v[f[6]], vn[f[2]] + v[f[0]], textype, vt[f[1]], vt[f[4]], vt[f[7]], texure, img));
            f += 9;
        }
    }
    int JudgeBoundingBox(const dvec3 &o, const dvec3 &dir) {
        for (Scene::const_iterator i = BoundingBox.begin(); i != BoundingBox.end(); i++) {
            Object *obj = *i;
            std::pair<int, dvec3> ret = obj->IntersectPoint(o, dir);
            if (ret.first) {
                BoundingBoxPoint = ret.second;
                return 1;
            }
        }
        return 0;
    }
    std::pair<int, dvec3> IntersectPoint(const dvec3 &o, const dvec3 &dir) {
        if (JudgeBoundingBox(o, dir) == 0)
            return std::make_pair(0, dvec3());
        dvec3 mnpos;
        double mnlen2 = FINF;
        int mnintersectFnum = -1;
        for (size_t i = 0; i < p.size(); i++) {
            std::pair<int, dvec3> res = p[i].IntersectPoint(o, dir);
            if (res.first == 0)
                continue;
            dvec3 pos = res.second;
            double len2 = norm2(pos - o);
            if (len2 < mnlen2)
                mnpos = pos, mnlen2 = len2, mnintersectFnum = i;
        }
        if (mnintersectFnum != -1) {
            intersectFnum = mnintersectFnum;
            return std::make_pair(1, mnpos);
        }
        return std::make_pair(0, dvec3());
    }
    dvec3 normdir(const dvec3 &pos) const {
        return p[intersectFnum].normdir(pos);
    }
    dvec3 GetTexure(const dvec3 &pos) const {
        return p[intersectFnum].GetTexure(pos);
    }
};
struct Light {
    dvec3 intensity; // 入射光强
    Light(const dvec3 &intensity) : intensity(intensity) {}
    // 计算这一点的光强和到光源的方向
    virtual dvec3 illuminate(const dvec3 &pos, dvec3 &dir) const = 0;
    virtual int judge(const dvec3 &st, const dvec3 &pos) const = 0;
};
struct PointLight : public Light {
    dvec3 o;
    double Kc, Kl, Kq;
    PointLight(const dvec3 &o, const dvec3 &intensity, const double Kc, const double Kl, const double Kq) : Light(intensity), o(o), Kc(Kc), Kl(Kl), Kq(Kq) {}
    dvec3 illuminate(const dvec3 &pos, dvec3 &dir) const {
        dir = o - pos;
        double d = norm(dir);
        dir = normalize(dir);
        double atten = 1 / (Kc + Kl * d + Kq * d * d);
        return intensity * atten;
    }
    int judge(const dvec3 &st, const dvec3 &pos) const {
        return norm(pos - st) < norm(o - st);
    }
};
struct DirectionLight : public Light {
    dvec3 direction;
    DirectionLight(const dvec3 &direction, const dvec3 &intensity) : Light(intensity), direction(normalize(direction)) {} // 方向归一化
    dvec3 illuminate(const dvec3 &pos, dvec3 &dir) const {
        dir = -direction;
        return intensity;
    }
    int judge(const dvec3 &st, const dvec3 &pos) const {
        return 1;
    }
};
typedef std::vector<Light *> Lighting;
struct Camera {
    dvec3 eye, center, up;
    double fovx, fovy;
    dvec3 u, v, w;
    Camera(const dvec3 &eye, const dvec3 &center, const dvec3 &up, const double &fovx, const double &fovy) : eye(eye), center(center), up(up), fovx(fovx), fovy(fovy) {
        w = normalize(eye - center);
        u = normalize(cross(up, w));
        v = cross(w, u);
    }
};
struct Ray {
    dvec3 o, dir;
    double etanow;
    Ray(const dvec3 &o, const dvec3 &dir, const double &etanow) : o(o), dir(dir), etanow(etanow) {}
    dvec3 trace(const Scene &scene, const Lighting &lighting, int depth, int maxdepth, const dvec3 &PIK, double minPIK, unsigned short Xi[]) {
        // 寻找与最近的物体的交点
        Object *mnobj = NULL;
        dvec3 mnpos;
        double mnlen2 = FINF;
        int mnintersectFnum;
        for (Scene::const_iterator i = scene.begin(); i != scene.end(); i++) {
            Object *obj = *i;
            std::pair<int, dvec3> res = obj->IntersectPoint(o, dir);
            if (res.first == 0)
                continue;
            dvec3 pos = res.second;
            double len2 = norm2(pos - o);
            if (len2 < mnlen2)
                mnobj = obj, mnpos = pos, mnlen2 = len2, mnintersectFnum = intersectFnum;
        }
        if (mnobj == NULL)
            return dvec3();              // 没有交点则返回黑色
        dvec3 N = mnobj->normdir(mnpos); // 法向
        dvec3 V = -dir;                  // 视线方向
        // 计算直接光照
        dvec3 I = mnobj->Ka; // 环境光
        for (Lighting::const_iterator j = lighting.begin(); j != lighting.end(); j++) {
            Light *lig = *j;
            dvec3 L;                              // 光线方向
            dvec3 Ip = lig->illuminate(mnpos, L); // 入射光强
            dvec3 R = 2 * dot(L, N) * N - L;
            int flag = 1;
            for (Scene::const_iterator i = scene.begin(); i != scene.end(); i++) {
                Object *obj = *i;
                std::pair<int, dvec3> res = obj->IntersectPoint(mnpos - dir * eps, L); // 稍微后退一点再看能不能看到光线
                if (res.first && lig->judge(mnpos, res.second))                        // 一定要看交点在光源前面还是后面，如果是平行光源就不需要考虑，只要看有没有交点
                {
                    flag = 0;
                    break;
                }
            }
            if (flag)
                I += Ip * (mnobj->Kd * dot(L, N) + mnobj->Ks * pow(max(0.0, dot(R, V)), mnobj->shininess));
        }
        intersectFnum = mnintersectFnum;
        dvec3 C = I * mnobj->GetTexure(mnpos);
        if (++depth < maxdepth) {
            // 计算反射
            dvec3 RPIK = mnobj->KsP * PIK;
            if (norm2(RPIK) > minPIK * minPIK) {
                dvec3 RR = 2 * dot(V, N) * N - V;
                C += mnobj->KsP * Ray(mnpos + RR * eps, RR, etanow).trace(scene, lighting, depth, maxdepth, RPIK, minPIK, Xi);
            }
            // 计算折射
            dvec3 FPIK = mnobj->KtP * PIK;
            if (norm2(FPIK) > minPIK * minPIK) {
                double etanext = mnobj->eta;
                if (dot(N, V) < 0)
                    etanext = 1, N = -N;
                ;
                double costheta1 = dot(N, V);
                double etaratio = etanext / etanow;
                double costheta2 = sqrt(1 - (1 - costheta1 * costheta1) / (etaratio * etaratio));
                dvec3 RF = -V / etaratio - (costheta2 - costheta1 / etaratio) * N;
                C += mnobj->KtP * Ray(mnpos + RF * eps, RF, etanext).trace(scene, lighting, depth, maxdepth, FPIK, minPIK, Xi);
            }
#ifdef MonteCarloSample
            // 计算漫反射（蒙特卡洛）
            dvec3 DPIK = mnobj->Kd * PIK;
            if (norm2(DPIK) > minPIK * minPIK) {
                double theta = 2 * PI * fz::erand48(Xi);
                double r = fz::erand48(Xi);
                double rs = sqrt(r);
                // 得到一个交点处的坐标系
                dvec3 w = N, u = normalize(cross(fabs(w.x) > 0.1 ? dvec3(0, 1, 0) : dvec3(1), w)), v = cross(w, u);
                // 在一定的立体角范围内随机发射光线
                dvec3 RD = normalize(u * cos(theta) * rs + v * sin(theta) * rs + w * sqrt(1 - r));
                // 调整一下所占比例
                C += 0.2 * mnobj->Kd * Ray(mnpos + RD * eps, RD, etanow).trace(scene, lighting, depth, maxdepth, DPIK, minPIK, Xi);
            }
#endif
        }
        return C;
    }
    void Debug(int i, int j) {
        std::cout << i << ' ' << j << ' ' << o << ' ' << dir << std::endl;
    }
};
Ray GetRay(const Camera &cam, int i, int j, int width, int height, double etanow, double dx, double dy) {
    double wf = width / 2.0;
    double hf = height / 2.0;
    // i-hf，因此图片左下角为(0,0)
    return Ray(cam.eye, normalize(double(tan(radians(cam.fovx / 2.0)) * ((j - wf + dx) / wf)) * cam.u + double(tan(radians(cam.fovy / 2.0)) * ((i - hf + dy) / hf)) * cam.v - cam.w), etanow);
}
#ifdef MPIRUN
MPI_Status status;
#endif
int rk = 0, sz = 1;
int raypercore, rayidfirst, rayidlast;
byte *buf;
void RayTrace(const Camera &cam, const Scene &scene, const Lighting &lighting, int width, int height, int samplenum) {
    int tot = width * height;
    raypercore = (tot + sz - 1) / sz;
    buf = new byte[raypercore * 3];
    rayidfirst = rk * raypercore;
    rayidlast = std::min((rk + 1) * raypercore, tot);
    double statcnt = 0.1;
//    printf("rayidfirst: %d, rayidlast: %d\n",rayidfirst,rayidlast);
#ifdef OPENACC
#pragma acc parallel loop private(cam, scene, lighting)
#endif
    for (int rayid = rayidfirst; rayid < rayidlast; rayid++) {
        if (rayid - rayidfirst + 1 >= (rayidlast - rayidfirst) * statcnt) {
            printf("Rank: %d, Progress: %.0f%%\n", rk, statcnt * 100);
            statcnt += 0.1;
        }
        int i = rayid / width;
        int j = rayid % width;
        dvec3 C = dvec3();
        //#define MonteCarloSample
        unsigned short Xi[] = {0, 0, (unsigned short)(j * j * j)};
        for (int s = 0; s < samplenum; s++) {
#ifdef MonteCarloSample
            double r1 = 2 * fz::erand48(Xi), dx = r1 < 1 ? sqrt(r1) - 1 : 1 - sqrt(2 - r1);
            double r2 = 2 * fz::erand48(Xi), dy = r2 < 1 ? sqrt(r2) - 1 : 1 - sqrt(2 - r2);
            Ray ray = GetRay(cam, i, j, width, height, 1.0, dx, dy);
#else
            Ray ray = GetRay(cam, i, j, width, height, 1.0, 0, 0);
#endif
            C += ray.trace(scene, lighting, 0, 5, dvec3(FINF), 0.0001, Xi) * (1.0 / samplenum);
        }
        C = dvec3(min(C.r, 1.0), min(C.g, 1.0), min(C.b, 1.0));
        int x = rayid - rayidfirst;
        buf[x * 3] = C.b * 255;
        buf[x * 3 + 1] = C.g * 255;
        buf[x * 3 + 2] = C.r * 255;
    }
}
void dealwithobj(const char *filename, double *v, double *vt, double *vn, int *f, int flag) {
    FILE *fp = fopen(filename, "r");
    char s[1000];
    int vcnt, vtcnt, vncnt, fcnt;
    vcnt = vtcnt = vncnt = fcnt = 0;
    while (fgets(s, 0x7fffffff, fp)) {
        if (s[0] != 'v' && s[0] != 'f')
            continue;
        if (s[0] == 'v') {
            if (s[1] == ' ')
                sscanf(s, "%*s %lf %lf %lf", &v[vcnt], &v[vcnt + 1], &v[vcnt + 2]), vcnt += 3;
            else if (s[1] == 'n')
                sscanf(s, "%*s %lf %lf %lf", &vn[vncnt], &vn[vncnt + 1], &vn[vncnt + 2]), vncnt += 3;
            else if (s[1] == 't')
                sscanf(s, "%*s %lf %lf", &vt[vtcnt], &vt[vtcnt + 1]), vtcnt += 2;
        } else if (s[0] == 'f') {
            if (flag)
                sscanf(s, "%*s %d/%d/%d %d/%d/%d %d/%d/%d", &f[fcnt], &f[fcnt + 1], &f[fcnt + 2], &f[fcnt + 3], &f[fcnt + 4], &f[fcnt + 5], &f[fcnt + 6], &f[fcnt + 7], &f[fcnt + 8]), fcnt += 9;
            else
                sscanf(s, "%*s %d//%d %d//%d %d//%d", &f[fcnt], &f[fcnt + 2], &f[fcnt + 3], &f[fcnt + 5], &f[fcnt + 6], &f[fcnt + 8]), f[fcnt + 1] = f[fcnt + 4] = f[fcnt + 7] = 1, fcnt += 9;
        }
    }
}
void addGridObj(Scene &scene, int NV, int NF, const char *filename, int flag, const dvec3 &objpos, double objscale, double L, double W, double H, int textype, const dvec3 &color, Image *img) {
    double *v = new double[NV * 3];
    double *vn = new double[NV * 3];
    double *vt = new double[NV * 2];
    int *f = new int[NF * 9];
    if (rk == 0) {
        dealwithobj(filename, v, vt, vn, f, flag);
        for (int i = 0; i < NF * 9; i++)
            f[i] -= 1;
#ifdef MPIRUN
        for (int i = 1; i < sz; i++) {
            MPI_Send(v, NV * 3, MPI_DOUBLE, i, 0, MPI_COMM_WORLD);
            MPI_Send(vn, NV * 3, MPI_DOUBLE, i, 0, MPI_COMM_WORLD);
            MPI_Send(vt, NV * 2, MPI_DOUBLE, i, 0, MPI_COMM_WORLD);
            MPI_Send(f, NF * 9, MPI_INT, i, 0, MPI_COMM_WORLD);
        }
#endif
    }
#ifdef MPIRUN
    else {
        MPI_Recv(v, NV * 3, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, &status);
        MPI_Recv(vn, NV * 3, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, &status);
        MPI_Recv(vt, NV * 2, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, &status);
        MPI_Recv(f, NF * 9, MPI_INT, 0, 0, MPI_COMM_WORLD, &status);
    }
#endif
    std::vector<dvec3> vv, vvn;
    std::vector<dvec2> vvt;
    for (int i = 0; i < NV * 3; i += 3) {
        vv.push_back(dvec3(v[i], v[i + 1], v[i + 2]) * objscale + objpos);
        vvn.push_back(dvec3(vn[i], vn[i + 1], vn[i + 2]));
    }
    for (int i = 0; i < NV * 2; i += 2)
        vvt.push_back(dvec2(vt[i], vt[i + 1]));
    scene.push_back(new Grid(dvec3(0.1), dvec3(0.4), dvec3(0), 1, 5, dvec3(0), dvec3(0), vv, vvt, vvn, f, NF, objpos, L, W, H, textype, color, img));
    //    delete[] v;
    //    delete[] vn;
    //    delete[] vt;
    //    delete[] f;
}
int main(int argc, char *argv[]) {
    long ts = time(NULL);
#ifdef MPIRUN
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rk);
    MPI_Comm_size(MPI_COMM_WORLD, &sz);
//    printf("Hello from rank: %d\n",rk);
#endif
    cinit();
    FIBITMAP *bitmap = NULL;
    if (rk == 0) {
        FreeImage_Initialise();
        bitmap = FreeImage_Load(FIF_JPEG, "image1.jpg", JPEG_DEFAULT);
        int imgwidth = FreeImage_GetWidth(bitmap);
        int imgheight = FreeImage_GetHeight(bitmap);
        int bytespp = FreeImage_GetLine(bitmap) / imgwidth;
        printf("Input Image Info: Width:%d\t Height:%d\t bytes per pixel:%d\n", imgwidth, imgheight, bytespp);
    }
    // 这里直接写死，就不用通信了
    int imgwidth = 800;
    int imgheight = 600;
    Image *imgp = new Image(imgwidth, imgheight);
    if (rk == 0) {
        FreeImage_ConvertToRawBits(imgp->img, bitmap, imgwidth * 3, 24, FI_RGBA_RED_MASK, FI_RGBA_GREEN_MASK, FI_RGBA_BLUE_MASK, true); // 此处要翻转
#ifdef MPIRUN
        for (int i = 1; i < sz; i++)
            MPI_Send(imgp->img, imgwidth * imgheight * 3, MPI_UNSIGNED_CHAR, i, 0, MPI_COMM_WORLD);
#endif
    }
#ifdef MPIRUN
    else
        MPI_Recv(imgp->img, imgwidth * imgheight * 3, MPI_UNSIGNED_CHAR, 0, 0, MPI_COMM_WORLD, &status);
#endif
    imgp->buildback(0, imgwidth * imgheight);

    Scene scene;
    Lighting lighting;
    // border
    scene.push_back(new Rectangle(dvec3(0.1), dvec3(0.4), dvec3(0), 1, 5, dvec3(0), dvec3(0.3), dvec3(-100, -50, -100), dvec3(-100, -50, 100), dvec3(100, -50, 100), dvec3(0), 1, dvec3(0.75, 0.75, 0.75), NULL)); // 下
    scene.push_back(new Rectangle(dvec3(0.1), dvec3(0.4), dvec3(0), 1, 5, dvec3(0), dvec3(0), dvec3(-100, 50, -100), dvec3(-100, 50, 100), dvec3(100, 50, 100), dvec3(0), 0, dvec3(0.75, 0.75, 0.75), NULL));      // 上
    scene.push_back(new Rectangle(dvec3(0.1), dvec3(0.4), dvec3(0), 1, 5, dvec3(0), dvec3(0), dvec3(-100, -50, -100), dvec3(-100, -50, 100), dvec3(-100, 50, 100), dvec3(0), 0, dvec3(0.75, 0.45, 0.45), NULL));   // 左
    scene.push_back(new Rectangle(dvec3(0.1), dvec3(0.4), dvec3(0), 1, 5, dvec3(0), dvec3(0), dvec3(100, -50, -100), dvec3(100, -50, 100), dvec3(100, 50, 100), dvec3(0), 0, dvec3(0.35, 0.35, 0.60), NULL));      // 右
    scene.push_back(new Rectangle(dvec3(0.1), dvec3(0.4), dvec3(0), 1, 5, dvec3(0), dvec3(0), dvec3(-100, -50, -100), dvec3(100, -50, -100), dvec3(100, 50, -100), dvec3(0), 0, dvec3(0.75, 0.75, 0.75), NULL));   // 前
    scene.push_back(new Rectangle(dvec3(0.1), dvec3(0.4), dvec3(0), 1, 5, dvec3(0), dvec3(0), dvec3(-100, -50, 100), dvec3(100, -50, 100), dvec3(100, 50, 100), dvec3(0), 0, dvec3(0.75, 0.75, 0.75), NULL));      // 后
    // ball
    scene.push_back(new Sphere(dvec3(0), dvec3(0), dvec3(0.3), 1.0, 10, dvec3(0), dvec3(1), dvec3(-30, -6.5, -25), 12, dvec3(1, 1, 1))); // 反射球
    scene.push_back(new Sphere(dvec3(0), dvec3(0), dvec3(0.0), 1.5, 10, dvec3(1), dvec3(0), dvec3(30, -35, -25), 15, dvec3(1, 1, 1)));   // 折射球
    // table
    DrawCuboid(scene, dvec3(0.1), dvec3(0.4), dvec3(0.1), 1, 5, dvec3(0), dvec3(0), dvec3(-30, -20, 0), 30, 3, 80, 0, dvec3(0.40, 0.20, 0.10));
    DrawCuboid(scene, dvec3(0.1), dvec3(0.4), dvec3(0.1), 1, 5, dvec3(0), dvec3(0), dvec3(-30 - 10, -35, -35), 3, 30, 3, 0, dvec3(0.40, 0.20, 0.10));
    DrawCuboid(scene, dvec3(0.1), dvec3(0.4), dvec3(0.1), 1, 5, dvec3(0), dvec3(0), dvec3(-30 + 10, -35, -35), 3, 30, 3, 0, dvec3(0.40, 0.20, 0.10));
    DrawCuboid(scene, dvec3(0.1), dvec3(0.4), dvec3(0.1), 1, 5, dvec3(0), dvec3(0), dvec3(-30 - 10, -35, 35), 3, 30, 3, 0, dvec3(0.40, 0.20, 0.10));
    DrawCuboid(scene, dvec3(0.1), dvec3(0.4), dvec3(0.1), 1, 5, dvec3(0), dvec3(0), dvec3(-30 + 10, -35, 35), 3, 30, 3, 0, dvec3(0.40, 0.20, 0.10));
    // mirror
    scene.push_back(new Rectangle(dvec3(0), dvec3(0), dvec3(0), 1, 5, dvec3(0), dvec3(1), dvec3(-40 - 30, 40, -99.5), dvec3(-40 - 30, -40, -99.5), dvec3(-40 + 30, -40, -99.5), dvec3(0), 0, dvec3(1, 1, 1), NULL));
    // image
    scene.push_back(new Rectangle(dvec3(0.1), dvec3(0.4), dvec3(0.1), 1, 5, dvec3(0), dvec3(0), dvec3(45 - 40, 30, -99.5), dvec3(45 - 40, -30, -99.5), dvec3(45 + 40, -30, -99.5), dvec3(0), 2, dvec3(1, 1, 1), imgp));
    // bookshelf
    DrawCuboid(scene, dvec3(0.1), dvec3(0.4), dvec3(0.1), 1, 5, dvec3(0), dvec3(0), dvec3(50 + 49, -10, -50), 2, 80, 40, 0, dvec3(0.80, 0.60, 0.30));               // 最里层
    DrawCuboid(scene, dvec3(0.1), dvec3(0.4), dvec3(0.1), 1, 5, dvec3(0), dvec3(0), dvec3(50 + 50 - 5, -10, -50 - 20), 10, 80, 2, 0, dvec3(0.80, 0.60, 0.30));      // 左侧
    DrawCuboid(scene, dvec3(0.1), dvec3(0.4), dvec3(0.1), 1, 5, dvec3(0), dvec3(0), dvec3(50 + 50 - 5, -10, -50 + 20), 10, 80, 2, 0, dvec3(0.80, 0.60, 0.30));      // 右侧
    DrawCuboid(scene, dvec3(0.1), dvec3(0.4), dvec3(0.1), 1, 5, dvec3(0), dvec3(0), dvec3(50 + 50 - 5, -10 + 39, -50), 10, 2, 40, 0, dvec3(0.80, 0.60, 0.30));      // 上侧
    DrawCuboid(scene, dvec3(0.1), dvec3(0.4), dvec3(0.1), 1, 5, dvec3(0), dvec3(0), dvec3(50 + 50 - 5, -10 - 39, -50), 10, 2, 40, 0, dvec3(0.80, 0.60, 0.30));      // 下侧
    DrawCuboid(scene, dvec3(0.1), dvec3(0.4), dvec3(0.1), 1, 5, dvec3(0), dvec3(0), dvec3(50 + 50 - 5, -10 + 39 - 20, -50), 10, 2, 40, 0, dvec3(0.80, 0.60, 0.30)); // 中层1
    DrawCuboid(scene, dvec3(0.1), dvec3(0.4), dvec3(0.1), 1, 5, dvec3(0), dvec3(0), dvec3(50 + 50 - 5, -10 + 39 - 40, -50), 10, 2, 40, 0, dvec3(0.80, 0.60, 0.30)); // 中层2
    DrawCuboid(scene, dvec3(0.1), dvec3(0.4), dvec3(0.1), 1, 5, dvec3(0), dvec3(0), dvec3(50 + 50 - 5, -10 + 39 - 60, -50), 10, 2, 40, 0, dvec3(0.80, 0.60, 0.30)); // 中层3
    // book
    DrawCuboid(scene, dvec3(0.1), dvec3(0.4), dvec3(0.1), 1, 5, dvec3(0), dvec3(0), dvec3(50 + 50 - 5 + 1, -10 + 39 - 20 + 8, -50 - 14), 8, 16, 2, 0, dvec3(0.30, 0.60, 0.30)); // 中层1上的书
    DrawCuboid(scene, dvec3(0.1), dvec3(0.4), dvec3(0.1), 1, 5, dvec3(0), dvec3(0), dvec3(50 + 50 - 5 + 1, -10 + 39 - 20 + 8, -50 - 12), 8, 16, 2, 0, dvec3(0.30, 0.20, 0.80));
    DrawCuboid(scene, dvec3(0.1), dvec3(0.4), dvec3(0.1), 1, 5, dvec3(0), dvec3(0), dvec3(50 + 50 - 5 + 1, -10 + 39 - 20 + 8, -50 - 10), 8, 16, 2, 0, dvec3(0.80, 0.20, 0.50));
    DrawCuboid(scene, dvec3(0.1), dvec3(0.4), dvec3(0.1), 1, 5, dvec3(0), dvec3(0), dvec3(50 + 50 - 5 + 1, -10 + 39 - 20 + 8, -50), 8, 16, 2, 0, dvec3(0.10, 0.70, 0.40));
    DrawCuboid(scene, dvec3(0.1), dvec3(0.4), dvec3(0.1), 1, 5, dvec3(0), dvec3(0), dvec3(50 + 50 - 5 + 1, -10 + 39 - 20 + 8, -50 + 6), 8, 16, 2, 0, dvec3(0.80, 0.50, 0.30));
    DrawCuboid(scene, dvec3(0.1), dvec3(0.4), dvec3(0.1), 1, 5, dvec3(0), dvec3(0), dvec3(50 + 50 - 5 + 1, -10 + 39 - 40 + 8, -50 - 8), 8, 16, 2, 0, dvec3(0.10, 0.20, 0.30)); // 中层2上的书
    DrawCuboid(scene, dvec3(0.1), dvec3(0.4), dvec3(0.1), 1, 5, dvec3(0), dvec3(0), dvec3(50 + 50 - 5 + 1, -10 + 39 - 40 + 8, -50 + 14), 8, 16, 2, 0, dvec3(0.30, 0.20, 0.40));
    DrawCuboid(scene, dvec3(0.1), dvec3(0.4), dvec3(0.1), 1, 5, dvec3(0), dvec3(0), dvec3(50 + 50 - 5 + 1, -10 + 39 - 60 + 8, -50 - 12), 8, 16, 2, 0, dvec3(0.70, 0.20, 0.30)); // 中层3上的书
    DrawCuboid(scene, dvec3(0.1), dvec3(0.4), dvec3(0.1), 1, 5, dvec3(0), dvec3(0), dvec3(50 + 50 - 5 + 1, -10 + 39 - 60 + 8, -50 - 2), 8, 16, 2, 0, dvec3(0.70, 0.20, 0.70));
    DrawCuboid(scene, dvec3(0.1), dvec3(0.4), dvec3(0.1), 1, 5, dvec3(0), dvec3(0), dvec3(50 + 50 - 5 + 1, -10 + 39 - 60 + 8, -50 + 10), 8, 16, 2, 0, dvec3(0.30, 0.50, 0.50));
    DrawCuboid(scene, dvec3(0.1), dvec3(0.4), dvec3(0.1), 1, 5, dvec3(0), dvec3(0), dvec3(50 + 50 - 5 + 1, -10 + 39 - 80 + 8, -50 - 2), 8, 16, 2, 0, dvec3(0.50, 0.50, 0.80)); // 底层上的书
    /*
    // BezierSurface
    double L = 10, W = 15, H = 10;
    const dvec3 BoxO = dvec3(30, -18.5 + W / 2, 0);
    const int Pm = 3, Pn = 3;
    dvec3 P[Pm][Pn] =
        {
            {dvec3(-L / 2, -W / 2, H / 2) + BoxO, dvec3(-L / 2, -W / 2, 0) + BoxO, dvec3(-L / 2, -W / 2, -H / 2) + BoxO},
            {dvec3(0, -W / 2, H / 2) + BoxO, dvec3(0, W / 2, 0) + BoxO, dvec3(0, -W / 2, -H / 2) + BoxO},
            {dvec3(L / 2, -W / 2, H / 2) + BoxO, dvec3(L / 2, -W / 2, 0) + BoxO, dvec3(L / 2, -W / 2, -H / 2) + BoxO},
        };
    CtrlPoints ctrl;
    for (int i = 0; i < Pm; i++)
        ctrl.push_back(std::vector<dvec3>(P[i], P[i] + Pn));
    scene.push_back(new BezierSurface(dvec3(0.1), dvec3(0.4), dvec3(0.0), 1.5, 10, dvec3(0), dvec3(0), ctrl, BoxO, L, W, H, dvec3(1, 1, 1)));
    */
    // Grid Object
    addGridObj(scene, 766, 1536, "Eight.obj", 1, dvec3(-30, -15, 20), 15.0, 30, 20, 30, 2, dvec3(), imgp);
    addGridObj(scene, 2588, 5146, "bunny.obj", 0, dvec3(70, -40, -40), 1 / 80.0, 30, 30, 30, 0, dvec3(0.8, 0.8, 0.8), imgp);

    // light
    // bookshelf light
    lighting.push_back(new PointLight(dvec3(50 + 50 - 5, -10 + 39 - 1.5, -50), dvec3(50.0), 0.1, 0.4, 0.5));
    lighting.push_back(new PointLight(dvec3(50 + 50 - 5, -10 + 39 - 1.5 - 20, -50), dvec3(50.0), 0.1, 0.4, 0.5));
    lighting.push_back(new PointLight(dvec3(50 + 50 - 5, -10 + 39 - 1.5 - 40, -50), dvec3(50.0), 0.1, 0.4, 0.5));
    lighting.push_back(new PointLight(dvec3(50 + 50 - 5, -10 + 39 - 1.5 - 60, -50), dvec3(50.0), 0.1, 0.4, 0.5));
#ifdef MULTILIGHT
    // 多光源模拟面光源
    double rs = 0.5, re = 10, rstep = 1;
    double thetas = 0, thetae = 360, thetastep = 6;
    for (double r = rs; r < re; r += rstep)
        for (double theta = thetas; theta < thetae; theta += thetastep)
            lighting.push_back(new PointLight(dvec3(r * sin(radians(theta)), 49.5, r * cos(radians(theta))), dvec3(10000.0 / ((re - rs) / rstep * (thetae - thetas) / thetastep)), 0.1, 0.4, 0.5));
#else
    // 单点光源
    lighting.push_back(new PointLight(dvec3(0, 49.5, 0), dvec3(10000.0), 0.1, 0.4, 0.5));
#endif
#ifdef SAMPLE
    int samplenum = 50;
#else
    int samplenum = 1;
#endif
    //    int width=800,height=600;
    //    double S=0.4;
    int S = 1;
    //    int S=3; // 1920*1080
    //    int S=3*4;
    //    int S=8;
    //    int width=640*S,height=360*S;
    int width = 256, height = 144;
    //    int width=128,height=72;
    double fovx = 80;
    //    Camera cam(dvec3(10,20,99),dvec3(0,-10,-1),dvec3(0,1,0),fovx,fovx/width*height);
    Camera cam(dvec3(0, -10, 99), dvec3(0, -10, -1), dvec3(0, 1, 0), fovx, fovx / width * height);
    RayTrace(cam, scene, lighting, width, height, samplenum);
    Image *res; // 注意不能设置成NULL！因为后面MPI_Gather即使没有实际用到数据，也要保证res->img是可以访问的
    if (rk == 0)
        res = new Image(width, height);
    else
        res = new Image(1, 1);
#ifdef MPIRUN
    MPI_Gather(buf, raypercore * 3, MPI_UNSIGNED_CHAR, res->img, raypercore * 3, MPI_UNSIGNED_CHAR, 0, MPI_COMM_WORLD);
#else
    memcpy(res->img, buf, width * height * 3);
#endif
    if (rk == 0) {
        FIBITMAP *image = FreeImage_ConvertFromRawBits(res->img, width, height, width * 3, 24, FI_RGBA_RED_MASK, FI_RGBA_GREEN_MASK, FI_RGBA_BLUE_MASK, false);
        FreeImage_Save(FIF_BMP, image, "result.bmp", 0);
        FreeImage_DeInitialise();
        printf("Time Use: %lds\n", time(NULL) - ts);
    }
#ifdef MPIRUN
    MPI_Finalize();
#endif
    return 0;
}
