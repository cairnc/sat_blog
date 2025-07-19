
#pragma once

#include <math.h>
#include <float.h>
#include <stdlib.h>
#include <immintrin.h>
#include <stdint.h>


#pragma warning(disable : 4200)

#ifdef _DEBUG
#define ASSERT(x) do { if (!(x)) __debugbreak(); } while (0)
#else
#define ASSERT(x) 
//#define ASSERT(x) do { if (!(x)) __debugbreak(); } while (0)
#endif


namespace Math
{
    const float PI = 3.14159f;
    const float DEG2RAD = (PI/180.0f);
    const float RAD2DEG = (180.0f/PI);

    inline float lerp(float a, float b, float t)   { return a*(1-t) + b*t; }
    template <typename T> inline T min(T x, T y)            { return x < y ? x : y; }
    template <typename T> inline T max(T x, T y)            { return x > y ? x : y; }
    template <typename T> inline T clamp(T x, T a, T b)    { return min(max(x, a), b); }

    inline float rand01()  { return rand()/(float)RAND_MAX; }

    template <typename T> 
    T sign(T x)
    {
        if(x == 0) return (T)0;
        if(x < 0) return (T)-1;
        return (T)1;
    }
}


struct Vec2 
{ 
    float x, y; 
};

struct Vec3 
{ 

    float x, y, z; 

    Vec3 operator+(float s) const { return {x+s,y+s,z+s}; }
    Vec3 operator+(Vec3 v)  const { return {x+v.x,y+v.y,z+v.z}; }
    Vec3 operator-(Vec3 v)  const { return {x-v.x,y-v.y,z-v.z}; }
    Vec3 operator-(float s) const { return {x-s,y-s,z-s}; }
    Vec3 operator*(float s) const { return {x*s,y*s,z*s}; }
    Vec3 operator/(float s) const { return {x/s,y/s,z/s}; }
    Vec3 operator-()        const  { return {-x, -y, -z}; }

    void operator+=(Vec3 v)     { *this = *this+v; }
    void operator-=(Vec3 v)     { *this = *this-v; }
    void operator*=(float s)    { *this = *this*s; }
    void operator/=(float s)    { *this = *this/s; }

    float   get(size_t index)  const { return (&x)[index]; }
    bool    isZero()        const { return x==0&&y==0&&z==0; }
    bool    isNan(Vec3 a)   const { return isnan(x) || isnan(y) || isnan(z); }
    bool    equals(Vec3 v)  const { return x == v.x && y == v.y && z == v.z; }
    float   dot(Vec3 v)     const { return x*v.x+y*v.y+z*v.z; }
    Vec3    cross(Vec3 v)   const { return { y*v.z - z*v.y, z*v.x - x*v.z, x*v.y - y*v.x }; }
    float   lengthSq()      const { return dot(*this); }
    float   length()        const { return sqrtf(dot(*this)); }
    Vec3    normalized()    const { return *this * (1.0f/length()); }
    Vec3    normalizedSafe()const { return isZero() ? *this : normalized(); }
    float   distSq(Vec3 v)  const { return (*this - v).lengthSq(); }
    float   absDist(Vec3 v) const { return fabsf(x-v.x)+fabsf(y-v.y)+fabsf(z-v.z); }
    float   dist(Vec3 v)    const { return (*this-v).length(); }
    Vec3    reflect(Vec3 n) const { return *this - n*2.0f*dot(n); }; 
    Vec3    clip(Vec3 n)    const { return *this - n*dot(n); }

    static Vec3 rand01()                        { return {Math::rand01(), Math::rand01(), Math::rand01()}; }
    static Vec3 rand11()                        { return rand01()*2.0f - 1.0f; }

    static Vec3 randSphere()
    {
        float theta = 2 * Math::PI * Math::rand01();
        float phi = acosf(1 - 2 * Math::rand01());
        float sphi = sinf(phi);
        return { sphi * cosf(theta), sphi * sinf(theta), cosf(phi) };
    }

    static Vec3 lerp(float t, Vec3 a, Vec3 b)   { return { (1-t)*a.x + t*b.x, (1-t)*a.y + t*b.y, (1-t)*a.z + t*b.z}; }
};

struct Vec3i 
{ 
    int x,y,z; 

    int     get(size_t index)  const { return (&x)[index]; }
    Vec3    toVec3()        const { return {(float)x, (float)y, (float)z}; }
    bool    equals(Vec3i v) const { return x == v.x && y == v.y && z == v.z; }

    static Vec3i    fromVec3(Vec3 v)            { return {(int)v.x, (int)v.y, (int)v.z}; }
};

struct Vec4 
{ 
    float x, y, z, w; 
}; 

struct Mat3 
{ 
    Vec3 c0,c1,c2; 

    Mat3 operator*(float s) const { return {c0*s, c1*s, c2*s}; }
    Mat3 operator+(Mat3 m)  const { return {c0+m.c0, c1+m.c1, c2+m.c2}; }
    Vec3 operator*(Vec3 v)  const { return c0*v.x + c1*v.y + c2*v.z; }
    Mat3 operator*(Mat3 m)  const { return {*this*m.c0, *this*m.c1, *this*m.c2}; }

    Mat3    scale(float s0, float s1, float s2) const  { return {c0*s0, c1*s1, c2*s2}; }
    float   get(size_t col, size_t row) const  { return (&c0.x)[col*3+row]; }
    float   det()       const { return c0.x*(c1.y*c2.z - c2.y*c1.z) - c1.x*(c0.y*c2.z - c2.y*c0.z) + c2.x*(c0.y*c1.z - c1.y*c0.z); }
    Mat3    transpose() const { return { c0.x,c1.x,c2.x, c0.y,c1.y,c2.y, c0.z,c1.z,c2.z, }; }

    Mat3 orthonormalized() const
    {
        Mat3 m;
        m.c0 = c0.normalized();
        m.c1 = c1.clip(m.c0).normalized();
        m.c2 = c2.clip(m.c0).clip(m.c1).normalized();
        return m;
    }

    static Mat3 identity()        { return {{1, 0, 0}, {0, 1, 0}, {0, 0, 1}}; }
    static Mat3 rotY(float angle) { float s = sinf(angle); float c = cosf(angle); return { c,0,-s, 0,1,0, s,0,c, }; }
    static Mat3 rotX(float angle) { float s = sinf(angle); float c = cosf(angle); return { 1,  0, 0, 0,  c, s, 0, -s, c, }; }
    static Mat3 rotZ(float angle) { float s = sinf(angle); float c = cosf(angle); return { c, -s, 0, s,  c, 0, 0,  0, 1, }; } 

    static Mat3 basis(Vec3 n)
    {
        n = n.normalized();
        Mat3 b;
        if(fabsf(n.x) > 0.577735027f)
        {
            b.c0 ={n.y, -n.x, 0};
        }
        else
        {
            b.c0 ={0, n.z, -n.y};
        }

        b.c0 = b.c0.normalized();
        b.c1 = n.cross(b.c0);
        b.c2 = n;

        return b;
    }

};

struct Mat4 
{ 
    Vec4 c0,c1,c2,c3; 

    static Mat4 identity() { return {{1,0,0,0},{0,1,0,0},{0,0,1,0},{0,0,0,1}}; }

    Vec4 mul(Vec4 v) const 
    {
        return {
            c0.x*v.x + c1.x*v.y + c2.x*v.z + c3.x*v.w,
            c0.y*v.x + c1.y*v.y + c2.y*v.z + c3.y*v.w,
            c0.z*v.x + c1.z*v.y + c2.z*v.z + c3.z*v.w,
            c0.w*v.x + c1.w*v.y + c2.w*v.z + c3.w*v.w,
        };
    }

    Mat4 mul(Mat4 m) const
    {
        return {mul(m.c0), mul(m.c1), mul(m.c2), mul(m.c3)};
    }

    static Mat4 setInvTR(Vec3 t, Mat3 r)
    {
        return {
            r.c0.x, r.c1.x, r.c2.x, 0,
            r.c0.y, r.c1.y, r.c2.y, 0,
            r.c0.z, r.c1.z, r.c2.z, 0,
            -r.c0.dot(t), -r.c1.dot(t), -r.c2.dot(t), 1
        };
    }

    static Mat4 setTR(Vec3 t, Mat3 r)
    {
        return {
            r.c0.x, r.c0.y, r.c0.z, 0,
            r.c1.x, r.c1.y, r.c1.z, 0,
            r.c2.x, r.c2.y, r.c2.z, 0,
            t.x, t.y, t.z, 1
        };
    }

    static Mat4 orthoLH(float l, float r, float t, float b, float n, float f)
    {
        // map z from [n,f] to [-1,1]
        // solve (A*n+B) = -1
        //       (A*f+B) =  1
        float A = 2/(f-n);
        float B = -(f+n)/(f-n);
        return {
            2/(r-l), 0, 0, 0,
            0, 2/(t-b), 0, 0,
            0, 0, A, 0,
            (l+r)/(l-r), (t+b)/(b-t), B, 1,
        };
    }

    static Mat4 perspectiveLH(float fovdeg, float aspectRatio, float n, float f)
    {
        float yScale = 1.0f/tanf(Math::DEG2RAD*fovdeg*0.5f);
        float xScale = yScale*aspectRatio;

        // map z from [n,f] to [-1,1]
        // solve (A*n+B)/n = -1
        //       (A*f+B)/f =  1

        float A = (f+n)/(f-n);
        float B = 2*f*n/(n-f);
        return {
            xScale, 0, 0, 0,
            0, yScale, 0, 0,
            0, 0, A, 1,
            0, 0, B, 0,
        };
    }
};

struct Quat 
{ 
    float x, y, z, w; 

    Quat mul(Quat q) const
    {
        return {
            x * q.w + y * q.z - z * q.y + w * q.x,
            -x * q.z + y * q.w + z * q.x + w * q.y,
            x * q.y - y * q.x + z * q.w + w * q.z,
            -x * q.x - y * q.y - z * q.z + w * q.w,
        };
    }

    // Assumes quaternion is unit length
    Vec3 rotate(Vec3 v) const
    {
        Vec3 r = {x, y, z};
        return v + r.cross(v*w + r.cross(v))*2.0f;
    }


    Quat conj() const
    {
        return {-x, -y, -z, w};
    }

    Mat3 toMat() const
    {
        float a  = x*x;
        float b  = y*y;
        float c  = z*z;
        float d  = w*w;
        float t1 = x*y;
        float t2 = z*w;
        float t3 = x*z;
        float t4 = y*w;
        float t5 = y*z;
        float t6 = x*w;
        float invs = 1 / (a + b + c + d);
        return {
            (a-b-c+d)*invs, 2*(t1 + t2)*invs, 2*(t3 - t4)*invs,
            2*(t1 - t2)*invs, (-a+b-c+d)*invs, 2*(t5 + t6)*invs,
            2*(t3 + t4)*invs, 2*(t5 - t6)*invs, (-a-b+c+d)*invs,
        };
    }


    static Quat fromAxisAngle(Vec3 axis, float angle)
    {
        axis = axis.normalized();
        float s = sinf(angle/2);
        float c = cosf(angle/2);
        return {s*axis.x, s*axis.y, s*axis.z, c};
    }
};

struct Transform 
{ 
    Mat3 R; 
    Vec3 p; 

    static Transform identity() { return {{{1, 0, 0}, {0, 1, 0}, {0, 0, 1}}, {0,0,0}}; }

    static Transform composeNeg(Transform a, Transform b)
    {
        Mat3 negInvRA =  a.R.transpose()*-1.0f;
        Transform res;
        res.R = negInvRA*b.R;
        res.p = negInvRA*(b.p - a.p);
        return res;
    }

    static Transform lerp(float t, Transform xf0, Transform xf1)
    {
        Transform xf;
        xf.R.c0 = Vec3::lerp(t, xf0.R.c0, xf1.R.c0);
        xf.R.c1 = Vec3::lerp(t, xf0.R.c1, xf1.R.c1);
        xf.R.c2 = Vec3::lerp(t, xf0.R.c2, xf1.R.c2);
        xf.R = xf.R.orthonormalized();
        xf.p = Vec3::lerp(t, xf0.p, xf1.p);
        return xf;
    }

    Transform inverse() const
    {
        Mat3 iR = R.transpose();
        return {iR, iR* -p};
    }

    Transform mul(Transform x) const
    {
        // aR*(bR*v + bp) + ap
        // aR*bR*v + aR*bp + ap
        Mat3 R1 = R*x.R;
        return {R1, R*x.p + p};
    }

    Vec3 mul(Vec3 v) const
    {
        return R*v + p;
    }

    Vec3 rotate(Vec3 n) const
    {
        return R*n;
    }

    Transform negate() const
    {
        return {R*-1, -p};
    }
};

struct Rect 
{ 
    float x0,y0,x1,y1; 
};

struct Aabb
{
    float min[4];
    float max[4];

    Vec3 getMin() const { return *(Vec3*)min; }
    Vec3 getMax() const { return *(Vec3*)max; }

    void expandByRadius(float r)
    {
        min[0] -= r;
        min[1] -= r;
        min[2] -= r;
        max[0] += r;
        max[1] += r;
        max[2] += r;
    }

    Aabb expand(Vec3 point) const
    {
        __m128 p = _mm_loadu_ps(&point.x);
        __m128 minA = _mm_loadu_ps(min);
        __m128 maxA = _mm_loadu_ps(max);
        __m128 minU = _mm_min_ps(minA, p);
        __m128 maxU = _mm_max_ps(maxA, p);
        Aabb u;
        _mm_storeu_ps(u.min, minU);
        _mm_storeu_ps(u.max, maxU);
        return u;
    }

    Aabb expand(const Aabb other) const
    {
        __m128 minA = _mm_loadu_ps(min);
        __m128 maxA = _mm_loadu_ps(max);
        __m128 minB = _mm_loadu_ps(other.min);
        __m128 maxB = _mm_loadu_ps(other.max);
        __m128 minU = _mm_min_ps(minA, minB);
        __m128 maxU = _mm_max_ps(maxA, maxB);
        Aabb u;
        _mm_storeu_ps(u.min, minU);
        _mm_storeu_ps(u.max, maxU);
        return u;
    }

    float getMaxExtent() const
    {
        __m128 d = _mm_sub_ps(_mm_loadu_ps(max), _mm_loadu_ps(min));
        return Math::max(Math::max(d.m128_f32[0], d.m128_f32[1]), d.m128_f32[2]);
    }

    float getMaxExtent(size_t &axisOut) const
    {
        __m128 d = _mm_sub_ps(_mm_loadu_ps(max), _mm_loadu_ps(min));
        size_t axis = 0;
        for(size_t i = 1; i < 3; i++)
        {
            if(d.m128_f32[i] > d.m128_f32[axis])
                axis = i;
        }
        axisOut = axis;
        return d.m128_f32[axis];
    }

    bool overlaps(Aabb other) const
    {
        return !(min[0] > other.max[0] || max[0] < other.min[0] ||
                 min[1] > other.max[1] || max[1] < other.min[1] ||
                 min[2] > other.max[2] || max[2] < other.min[2]);
    }

    static Aabb empty()                     { return {{ INFINITY,  INFINITY,  INFINITY, INFINITY}, {-INFINITY, -INFINITY, -INFINITY, -INFINITY}}; };
    static Aabb fromRadius(Vec3 p, float r) { return {{p.x-r, p.y-r, p.z-r}, {p.x+r, p.y+r, p.z+r}}; }
};

struct Plane
{
    Vec3 normal;
    float dist;
};

inline Vec3 triComputeNormal(Vec3 a, Vec3 b, Vec3 c)        { return (b-a).cross(c-b).normalized(); }
inline Vec3 triComputeCentroid(Vec3 a, Vec3 b, Vec3 c)      { return (a+b+c)*(1.0f/3.0f); }

struct Bvh
{
    union Obj
    {
        Aabb bounds;
        struct
        {
            float min[3];
            union
            {
                uint32_t bin;
                float count;
            };
            float max[3];
            uint32_t id;
        };
    };

    struct Node
    {
        Aabb bounds;
        uint32_t childStart;
        uint8_t numObjs;
    };

    Obj *m_objs;
    Node *m_nodes;
    size_t m_numNodes;
    size_t m_numObjs;
    size_t m_maxObjs;

    void buildFromTriangles(Vec3 verts[], uint32_t indices[], size_t numTris);
    void buildFromAabbs(Aabb aabbs[], size_t numAabbs);
    void destroy();

    size_t overlap(Aabb bounds, uint32_t outObjs[], size_t maxObjs) const;
};

struct ConvexHull
{
    struct Face
    {
        uint16_t firstEdge;
        uint16_t numEdges;
        Vec3 normal;
        Vec3 centroid;
        float dist;
    };

    struct Edge
    {
        uint16_t v0;
        uint16_t v1;
        uint16_t f0;
        uint16_t f1;
    };

    Face *m_faces;
    size_t m_numFaces;

    Edge *m_edges;
    size_t m_numEdges;

    Vec3 *m_verts;
    size_t m_numVerts;

    void build(const Vec3 inputVerts[], const size_t numVerts);
    void destroy();

    static void test();
};

struct Allocator
{
    virtual void *alloc(size_t size) = 0;
    virtual void free(void *pointer) = 0;
};

struct SatShape
{
    struct EdgeList 
    { 
        uint8_t first, num; 
    };

    struct Edge 
    { 
        uint8_t v0, v1;
        uint8_t f0, f1; 
    };

    uint8_t numFaces;
    uint8_t numVerts;
    uint8_t numEdges;

    EdgeList   *faces;             // numFaces
    EdgeList   *verts;             // numVerts
    Vec3       *vertPos;           // numVerts
    Plane      *facePlanes;        // numFaces
    Edge       *faceEdges;         // numEdges*2
    Edge       *vertEdges;         // numEdges*2
    Edge       *edges;             // numEdges

    float boundingRadius;
    Mat3 invInertia;
    float invMass;
    Vec3 seg0, seg1;

    static SatShape *createFromVerts(Allocator *allocator, Vec3 verts[], size_t numVerts);
};

struct TriShapeData
{
    SatShape::EdgeList faces[2];
    SatShape::EdgeList verts[3];
    SatShape::Edge edges[3];
    SatShape::Edge faceEdges[6];
    SatShape::Edge vertEdges[6];
};

struct TriShape 
{
    TriShapeData data;
    Plane facePlanes[2];
    Vec3 vertPos[4];
    SatShape sat;
    void setFromVerts(const Vec3 verts[3]);
};

struct SatResult
{
    enum Type 
    {
        NONE,
        FACE_VERT,
        VERT_FACE,
        EDGE_EDGE,
    };

    Type type;

    Vec3 mtv;
    float support;


    size_t face;
    size_t vert;

    SatShape::Edge edge1;
    SatShape::Edge edge2;
};

void satCollideGraph(const SatShape *a, Transform xfA, const SatShape *b, Transform xfB, SatResult *res);
void satCollideReference(const SatShape *a, Transform xfA, const SatShape *b, Transform xfB, SatResult *res);
void getSupport(Vec3 dir,  const Vec3 verts[], size_t numVerts, float *outSupp, size_t *outIndex);
float castSphereRay(const SatShape *shape, size_t curVertA, Vec3 rayOrigin, Vec3 rayDir, SatShape::Edge *outEdge);

bool raycastSatShape(Vec3 rayOrigin, Vec3 rayDir, const SatShape *shape, Transform shapeTransform, float *tOut, Vec3 *hitOut);

const size_t MAX_CONTACTS = 4;

const size_t SOLVER_MAX_BODIES = 4096;
const size_t SOLVER_MAX_MANIFOLDS = (32*SOLVER_MAX_BODIES);
const float SOLVER_DELTA_SLOP = -0.02f;
const float SOLVER_BETA = 0.4f;
const float SOLVER_DT = (1.0f/60.0f);

struct SolverContact
{
    Vec3 p;
    float bias;
    float normalLambda;
    float tangentLambda;
    float bitangentLambda;
};

struct SolverManifold
{
    Vec3 normal, tangent, bitangent;
    SolverContact contacts[MAX_CONTACTS];
    size_t numContacts;
    uint16_t indexA;
    uint16_t indexB;
};

bool generateSatVsSatContacts(const SatShape *shapeA, Transform xfA, size_t indexA,
                             const SatShape *shapeB, Transform xfB, size_t indexB,
                             SolverManifold *manifold, bool useGraph = false);


struct TriMeshShape
{
    struct TriData
    {
        Vec3 verts[3];
        Vec3 edgeNormals[3];
        Vec3 vertNormals[3];
    };

    TriData *tris;
    size_t numTris;
    Bvh bvh;
};

bool loadTriMeshShape(TriMeshShape *shape, const char *filename);
size_t generateTriMeshVsSatContacts(const TriMeshShape  *triMesh, Transform xfA, size_t indexA, const SatShape*shapeB, Transform xfB, size_t indexB, SolverManifold *manifolds, bool useGraph = false);

struct SolverBody
{
    Vec3 w;
    Vec3 v;
    float iM;
    Mat3 iI;
    Vec3 p;
};

void solveConstraints(SolverManifold manifolds[], size_t numManifolds, SolverBody bodies[], size_t numBodies);


bool intersectSphere(Vec3 dir, Vec3 orig, Vec3 center, float radius, float *t_out);


void ods(const char* fmt, ...);
double getTime();


#define IM_VEC2_CLASS_EXTRA operator Vec2() const { return {x, y}; }
#include "imgui\imgui.h"

typedef unsigned int Color;
typedef void *DebugMeshId;

void getPickRay(Vec3 &origin, Vec3 &dir);
void setCamera(Mat3 rot, Vec3 pos);

void _drawTriVert(Vec3 p, Vec3 n, Color color);
void _drawLineVert(Vec3 p, Color color);
void _drawQuad(Vec3 a, Vec3 b, Vec3 c, Vec3 d, Vec3 n, Color color);
void _drawLine(Vec3 a, Vec3 b, Color color);
void _drawTri(Vec3 a, Vec3 b, Vec3 c, Color color);
void _drawWireTri(Vec3 a, Vec3 b, Vec3 c, Color color);
void drawBox(Vec3 a, Vec3 b, Color color);
void drawWireBox(Vec3 a, Vec3 b, Color color);
void drawAabb(Aabb aabb, Color color);
void drawLine(Vec3 a, Vec3 b, Color color);
void drawWireSphere(Vec3 p, float radius, Color color);
void drawSphere(Vec3 origin, float radius, Color color, size_t numLon, size_t numLat);
void drawWireCylinder(Vec3 a, Vec3 b, float eX, float eY, Color color);
void drawArrow(Vec3 origin, Vec3 arrow, Color color, float scale);
void drawArrowTo(Vec3 a, Vec3 b, Color color, float scale);
void drawArcBetween(Vec3 origin, Vec3 a, Vec3 b, float radius, Color color, bool directed);
void drawWireCone(Vec3 a, Vec3 b, float eX, float eY, Color color);
void drawCone(Vec3 a, Vec3 b, float eX, float eY, Color color);
void drawWireCapsule(Vec3 a, Vec3 b, float radius, Color color);
void drawCapsule(Vec3 a, Vec3 b, float radius, Color color);
void drawCoordinateSystem(Vec3 origin, float scale);
void drawPoint(Vec3 origin, Color color);
void drawPointEx(Vec3 origin, Color color, float scale);
void drawPlane(Vec3 origin, Vec3 normal, Color color, float scale);
void drawDisk(Vec3 origin, Vec3 xAxis, Vec3 yAxis, float eX, float eY);
void drawCircle(Vec3 origin, Vec3 normal, float extent);
void drawOrientedWireBox(Vec3 origin, Vec3 xAxis, Vec3 yAxis, Vec3 zAxis, Vec3 extent);
void drawDemo(void);
void drawMesh(DebugMeshId meshId, Transform transform, Color color);
void createDebugMesh(DebugMeshId id, const Vec3 vertices[], size_t numVertices, const uint32_t indices[], size_t numIndices);
void destroyDebugMesh(DebugMeshId id);

const Color COLOR_BLACK       = 0xff000000;
const Color COLOR_GREY        = 0xff444444;
const Color COLOR_WHITE       = 0xffffffff;
const Color COLOR_BLUE        = 0xffff0000;
const Color COLOR_GREEN       = 0xff00ff00;
const Color COLOR_RED         = 0xff0000ff;
const Color COLOR_YELLOW      = 0xff00ffff;
const Color COLOR_ORANGE      = 0xff0088ff;
const Color COLOR_PURPLE      = 0xffff00ff;
const Color COLOR_LIGHT_BLUE  = 0xffe6d8ad;
const Color COLOR_LIGHT_GREEN = 0xff90EE90;

enum 
{
    KEY_LEFT        = 0x25,
    KEY_RIGHT       = 0x27,
    KEY_SPACE       = 0x20,
    KEY_ESCAPE      = 0x1B,
    KEY_LBUTTON     = 0x01,
    KEY_RBUTTON     = 0x02,
};

// Special drawing functions
extern Vec3 g_sphereOrigin;
void drawBvh(Bvh *bvh);
void drawConvexity(const TriMeshShape *shape);
void drawGaussRegion(size_t vertex, const SatShape *shape, Transform xf, Color color);
void drawGaussMap(const SatShape *shape, Transform xf, Color color);
void drawGaussPoint(Vec3 p, Color color);
void drawSatShape(const SatShape * shape, Transform transform, Color color);

