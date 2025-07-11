
#include "collision.h"

#include <immintrin.h>
#include <string.h>
#include <stdint.h>
#include <stdarg.h>

#include <vector>
#include <unordered_map>
#include <vector>

#include "tracy\Tracy.hpp"


char  *readFile(const char *filename, size_t &outSize)
{
    FILE *file = NULL;
    fopen_s(&file, filename, "rb");
    ASSERT(file);
    fseek(file, SEEK_SET, SEEK_END);
    size_t size = ftell(file);
    rewind(file);
    void *bytes = malloc(size);
    ASSERT(bytes);
    ASSERT(fread(bytes, size, 1, file) == 1);
    outSize = size;
    return (char*)bytes;
}

struct LARGE_INTEGER {int64_t QuadPart;};
extern "C" bool __stdcall QueryPerformanceFrequency(LARGE_INTEGER *lpFrequency);
extern "C" bool __stdcall QueryPerformanceCounter(LARGE_INTEGER *lpPerformanceCount);
extern "C" void __stdcall OutputDebugStringA(const char * lpOutputString);

double getTime()
{
    static LARGE_INTEGER freq;
    if (!freq.QuadPart)
    {
        QueryPerformanceFrequency(&freq);
    }
    LARGE_INTEGER count;
    QueryPerformanceCounter(&count);
    return count.QuadPart/(double)freq.QuadPart;
}

void ods(const char* fmt, ...)
{
    va_list args;
    va_start(args, fmt);
    static char buf[2048];
    vsnprintf(buf,sizeof(buf),fmt,args);
    OutputDebugStringA(buf);
    va_end(args);
}


static const unsigned char mask128_epi32[] =
{
    0x0,0x1,0x2,0x3,0x4,0x5,0x6,0x7,0x8,0x9,0xa,0xb,0xc,0xd,0xe,0xf,0x4,0x5,0x6,0x7,0x8,0x9,0xa,0xb,0xc,0xd,0xe,0xf,0xc,0xd,0xe,0xf,
    0x0,0x1,0x2,0x3,0x8,0x9,0xa,0xb,0xc,0xd,0xe,0xf,0xc,0xd,0xe,0xf,0x8,0x9,0xa,0xb,0xc,0xd,0xe,0xf,0xc,0xd,0xe,0xf,0xc,0xd,0xe,0xf,
    0x0,0x1,0x2,0x3,0x4,0x5,0x6,0x7,0xc,0xd,0xe,0xf,0xc,0xd,0xe,0xf,0x4,0x5,0x6,0x7,0xc,0xd,0xe,0xf,0xc,0xd,0xe,0xf,0xc,0xd,0xe,0xf,
    0x0,0x1,0x2,0x3,0xc,0xd,0xe,0xf,0xc,0xd,0xe,0xf,0xc,0xd,0xe,0xf,0xc,0xd,0xe,0xf,0xc,0xd,0xe,0xf,0xc,0xd,0xe,0xf,0xc,0xd,0xe,0xf,
    0x0,0x1,0x2,0x3,0x4,0x5,0x6,0x7,0x8,0x9,0xa,0xb,0x8,0x9,0xa,0xb,0x4,0x5,0x6,0x7,0x8,0x9,0xa,0xb,0x8,0x9,0xa,0xb,0x8,0x9,0xa,0xb,
    0x0,0x1,0x2,0x3,0x8,0x9,0xa,0xb,0x8,0x9,0xa,0xb,0x8,0x9,0xa,0xb,0x8,0x9,0xa,0xb,0x8,0x9,0xa,0xb,0x8,0x9,0xa,0xb,0x8,0x9,0xa,0xb,
    0x0,0x1,0x2,0x3,0x4,0x5,0x6,0x7,0x4,0x5,0x6,0x7,0x4,0x5,0x6,0x7,0x4,0x5,0x6,0x7,0x4,0x5,0x6,0x7,0x4,0x5,0x6,0x7,0x4,0x5,0x6,0x7,
    0x0,0x1,0x2,0x3,0x0,0x1,0x2,0x3,0x0,0x1,0x2,0x3,0x0,0x1,0x2,0x3,0xff,0xff,0xff,0xff,0xff,0xff,0xff,0xff,0xff,0xff,0xff,0xff,0xff,0xff,0xff,0xff,
};

inline __m128i prune_epi32(__m128i x, int mask) 
{
  return _mm_shuffle_epi8(x, _mm_loadu_si128((const __m128i *)mask128_epi32 + mask));
}

inline __m128 dot4ps(const __m128 &ax, const __m128 &ay, const __m128 &az, const __m128 &bx, const __m128 &by, const __m128 &bz)
{
    //return _mm_fmadd_ps(ax,bx, _mm_fmadd_ps(ay,by, _mm_mul_ps(az, bz)))
    return _mm_add_ps(_mm_mul_ps(ax, bx), _mm_add_ps(_mm_mul_ps(ay, by), _mm_mul_ps(az, bz)));
}

inline __m128i blendvepi32(const __m128i &a, const __m128i &b, const __m128 &mask)
{
    return _mm_castps_si128(_mm_blendv_ps(_mm_castsi128_ps(a), _mm_castsi128_ps(b), mask));
}

inline __m128 horizontalMin(__m128 x) 
{
    __m128 a = _mm_shuffle_ps(x, x, _MM_SHUFFLE(2,3,0,1));
    __m128 b = _mm_min_ps(x, a);
    __m128 c = _mm_shuffle_ps(b, b, _MM_SHUFFLE(0,0,2,2));
    __m128 d = _mm_min_ps(b, c);
    return d;
}

inline __m128 horizontalMax(__m128 x) 
{
    __m128 a = _mm_shuffle_ps(x, x, _MM_SHUFFLE(2,3,0,1));
    __m128 b = _mm_max_ps(x, a);
    __m128 c = _mm_shuffle_ps(b, b, _MM_SHUFFLE(0,0,2,2));
    __m128 d = _mm_max_ps(b, c);
    return d;
}

inline int horizontalArgmax(__m128 values, __m128i indices, float *max_out)
{
    __m128 max_splat = horizontalMax(values);
    __m128i pruned_indices = prune_epi32(indices, _mm_movemask_ps(_mm_cmplt_ps(values, max_splat)));
    *max_out = _mm_cvtss_f32(max_splat);
    int index = _mm_cvtsi128_si32(pruned_indices);
    return index;
}

inline int horizontalArgmin(__m128 values, __m128i indices, float *min_out)
{
    __m128 min_splat = horizontalMin(values);
    __m128i pruned_indices = prune_epi32(indices, _mm_movemask_ps(_mm_cmpgt_ps(values, min_splat)));
    *min_out = _mm_cvtss_f32(min_splat);
    int index = _mm_cvtsi128_si32(pruned_indices);
    return index;
}

inline void transposeVec3(const Vec3 *v, __m128 &X, __m128 &Y, __m128 &Z) 
{
    __m128 _tmp0 = _mm_loadu_ps(((float*)(v))+0);
    __m128 _tmp1 = _mm_loadu_ps(((float*)(v))+4);
    __m128 _tmp2 = _mm_loadu_ps(((float*)(v))+8);
    X = _mm_shuffle_ps(_tmp0, _mm_shuffle_ps(_tmp1, _tmp2, _MM_SHUFFLE(1, 1, 2, 2)), _MM_SHUFFLE(3, 1, 3, 0));
    Y = _mm_shuffle_ps(_tmp1, _mm_shuffle_ps(_tmp2, _tmp0, _MM_SHUFFLE(1, 1, 2, 2)), _MM_SHUFFLE(3, 1, 3, 0));
    Z = _mm_shuffle_ps(_tmp2, _mm_shuffle_ps(_tmp0, _tmp1, _MM_SHUFFLE(1, 1, 2, 2)), _MM_SHUFFLE(3, 1, 3, 0));
    Y = _mm_shuffle_ps(Y, Y, _MM_SHUFFLE(2,1,0,3));
    Z = _mm_shuffle_ps(Z, Z, _MM_SHUFFLE(1,0,3,2));
}

union M128I_BYTE16 
{
    char b[16];
    __m128i value;
};

union M128_FLOAT4
{
    float f[4];
    __m128 value;
};

union M128_INT4
{
    int i[4];
    __m128i value;
};

const M128_FLOAT4 FLOAT4_INF    = {{INFINITY, INFINITY, INFINITY, INFINITY,}};
const M128_FLOAT4 FLOAT4_NEGINF = {{-INFINITY, -INFINITY, -INFINITY, -INFINITY,}};
const M128_INT4 INT4_0123       = {{0, 1, 2, 3}};
const M128_INT4 INT4_0000       = {{0, 0, 0, 0}};
const M128_INT4 INT4_1111       = {{1, 1, 1, 1}};
const M128_INT4 INT4_4444       = {{4, 4, 4, 4}};

inline size_t clipPolygonAgainstPlane(Vec3 plane_n, float plane_d, Vec3 verts[], size_t numVerts, Vec3 clipped[])
{
    ZoneScoped;

    size_t numClipped = 0;
    for(size_t k1 = 0, k0 = numVerts-1; k1 < numVerts; k0 = k1, k1++)
    {
        Vec3 p0 = verts[k0];
        Vec3 p1 = verts[k1];

        float dist0 = plane_n.dot(p0) - plane_d;
        float dist1 = plane_n.dot(p1) - plane_d;

        if(dist0 <= 0)
        {
            clipped[numClipped++] = p0;
        }

        if(dist0 * dist1 < 0)
        {
            Vec3 dp =(p1 - p0);
            float den = plane_n.dot(dp);
            if (den == 0) continue;
            float t = -(plane_n.dot(p0) - plane_d) / den;
            Vec3 hit = p0 + dp * t;
            clipped[numClipped++] = hit;
        }
    }
    return numClipped;
}



#ifdef DEBUG_INTERACTIVE 
#define DEBUG_LAST(...)             do { if (debugIsLastStep()) { __VA_ARGS__; } } while (0)
#define DEBUG_QUIT()                do { throw -1; } while (0)
#define DEBUG_NEXT()                do { if (++g_debugCurrentStep > g_debugLastStep) longjmp(g_debugJmpBuf, 1); } while (0)
#define DEBUG_TRY(...)              try { g_debugCurrentStep = 0; __VA_ARGS__; } catch (...) { }
#define DEBUG_ANY(...)              do { __VA_ARGS__; } while (0)
#define DEBUG_EXACTLY(step, ...)    do { if (g_debugCurrentStep == (step)) { __VA_ARGS__; } } while (0)
#define DEBUG_EXCLUDE(...)          do {int _tmp = g_debugLastStep; g_debugLastStep = 10000000; __VA_ARGS__; g_debugLastStep = _tmp; g_debugCurrentStep = 0; } while (0)
static int g_debugLastStep = 0;
static int g_debugCurrentStep;
static jmp_buf g_debugJmpBuf;
static bool debugIsLastStep()
{
    if (g_debugCurrentStep == g_debugLastStep)
    {
        return true;
    }
    return false;
}
#else
#define DEBUG_EXACTLY(step, ...)  
#define DEBUG_QUIT()
#define DEBUG_LAST(...) 
#define DEBUG_NEXT() 
#define DEBUG_START(...)
#define DEBUG_ANY(...)
#define DEBUG_EXCLUDE(...) do {__VA_ARGS__;} while (0)
#endif

const size_t BVH_NUM_BINS = 8;
const size_t BVH_NUM_SPLITS = (BVH_NUM_BINS-1);
const size_t BVH_MAX_OBJS_IN_LEAF = 4;

#define SHUF0 0,1,2,3
#define SHUF1 4,5,6,7
#define SHUF2 8,9,10,11
#define SHUF3 12,13,14,15

static const unsigned char extractSplitMask[] ={
    SHUF0, SHUF1, SHUF2, SHUF3,
    SHUF1, SHUF0, SHUF2, SHUF3,
    SHUF2, SHUF0, SHUF1, SHUF3,
};

static const unsigned char moveToBinMask[] ={
    SHUF3, SHUF1, SHUF2, SHUF0,
    SHUF0, SHUF3, SHUF2, SHUF1,
    SHUF0, SHUF1, SHUF3, SHUF2,
};

#define INC_BIN _mm_setr_ps(0,0,0,1)

static inline void computeBins(Bvh::Obj bins[], Bvh::Obj objs[], size_t numObjs, float splitExtent, float splitMin, size_t splitAxis)
{
    const __m128 A = _mm_set1_ps(0.5f*BVH_NUM_BINS/splitExtent);
    const __m128 C = _mm_set1_ps(-2.0f*splitMin);
    const __m128i M1 = _mm_loadu_si128((__m128i*)(extractSplitMask+16*splitAxis));
    const __m128i M2 = _mm_loadu_si128((__m128i*)(moveToBinMask+16*splitAxis));
    // A*((min+max)+C) 
    // = 0.5f*Nb/E*((min+max)-2*Smin)
    // = (0.5f*(min+max)-SMin)/E * Nb
    for (size_t i = 0; i < numObjs; i++) 
    {
        __m128 objMin = _mm_loadu_ps(objs[i].bounds.min);
        __m128 objMax = _mm_loadu_ps(objs[i].bounds.max);
        __m128 t = _mm_mul_ps(A, _mm_add_ps(_mm_add_ps(objMin, objMax), C));
        __m128i b = _mm_cvttps_epi32(t);
        __m128 updated = _mm_blend_ps(objMin, _mm_castsi128_ps(_mm_shuffle_epi8(b, M2)), 0b1000);
        _mm_storeu_ps(objs[i].bounds.min, updated);

        // TODO: requres shl bin,5
        uint64_t bin = _mm_cvtsi128_si32 (_mm_shuffle_epi8(b, M1)); 

        ASSERT(bin < BVH_NUM_BINS);

        __m128 binMin = _mm_loadu_ps(bins[bin].bounds.min);
        __m128 binMax = _mm_loadu_ps(bins[bin].bounds.max);

        __m128 binInc = _mm_add_ps(binMin, INC_BIN);

        __m128 newBinMin = _mm_min_ps(binMin, objMin);
        __m128 newBinMax = _mm_max_ps(binMax, objMax);

        newBinMin = _mm_blend_ps(newBinMin, binInc, 0b1000);

        _mm_storeu_ps(bins[bin].bounds.min,  newBinMin);
        _mm_storeu_ps(bins[bin].bounds.max,  newBinMax);
    }
}

static inline Aabb computeTotalBounds(Bvh::Obj *objs, size_t numObjs)
{
    Aabb totalBounds = Aabb::empty();
    for(size_t i = 0; i < numObjs; i++)
    {
        totalBounds = totalBounds.expand(objs[i].bounds);
    }
    totalBounds.expandByRadius(0.001f);
    return totalBounds;

}

size_t partitionBounds(Bvh::Obj objs[], size_t numObjs, size_t midBin)
{
    __m128i* ptrJ = (__m128i*)objs;
    __m128i* ptrI = (__m128i*)objs;
    size_t j = 0;
    for(size_t i = 0; i < numObjs; i++)
    {
        if(objs[i].bin < midBin)
        {   
            // Should just be able to swap Bvh::Objs but some compilers seem to generate extra copies
            // Not sure why
            __m128i objMinI = _mm_loadu_si128(ptrI+2*i+0);     
            __m128i objMaxI = _mm_loadu_si128(ptrI+2*i+1);

            __m128i objMinJ = _mm_loadu_si128(ptrJ+2*j+0);
            __m128i objMaxJ = _mm_loadu_si128(ptrJ+2*j+1);

            _mm_storeu_si128(ptrI+2*i+0, objMinJ);
            _mm_storeu_si128(ptrI+2*i+1, objMaxJ);
            
            _mm_storeu_si128(ptrJ+2*j+0, objMinI);
            _mm_storeu_si128(ptrJ+2*j+1, objMaxI);

            j++;
        }
    }
    return j;
}

// Doesn't help much but it's cool :)
static inline Bvh::Obj addBins(Bvh::Obj a, Bvh::Obj b)
{
    __m128 minA = _mm_loadu_ps(a.bounds.min);
    __m128 maxA = _mm_loadu_ps(a.bounds.max);
    __m128 minB = _mm_loadu_ps(b.bounds.min);
    __m128 maxB = _mm_loadu_ps(b.bounds.max);

    __m128 minU = _mm_min_ps(minA, minB);
    __m128 maxU = _mm_max_ps(maxA, maxB);

    __m128 countSum = _mm_add_ps(minA, minB);
    minU = _mm_blend_ps(minU, countSum, 0b1000);

    Bvh::Obj u;
    _mm_storeu_ps(u.bounds.min, minU);
    _mm_storeu_ps(u.bounds.max, maxU);

    return u;
}


static const M128I_BYTE16 XYZW_TO_YZXW  = {{SHUF1, SHUF2, SHUF0, SHUF3}}; 
static const M128I_BYTE16 BROADCAST_W   = {{SHUF3, SHUF3, SHUF3, SHUF3}};

#if 0
static inline float binCost(Bvh::Obj bin)
{
    float x = bin.max[0] - bin.min[0];
    float y = bin.max[1] - bin.min[1];
    float z = bin.max[2] - bin.min[2];
    float w = bin.count;
    return (x*y + y*z + z*x)*w;
}

#else 
static inline float binCost(Bvh::Obj bin)
{
    // (x,y,z) = max-min
    // w = count
    __m128 min = _mm_loadu_ps(bin.bounds.min);
    __m128 max = _mm_loadu_ps(bin.bounds.max);
    __m128 xyzw = _mm_sub_ps(max,min);
    __m128 yzxw = _mm_castsi128_ps(_mm_shuffle_epi8(_mm_castps_si128(xyzw), XYZW_TO_YZXW.value));
    __m128 wwww = _mm_castsi128_ps(_mm_shuffle_epi8(_mm_castps_si128(min), BROADCAST_W.value));
    // v = (x*y + y*z + z*x)*w
    __m128 v = _mm_mul_ps(_mm_mul_ps(xyzw, yzxw), wwww);

    // horizontal sum  first 3 components of v
    __m128 t1 = _mm_movehl_ps(v, v);
    __m128 t2 = _mm_add_ps(v, t1);
    __m128 sum = _mm_add_ss(t2, _mm_shuffle_ps(v, v, _MM_SHUFFLE(1,1,1,1)));
    float cost = _mm_cvtss_f32(sum);

    return cost;
}
#endif

struct BuildParams
{
    size_t objsBegin;
    size_t numObjs;
    size_t outNode;
};

static void buildBvhInternal(Bvh &bvh)
{
    ASSERT(bvh.m_objs && bvh.m_numObjs != -1);


#ifdef DEBUG_INTERACTIVE
    //g_debugLastStep = 0;
    if (keyPressed(PLATFORM_KEY_LEFT)) g_debugLastStep--;
    if (keyPressed(PLATFORM_KEY_RIGHT)) g_debugLastStep++;
    if (keyPressed(KEY_F)) g_debugLastStep = 844;
#endif

    size_t nodeCapacity = bvh.m_maxObjs*2;

    DEBUG_START({ return; });

    BuildParams paramStack[128];
    size_t paramStackSize = 0;

    paramStack[paramStackSize++] = { 0, bvh.m_numObjs, bvh.m_numNodes++};

    while (paramStackSize)
    {
        BuildParams params = paramStack[--paramStackSize];

        ASSERT(params.objsBegin != -1 && params.objsBegin < nodeCapacity);
        ASSERT(params.numObjs);

        const size_t objsBegin = params.objsBegin;
        const size_t objsEnd = objsBegin+params.numObjs;
        const size_t objsCount = params.numObjs;

        Aabb totalBounds = computeTotalBounds(bvh.m_objs+objsBegin, objsCount);

        DEBUG_LAST(drawAabb(totalBounds, ColorPurple));
        DEBUG_NEXT();


        if (objsCount == 1)
        {
            Bvh::Node *leaf = bvh.m_nodes+params.outNode;
            leaf->childStart = (uint32_t)objsBegin; 
            leaf->numObjs = 1;
            leaf->bounds = totalBounds;
        }
        else if (objsCount <= BVH_MAX_OBJS_IN_LEAF) 
        {
            Bvh::Node *leaf = bvh.m_nodes + params.outNode;
            leaf->childStart = (uint32_t)objsBegin;
            leaf->numObjs = (uint32_t)objsCount;
            leaf->bounds = totalBounds;
        }
        else 
        {
            size_t splitAxis;
            float splitExtent = totalBounds.getMaxExtent(splitAxis);
            float splitMin = totalBounds.min[splitAxis];
            float invSplitExtent = 1.0f/splitExtent;
            ASSERT(splitExtent);


            Bvh::Obj bins[BVH_NUM_BINS];
            for (size_t i = 0; i < BVH_NUM_BINS; i++) bins[i].bounds = Aabb::empty();

            computeBins(bins, bvh.m_objs+objsBegin, objsCount, splitExtent, splitMin, splitAxis);

            DEBUG_LAST(
                for (size_t i = 0; i < BVH_NUM_BINS; i++)
                {
                    drawAabb(bins[i].bounds, ColorRed);
                }
            );
            DEBUG_NEXT();

            Bvh::Obj below[BVH_NUM_SPLITS];
            Bvh::Obj above[BVH_NUM_SPLITS];
            below[0] = bins[0];
            for (size_t i = 1; i < BVH_NUM_SPLITS; i++) below[i] = addBins(below[i-1], bins[i]);
            above[BVH_NUM_SPLITS-1] = bins[BVH_NUM_BINS - 1];
            for (size_t i = BVH_NUM_SPLITS - 1; i > 0; i--) above[i] = addBins(above[i+1-1], bins[i+1-1]);

            // Find bin to split at that minimizes SAH metric
            size_t minCostSplitBin = -1;
            float minCost = INFINITY;
            for (size_t i = 0; i < BVH_NUM_SPLITS; ++i) 
            {
                // Compute cost for candidate split and update minimum if
                // necessary
                if (below[i].count == 0 || above[i].count == 0)
                {
                    continue;
                }

                float cost = binCost(below[i]) + binCost(above[i]);

                if (cost < minCost) 
                {
                    minCost = cost;
                    minCostSplitBin = i;
                }
            }

            if (minCostSplitBin == -1)
            {
                // Split objs arbitrarily in half
                // TODO: this can lead to an unbalanced tree when an obj is very large
                Bvh::Node *interior = bvh.m_nodes+params.outNode;
                interior->childStart = (uint32_t)bvh.m_numNodes;
                interior->numObjs = 0;
                interior->bounds = totalBounds;

                size_t objsMid = (objsBegin+objsEnd)/2;
                paramStack[paramStackSize++] = {
                    objsBegin,
                    objsMid-objsBegin,
                    bvh.m_numNodes++
                };

                paramStack[paramStackSize++] = {
                    objsMid,
                    objsEnd-objsMid,
                    bvh.m_numNodes++
                };

                continue;
            }

            ASSERT(minCostSplitBin != -1);

#ifdef DEBUG_INTERACTIVE
            {
                for(size_t i = 0; i < BVH_NUM_SPLITS; i++)
                {
                    DEBUG_LAST(
                        drawAabb(below[i].bounds, ColorLightBlue);
                        drawAabb(above[i].bounds, ColorLightGreen);
                    );
                    DEBUG_NEXT();
                }

                DEBUG_LAST(
                    drawAabb(below[minCostSplitBin].bounds, ColorOrange);
                    drawAabb(above[minCostSplitBin].bounds, ColorPurple);
                );
                DEBUG_NEXT();
            }
#endif

            // Partition objects in front and behind split
            size_t objsMid = objsBegin + partitionBounds(bvh.m_objs+objsBegin, objsCount, minCostSplitBin+1); 

#ifdef DEBUG_INTERACTIVE
            for(size_t i = objsBegin; i < objsMid; i++) ASSERT(bvh.objs[i].bin <= minCostSplitBin);
            for(size_t i = objsMid; i < objsEnd; i++) ASSERT(bvh.objs[i].bin > minCostSplitBin);

            {
                for(size_t i = objsBegin; i < objsMid; i++)
                {
                    DEBUG_LAST(drawAabb(bvh.objs[i].bounds, ColorLightBlue));
                }
                for(size_t i = objsMid; i < objsEnd; i++)
                {
                    DEBUG_LAST(drawAabb(bvh.objs[i].bounds, ColorLightGreen));
                }
                DEBUG_NEXT();
            }
#endif


            {
                Bvh::Node *interior = bvh.m_nodes+params.outNode;
                interior->childStart = (uint32_t)bvh.m_numNodes;
                interior->numObjs = 0;
                interior->bounds = totalBounds;
            }

            ASSERT(bvh.m_numNodes+2 < nodeCapacity);
            ASSERT(paramStackSize < _countof(paramStack));

            paramStack[paramStackSize++] = {
                objsBegin,
                objsMid-objsBegin,
                bvh.m_numNodes++
            };

            paramStack[paramStackSize++] = {
                objsMid,
                objsEnd-objsMid,
                bvh.m_numNodes++
            };
        }

    }

#if 0
#endif
}


void Bvh::buildFromTriangles(Vec3 verts[], uint32_t indices[], size_t numTris)
{
    if (m_maxObjs < numTris)
    {
        m_maxObjs = numTris;
        delete [] m_nodes;
        delete [] m_objs;
        m_nodes = new Bvh::Node[m_maxObjs*2];
        m_objs = new Bvh::Obj[m_maxObjs];
    }

    m_numNodes = 0;
    m_numObjs = numTris;
    for (size_t i = 0; i <  numTris; i++)
    {
        size_t base = i*3;
        Vec3 v0 = verts[indices[base+0]];
        Vec3 v1 = verts[indices[base+1]];
        Vec3 v2 = verts[indices[base+2]];
        Aabb bounds = Aabb::empty(); 
        bounds = bounds.expand(v0);
        bounds = bounds.expand(v1);
        bounds = bounds.expand(v2);
        m_objs[i].bounds = bounds;
        m_objs[i].id = (uint32_t)i;
    }

    double t0 = getTime();
    buildBvhInternal(*this); 
    ods("bvhBuildFromTriangles: buildBvhInternal Time = %fms\n", (getTime()-t0)*1000.0);
}


void Bvh::buildFromAabbs(Aabb aabbs[], size_t numAabbs)
{
    if (m_maxObjs < numAabbs)
    {
        m_maxObjs = numAabbs;
        delete [] m_nodes;
        delete [] m_objs;
        m_nodes = new Bvh::Node[m_maxObjs*2];
        m_objs = new Bvh::Obj[m_maxObjs];
    }

    m_numNodes = 0;
    m_numObjs = numAabbs;
    for (size_t i = 0; i <  numAabbs; i++)
    {
        m_objs[i].bounds = aabbs[i];
        m_objs[i].id = (uint32_t)i;
    }

    buildBvhInternal(*this); 
}

size_t Bvh::overlap(Aabb bounds, uint32_t outObjs[], size_t outNumMax) const
{
    size_t stack[64];
    stack[0] = 0;
    size_t stackSize = 1;
    size_t outNum = 0;
    while(stackSize)
    {
        size_t nodeIndex = stack[--stackSize];
        Bvh::Node node = m_nodes[nodeIndex];

        if(bounds.overlaps(node.bounds))
        {
            if(node.numObjs != 0)
            {
                size_t objsBegin = node.childStart;
                size_t objsEnd = node.childStart+node.numObjs;
                for(size_t i = objsBegin; i < objsEnd; i++)
                {
                    Bvh::Obj obj = m_objs[i];

                    //drawAabb(obj.bounds, COLOR_RED);
                    if (bounds.overlaps(obj.bounds))
                    {
                        if(outNum == outNumMax)
                        {
                            ods("warning: colliding with too many BVH objs\n"); 
                            return outNum;
                        }

                        size_t idx = outNum++;
                        outObjs[idx] = m_objs[i].id;
                        //res->bounds[idx] = bvh->objs[i].bounds;
                    }

                }
            }

            if(node.numObjs == 0)
            {
                ASSERT(stackSize + 2 < _countof(stack));
                stack[stackSize++] = node.childStart+0;
                stack[stackSize++] = node.childStart+1;
            }
        }
    }

    return outNum;
}

void Bvh::destroy()
{
    delete [] m_nodes;
    delete [] m_objs;
    *this = {};
}

///////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////



static inline int64_t calcIntTriArea(Vec3i a, Vec3i b, Vec3i c)
{
    int64_t bx = (int64_t)b.x - a.x;
    int64_t by = (int64_t)b.y - a.y;
    int64_t bz = (int64_t)b.z - a.z;

    int64_t cx = (int64_t)c.x - a.x;
    int64_t cy = (int64_t)c.y - a.y;
    int64_t cz = (int64_t)c.z - a.z;

    int64_t rx = by*cz - bz*cy;
    int64_t ry = bz*cx - bx*cz;
    int64_t rz = bx*cy - by*cx;

    return rx*rx + ry*ry + rz*rz;
}

static inline int64_t calcIntParalleotopeVolume(Vec3i a, Vec3i b, Vec3i c, Vec3i d)
{
    int64_t ax = (int64_t)a.x - d.x;
    int64_t ay = (int64_t)a.y - d.y;
    int64_t az = (int64_t)a.z - d.z;

    int64_t bx = (int64_t)b.x - d.x;
    int64_t by = (int64_t)b.y - d.y;
    int64_t bz = (int64_t)b.z - d.z;

    int64_t cx = (int64_t)c.x - d.x;
    int64_t cy = (int64_t)c.y - d.y;
    int64_t cz = (int64_t)c.z - d.z;

    int64_t vol =   ax*(by*cz - bz*cy) 
                  + ay*(bz*cx - bx*cz) 
                  + az*(bx*cy - by*cx);

    return vol;
}

static void findMaximalTetrahedron(Vec3i verts[], size_t numVerts, uint16_t trisOut[4][3])
{
    {
        int64_t minProj[3] ={ INT64_MAX, INT64_MAX, INT64_MAX};
        int64_t maxProj[3] ={ INT64_MIN, INT64_MIN, INT64_MIN};
        size_t minIndex[3];
        size_t maxIndex[3];
        for(size_t i = 0; i < numVerts; i++)
        {
            Vec3i v = verts[i];
            for(size_t j = 0; j < 3; j++)
            {
                int64_t proj = v.get(j);
                if(proj < minProj[j])
                {
                    minProj[j] = proj;
                    minIndex[j] = i;
                }
                if(proj > maxProj[j])
                {
                    maxProj[j] = proj;
                    maxIndex[j] = i;
                }
            }
        }
        ASSERT(maxProj[0] >= minProj[0] && maxProj[1] >= minProj[1] && maxProj[2] >= minProj[2]);

        size_t maxAxis = 0;
        int64_t maxDelta = maxProj[0] - minProj[0];
        for(size_t i = 1; i < 3; i++)
        {
            int64_t delta = maxProj[i]-minProj[i];
            if(delta > maxDelta)
            {
                maxAxis = i;
                maxDelta = delta;
            }
        }


        size_t v0 = minIndex[maxAxis];
        size_t v1 = maxIndex[maxAxis];
        std::swap(verts[v0], verts[0]);
        if(v1 == 0)
        {
            std::swap(verts[v0], verts[1]);
        }
        else
        {
            std::swap(verts[v1], verts[1]);
        }
    }

    {
        size_t v2 = -1;
        int64_t maxArea = 0;
        for(size_t i = 2; i < numVerts; i++)
        {
            int64_t areaSq = calcIntTriArea(verts[0], verts[1], verts[i]);
            if(areaSq > maxArea)
            {
                v2 = i;
                maxArea = areaSq;
            }
        }
        ASSERT(maxArea);
        std::swap(verts[v2], verts[2]);
    }

    {
        size_t v3 = -1;
        int64_t maxVol = 0;
        for(size_t i = 3; i < numVerts; i++)
        {
            int64_t vol = calcIntParalleotopeVolume(verts[0], verts[1], verts[2], verts[i]);
            if(llabs(vol) > llabs(maxVol))
            {
                maxVol = vol;
                v3 = i;
            }
        }
        ASSERT(maxVol);
        std::swap(verts[v3], verts[3]);

        if(maxVol < 0)
        {
            static const uint16_t tris[4][3] = {
                1, 2, 3,
                2, 0, 3,
                0, 1, 3,
                0, 2, 1,
            };
            memcpy(trisOut, tris, sizeof(tris)); 
        }
        else
        {
            static const uint16_t tris[4][3] = {
                2, 1, 3,
                0, 2, 3,
                1, 0, 3,
                2, 0, 1,
            };
            memcpy(trisOut, tris, sizeof(tris)); 
        }
    }
}


const float MIN_FACE_ANGLE = 15.0f;
const float MERGE_VERT_DIST = 0.005f;

struct IntHull
{
    struct Edge
    {
        uint16_t v0;
        uint16_t v1;
        uint16_t twin;
    };

    struct Tri
    {
        Edge edges[3];
    };

    float m_cellSize = 0.0001f;
    int m_cellMax = (int)(1.1f/m_cellSize);
    float m_extentMax = 5.0f;
    float m_minFaceArea = 0.01f;

    std::vector<Tri> m_tris;
    std::vector<Vec3i> m_verts;

    Edge &getEdge(size_t index)         { return ((Edge*)&m_tris[0])[index]; }
    Edge &getEdge(size_t index) const   { return ((Edge*)&m_tris[0])[index]; }

    Vec3 decode(Vec3i p) const
    {
        Vec3 res = {
            (p.x + 0.5f)*m_cellSize,
            (p.y + 0.5f)*m_cellSize,
            (p.z + 0.5f)*m_cellSize,
        };
        return res;
    }

    Vec3 decode(size_t index) const
    {
        return decode(m_verts[index]);
    }


    Vec3i encode(Vec3 p)
    {
        ASSERT(p.x >= -m_extentMax && p.x <= m_extentMax);
        ASSERT(p.y >= -m_extentMax && p.y <= m_extentMax);
        ASSERT(p.z >= -m_extentMax && p.z <= m_extentMax);
        Vec3i res ={
            (int)(p.x/m_cellSize),
            (int)(p.y/m_cellSize),
            (int)(p.z/m_cellSize),
        };
        return res;
    }

    void drawTri(Tri tri, Color color) const
    {
        _drawTri(
            decode(tri.edges[0].v0),
            decode(tri.edges[1].v0),
            decode(tri.edges[2].v0),
            color);
    }

    void drawHull() const
    {
        for(const Tri tri : m_tris)
        {
            drawTri(tri, COLOR_YELLOW);

            for(size_t j = 0; j < 3; j++)
            {
                Color color = COLOR_BLUE;
                if(tri.edges[j].twin == 0xffff)
                {
                    drawArrowTo(decode(tri.edges[j].v0),
                                decode(tri.edges[j].v1), COLOR_PURPLE, 1.0f);
                }
                else
                {
                    _drawLine(decode(tri.edges[j].v0),
                              decode(tri.edges[j].v1), COLOR_BLUE);
                }

                drawPointEx(decode(tri.edges[j].v0), COLOR_GREEN, 0.1f);
            }
        }
    }

    void drawGroupBorder(uint16_t groupMap[])
    {
        for(size_t i = 0; i < m_tris.size(); i++)
        {
            Tri tri = m_tris[i];
            size_t myGroup = groupMap[i];

            for(size_t j = 0; j < 3; j++)
            {
                Edge e = tri.edges[j];
                size_t twinGroup = groupMap[e.twin/3];
                if(twinGroup != myGroup)
                {
                    drawLine(decode(e.v0), decode(e.v1), COLOR_BLUE);
                }
            }
        }

        for(size_t i = 0; i < m_tris.size(); i++)
        {
            drawTri(m_tris[i], COLOR_YELLOW);
        }
    }


    void checkHull() const
    {
        for(const Tri tri : m_tris)
        {
            Vec3i a = m_verts[tri.edges[0].v0];
            Vec3i b = m_verts[tri.edges[1].v0];
            Vec3i c = m_verts[tri.edges[2].v0];

            int64_t areaSq = calcIntTriArea(a, b, c);
            if(areaSq <= 0)
            {
                //drawHull();
            }
            ASSERT(areaSq > 0);

            for(size_t j = 0; j < m_verts.size(); j++)
            {
                Vec3i v = m_verts[j];
                int64_t vol = calcIntParalleotopeVolume(a, b, c, v);
                ASSERT(vol >= 0);
            }
        }

        for(size_t i = 0; i < m_tris.size()*3; i++)
        {
            Edge edge = getEdge(i);
            Edge twin = getEdge(edge.twin);
            ASSERT(twin.twin == i);
            ASSERT(edge.v0 == twin.v1 && edge.v1 == twin.v0);
        }

        // v + f - e = 2
        ASSERT(m_verts.size() + m_tris.size() - m_tris.size()*3/2 == 2);
    }

    void compact(std::vector<uint16_t>& vertexRemap, 
                std::vector<uint16_t>& triRemap, 
                std::vector<uint16_t>& horizon)
    {
        {
            size_t numVerts = 0;
            for(size_t i = 0; i < m_verts.size(); i++)
            {
                if(vertexRemap[i])
                {
                    vertexRemap[i] = (uint16_t)numVerts;
                    m_verts[numVerts++] = m_verts[i];
                }
                else
                {
                    vertexRemap[i] = 0xffff;
                }
            }
            m_verts.resize(numVerts);
        }

        {
            size_t numTris = 0;
            for(size_t i = 0; i < m_tris.size(); i++)
            {
                if(triRemap[i])
                {
                    triRemap[i] = (uint16_t)numTris;
                    m_tris[numTris++] = m_tris[i];
                }
                else
                {
                    triRemap[i] = 0xffff;
                }
            }
            m_tris.resize(numTris);
        }

        for(size_t i = 0; i < m_tris.size(); i++)
        {
            Tri &tri = m_tris[i];
            for(size_t j = 0; j < 3; j++)
            {
                Edge &e = tri.edges[j];
                ASSERT(vertexRemap[e.v0] != 0xffff);
                e.v0 = vertexRemap[e.v0];
                e.v1 = vertexRemap[e.v1];

                ASSERT(e.twin != 0xffff);
                size_t triIndex = e.twin/3;
                size_t edgeIndex = e.twin%3;
                if(triRemap[triIndex] == 0xffff)
                {
                    e.twin = 0xffff;
                    horizon.push_back((uint16_t)(3*i + j));
                }
                else
                {
                    e.twin = (uint16_t)(3*triRemap[triIndex] + edgeIndex);
                }
            }
        }
    }

    void connectEdgesBruteForce(size_t begin, size_t end)
    {
        // Brute force connect edges
        // TODO: order edges and connect in linear time
        for(size_t i = begin; i < end; i++)
        {
            Edge &a = getEdge(i);
            for(size_t j = begin; j < end; j++)
            {
                Edge &b = getEdge(j);
                if(a.v0 == b.v1 && a.v1 == b.v0)
                {
                    a.twin = (uint16_t)j;
                    b.twin = (uint16_t)i;
                }
            }
        }
    }

    void build(std::vector<Vec3i> inVerts)
    {
        {
            uint16_t tris[4][3];
            findMaximalTetrahedron(&inVerts[0], (int)inVerts.size(), tris);

            m_verts.resize(4);
            for(size_t i = 0; i < 4; i++)
            {
                m_verts[i] = inVerts[i];
            }

            m_tris.resize(4);
            for(size_t i = 0; i < 4; i++)
            {
                for(size_t j = 0; j < 3; j++)
                {
                    getEdge(i*3 + j) = {tris[i][j], tris[i][(j+1)%3]};
                }
            }
            connectEdgesBruteForce(0, 12);
        }

        std::vector<uint16_t> vertexRemap;
        std::vector<uint16_t> triRemap;
        std::vector<uint16_t> horizon;

        checkHull();
        DEBUG_LAST(drawHull());
        DEBUG_NEXT();

        for(size_t i = 4; i < inVerts.size(); i++)
        {
            Vec3i vert = inVerts[i];

            triRemap.assign(m_tris.size(), 0);
            vertexRemap.assign(m_verts.size(), 0);

            bool insideHull = true;
            for(size_t j = 0; j < m_tris.size(); j++)
            {
                Tri tri = m_tris[j];


                Vec3i a = m_verts[tri.edges[0].v0];
                Vec3i b = m_verts[tri.edges[1].v0];
                Vec3i c = m_verts[tri.edges[2].v0];

                int64_t vol = calcIntParalleotopeVolume(a, b, c, vert);
                if(vol < 0)
                {
                    DEBUG_LAST(drawTri(hull, tri, COLOR_PURPLE));
                    insideHull = false;
                }
                else
                {
                    triRemap[j] = 1;
                    for(size_t k = 0; k < 3; k++)
                    {
                        Edge e = tri.edges[k];
                        vertexRemap[e.v0] = 1;
                    }
                }
            }


            if(insideHull)
            {
                DEBUG_LAST(drawPoint(decode(vert), COLOR_RED));
                DEBUG_LAST(drawHull(hull));
                DEBUG_NEXT();
                continue;
            }

            DEBUG_LAST(drawPoint(decode(vert), ColorLightBlue));
            DEBUG_LAST(drawHull(hull));
            DEBUG_NEXT();

            horizon.clear();
            compact(vertexRemap, triRemap, horizon);

            DEBUG_LAST(drawPoint(decode(vert), ColorLightBlue));
            DEBUG_LAST(drawHull(hull));
            DEBUG_NEXT();

            size_t vertIndex = m_verts.size();
            m_verts.push_back(vert);
            for(size_t i = 0; i < horizon.size(); i++)
            {
                size_t eIdx = horizon[i];
                Edge &e = getEdge(eIdx);
                e.twin = (uint16_t)(m_tris.size()*3);

                Tri tri;
                tri.edges[0].v0 = e.v1;
                tri.edges[0].v1 = e.v0;
                tri.edges[0].twin = horizon[i];

                tri.edges[1].v0 = e.v0;
                tri.edges[1].v1 = (uint16_t)vertIndex;

                tri.edges[2].v0 = (uint16_t)vertIndex;
                tri.edges[2].v1 = e.v1;
                m_tris.push_back(tri);
            }

            connectEdgesBruteForce((int)m_tris.size()*3 - (int)horizon.size()*3, (int)m_tris.size()*3);

            //checkHull(hull);


            DEBUG_LAST(drawPoint(decode(vert), ColorLightBlue));
            DEBUG_LAST(drawHull(hull));
            DEBUG_NEXT();
        }
        checkHull();
    }


    void findHalfSpaces(std::vector<Plane> &planes) const
    {
        struct WeightedPlane
        {
            Vec3 normal;
            float dist;
            float area;

            static int compare(const void* a, const void* b)
            {
                if(((WeightedPlane*)a)->area < ((WeightedPlane*)b)->area) return 1;
                return -1;
            }
        };

        std::vector<WeightedPlane> weighted;
        weighted.reserve(m_tris.size());

        float cosMax = cosf(Math::DEG2RAD*MIN_FACE_ANGLE);
        for(size_t i = 0; i < m_tris.size(); i++)
        {
            Tri tri = m_tris[i];
            Vec3 a = decode(tri.edges[0].v0);
            Vec3 b = decode(tri.edges[1].v0);
            Vec3 c = decode(tri.edges[2].v0);
            Vec3 normal = triComputeNormal(a, b, c);
            float len = normal.length();

            {
                float A = (b-a).length();
                float B = (c-b).length();
                float C = (a-c).length();
                float S = 0.5f*(A+B+C);
                float AR = (A*B*C)/(8*(S-A)*(S-B)*(S-C));
                if (AR > 20.0f)
                    continue;
            }

            WeightedPlane w;
            w.area = len/2;
            w.normal = normal/len;
            w.dist = w.normal.dot(a);
            weighted.push_back(w);
        }
        qsort(&weighted[0], weighted.size(), sizeof(weighted[0]), WeightedPlane::compare);

        std::vector<WeightedPlane> merged;
        for(const WeightedPlane &w : weighted)
        {
            size_t index = -1;
            for(size_t j = 0; j < merged.size(); j++)
            {
                float cosAngle = w.normal.dot(merged[j].normal);
                if(cosAngle >= cosMax)
                {
                    index = j;
                    break;
                }
            }

            if(index == -1)
            {
                merged.push_back(w);
            }
            else
            {
                merged[index].area += w.area;
            }
        }

        for(size_t i = 0; i < merged.size(); i++)
        {
            WeightedPlane w = merged[i];
            if(w.area > m_minFaceArea)
            {
                Plane p;
                p.dist = w.dist;
                p.normal = w.normal;
                planes.push_back(p);
            }
        }
    }
};

static void drawPoly(Vec3 verts[], size_t numVerts)
{
    Vec3 v0 = verts[0];
    for(size_t j = 1; j < numVerts-1; j++)
    {
        Vec3 v1 = verts[j];
        Vec3 v2 = verts[j+1];
        _drawTri(v0, v1, v2, COLOR_YELLOW);
        _drawTri(v0, v2, v1, COLOR_ORANGE);
    }
    for (size_t i = 0; i < numVerts; i++)
    {
        Vec3 v1 = verts[i];
        Vec3 v2 = verts[(i+1)%numVerts];
        drawLine(v1,v2,COLOR_BLUE);
    }
}

static void drawConvexHull(const ConvexHull &h)
{
    for(size_t i = 0; i < h.m_numFaces; i++)
    {
        ConvexHull::Face f = h.m_faces[i];
        ConvexHull::Edge e0 = h.m_edges[f.firstEdge];
        for(size_t j = 1; j < f.numEdges-1; j++)
        {
            ConvexHull::Edge e1 = h.m_edges[f.firstEdge+j];
            _drawTri(h.m_verts[e0.v0], h.m_verts[e1.v0], h.m_verts[e1.v1], COLOR_YELLOW);
        }

        drawArrow(f.centroid, f.normal*0.2f, COLOR_RED, 0.2f);
    }
    for (size_t i = 0; i < h.m_numEdges; i++)
    {
        ConvexHull::Edge e = h.m_edges[i];
        drawLine(h.m_verts[e.v0], h.m_verts[e.v1], COLOR_BLUE);
    }
    for(size_t i = 0; i < h.m_numVerts; i++)
    {
        drawPointEx(h.m_verts[i], COLOR_GREEN, 0.1f);
    }
}

static void findExtremePoints(std::vector<Vec3>& out, const std::vector<Plane> &planes) 
{
    float cosMax = cosf(Math::DEG2RAD*(MIN_FACE_ANGLE-0.01f));

    std::vector<Vec3> source;
    std::vector<Vec3> clipped;
    clipped.resize(planes.size()*2);

	for (size_t i = 0; i < planes.size(); i++) 
    {
		Plane p = planes[i];

        Mat3 b = Mat3::basis(p.normal);
		Vec3 center = p.normal*p.dist;

        source.clear();

        float extent = 5;
        source.push_back(center - b.c0*extent - b.c1*extent);
        source.push_back(center + b.c0*extent - b.c1*extent);
        source.push_back(center + b.c0*extent + b.c1*extent);
        source.push_back(center - b.c0*extent + b.c1*extent);


        DEBUG_LAST(drawPoly(source, source.size()));
        DEBUG_NEXT();
        for(size_t j = 0; j < planes.size(); j++) 
        {
            if(j == i) 
            {
                continue;
            }


            Plane clip = planes[j];
            float cosAngle = clip.normal.dot(p.normal);
            ASSERT(cosAngle < cosMax); 
            size_t numClipped = clipPolygonAgainstPlane(clip.normal, clip.dist, &source[0], (int)source.size(), &clipped[0]);
            source.assign(&clipped[0], &clipped[0] + numClipped);
            if (source.size() < 3)
            {
                break;
            }
            DEBUG_LAST(drawPoly(source, source.size()));
            DEBUG_NEXT();
        }

        if (source.size() < 3)
        {
            continue;
        }

        DEBUG_ANY(drawPoly(source, source.size()));

        for(size_t j = 0; j < source.size(); j++)
        {
            ASSERT(source[j].length() < 10);
            size_t index = -1;
            for(size_t k = 0; k < out.size(); k++)
            {
                float d = out[k].distSq(source[j]);
                if(d < MERGE_VERT_DIST*MERGE_VERT_DIST)
                {
                    index = k;
                    break;
                }
            }
            if (index == -1)
            {
                out.push_back(source[j]);
            }
        }
    }
}

static void newellPlane(Vec3 points[], size_t num, Vec3 *outNormal, float *outDist, Vec3 *outCentroid)
{
    Vec3 centroid = {0};
    Vec3 normal = {0};
    Vec3 v0 = points[num-1];
    for (size_t i = 0; i < num; i++)
    {
        Vec3 v1 = points[i];
        normal.x += (v0.y - v1.y) * (v0.z + v1.z); 
        normal.y += (v0.z - v1.z) * (v0.x + v1.x);
        normal.z += (v0.x - v1.x) * (v0.y + v1.y);
        centroid = centroid + v1;
        v0 = v1;
    }

    ASSERT(normal.x || normal.y || normal.z);
    bool res = true;
    normal = normal.normalized();
    

    centroid = centroid/(float)num;

    float dist = normal.dot(centroid);

    *outNormal = normal;
    *outDist = dist;
    *outCentroid = centroid;
}

void ConvexHull::destroy()
{
    free(m_faces);
    free(m_edges);
    free(m_verts);
}

void ConvexHull::build(const Vec3 inVerts[], const size_t numVerts)
{
    ASSERT(numVerts >= 4);

    std::vector<Vec3i> intVerts;
    intVerts.resize(numVerts);
    float outputScale;
    Aabb bounds = Aabb::empty();
    for(size_t i = 0; i < numVerts; i++)
    {
        bounds = bounds.expand(inVerts[i]);
    }

    IntHull intHull;

    outputScale = bounds.getMaxExtent();
    float inputScale = 1.0f/outputScale;
    for(size_t i = 0; i < numVerts; i++)
    {
        Vec3 rel = inVerts[i] - bounds.getMin();
        intVerts[i] = intHull.encode(rel*inputScale);
    }


    DEBUG_EXCLUDE(intHull.build(intVerts));

    //DEBUG_ANY(drawHull(hull));
    //g_debugCurrentStep = 0;

    std::vector<Plane> planes;
    intHull.findHalfSpaces(planes);

    std::vector<Vec3> extremePoints;
    findExtremePoints(extremePoints, planes);
    ASSERT(extremePoints.size() <= numVerts);
    intVerts.resize(extremePoints.size());
    for(size_t i = 0; i < extremePoints.size(); i++)
    {
        intVerts[i] = intHull.encode(extremePoints[i]);
    }

    DEBUG_EXCLUDE(intHull.build(intVerts));

    std::vector<uint16_t> groupMap;
    size_t groupCount = 0;
    {
        groupMap.assign(intHull.m_tris.size(), 0xffff);

        float cosMax = cosf(Math::DEG2RAD*2.0f);

        std::vector<size_t> frontier;
        for (size_t i = 0; i < intHull.m_tris.size(); i++)
        {
            if (groupMap[i] != 0xffff)
            {
                continue;
            }

            size_t myGroup;
            Vec3 myNormal;
            frontier.clear();

            {
                myGroup = groupCount++;
                groupMap[i] = (uint16_t)myGroup;

                IntHull::Tri tri = intHull.m_tris[i];
                Vec3 a = intHull.decode(tri.edges[0].v0);
                Vec3 b = intHull.decode(tri.edges[1].v0);
                Vec3 c = intHull.decode(tri.edges[2].v0);
                myNormal = triComputeNormal(a,b,c);

                for(size_t j = 0; j < 3; j++)
                {
                    size_t twin = tri.edges[j].twin/3;
                    if(groupMap[twin] == 0xffff)
                    {
                        frontier.push_back(twin);
                    }
                }
            }

            while(frontier.size())
            {
                size_t curIndex = frontier.back();
                frontier.pop_back();

                IntHull::Tri cur = intHull.m_tris[curIndex];
                Vec3 a = intHull.decode(cur.edges[0].v0);
                Vec3 b = intHull.decode(cur.edges[1].v0);
                Vec3 c = intHull.decode(cur.edges[2].v0);
                Vec3 normal = triComputeNormal(a,b,c);

                float cosAngle = normal.dot(myNormal);
                if (cosAngle > cosMax)
                {
                    groupMap[curIndex] = (uint16_t)myGroup;

                    for(size_t j = 0; j < 3; j++)
                    {
                        size_t twin = cur.edges[j].twin/3;
                        size_t twinGroup = groupMap[twin];
                        if(twinGroup == 0xffff)
                        {
                            frontier.push_back(twin);
                        }
                    }
                }
            }
        }
    }

    //DEBUG_ANY(drawGroupBorder(hull, groupMap));


    std::vector<Face> outFaces;
    std::vector<Edge> outEdges;

    std::vector<Edge> faceEdges;
    for(size_t group = 0; group < groupCount; group++)
    {
        faceEdges.clear();

        for(size_t i = 0; i < intHull.m_tris.size(); i++)
        {
            if (groupMap[i] != group)
            {
                continue;
            }

            IntHull::Tri tri = intHull.m_tris[i];
            for(size_t j = 0; j < 3; j++)
            {
                IntHull::Edge e = tri.edges[j];
                size_t twinGroup = groupMap[e.twin/3];
                if(twinGroup != group)
                {
                    Edge tmp;
                    tmp.v0 = e.v0;
                    tmp.v1 = e.v1;
                    tmp.f0 = (uint16_t)group;
                    tmp.f1 = (uint16_t)twinGroup;
                    faceEdges.push_back(tmp);
                }
            }
        }

        ASSERT(faceEdges.size() >= 3);
        for (size_t i = 0; i < faceEdges.size()-1; i++)
        {
            Edge a = faceEdges[i];
            for (size_t j = i+1; j < faceEdges.size(); j++)
            {
                Edge b = faceEdges[j];
                if (a.v1 == b.v0)
                {
                    std::swap(faceEdges[i+1], faceEdges[j]);
                    break;
                }
            }
        }

        // TODO: check face is convex
        for (size_t i = 0; i < faceEdges.size(); i++)
        {
            ASSERT(faceEdges[i].v1 == faceEdges[(i+1)%faceEdges.size()].v0);
        }

        Face face;
        face.firstEdge = (int)outEdges.size();
        face.numEdges = (int)faceEdges.size();
        outEdges.insert(outEdges.end(), faceEdges.begin(), faceEdges.end());
        outFaces.push_back(face);
    }

    std::vector<uint16_t> vertRemap;
    vertRemap.assign(intHull.m_verts.size(), 0);
    for (size_t i = 0; i < outEdges.size(); i++)
    {
        vertRemap[outEdges[i].v0] = 1;
    }


    std::vector<Vec3> outVerts;
    for (size_t i = 0; i < intHull.m_verts.size(); i++)
    {
        if (vertRemap[i] == 1)
        {
            size_t index = outVerts.size();
            outVerts.push_back(intHull.decode(intHull.m_verts[i]));
            vertRemap[i] = (uint16_t)index;
        }
    }

    for (size_t i = 0; i < outEdges.size(); i++)
    {
        Edge &e = outEdges[i];
        e.v0 = vertRemap[e.v0];
        e.v1 = vertRemap[e.v1];
        ASSERT(e.v0 != 0xffff);
        ASSERT(e.v1 != 0xffff);
    }

    for (size_t i = 0; i < outEdges.size(); i++)
    {
        ASSERT(outEdges[i].f1 != -1);
    }

    ASSERT(outVerts.size() + outFaces.size() - outEdges.size()/2 == 2);

    for (size_t i = 0; i < outVerts.size(); i++)
    {
        outVerts[i] = bounds.getMin() + outVerts[i]*outputScale;
    }

    for (size_t i = 0; i < outFaces.size(); i++)
    {
        Face &face = outFaces[i];
        Vec3 faceVerts[128];
        ASSERT(face.numEdges < _countof(faceVerts));
        for (size_t j = 0; j < face.numEdges; j++)
        {
            faceVerts[j] = outVerts[outEdges[face.firstEdge + j].v0];
        }

        newellPlane(faceVerts, face.numEdges, &face.normal, &face.dist, &face.centroid);
    }

    {
        size_t size = sizeof(Face)*outFaces.size();
        m_numFaces = (int)outFaces.size();
        m_faces = (Face*)malloc(size);
        memcpy(m_faces, &outFaces[0], size);
    }
    {
        size_t size = sizeof(Edge)*outEdges.size();
        m_numEdges = (int)outEdges.size();
        m_edges = (Edge*)malloc(size);
        memcpy(m_edges, &outEdges[0], size);
    }
    {
        size_t size = sizeof(Vec3)*outVerts.size();
        m_numVerts = (int)outVerts.size();
        m_verts = (Vec3*)malloc(size);
        memcpy(m_verts, &outVerts[0], size);
    }


    DEBUG_ANY(drawConvexHull(*this));
}

static const Vec3 *genBox(size_t numPoints, size_t seed)
{
    static Vec3 points[4096];
    srand((unsigned int)seed);
    ASSERT(numPoints < _countof(points));
    for(size_t i = 0; i < numPoints; i++)
    {
        points[i] = Vec3::rand01();
    }
    return points;
}

void testRandomPoints(size_t numPoints, size_t numTests, float minFaceAngleDeg)
{
    size_t testsPerPercent = numTests/100;

    for (size_t seed = 0; seed < numTests; seed++)
    {
        srand((unsigned int)seed);

        const Vec3 *points = genBox(numPoints, seed);

        if ((seed % testsPerPercent) == 0)
        {
            ods("progress:%d%%\n", seed/testsPerPercent);
        }

        ConvexHull hull;
        hull.build(points, numPoints);
        hull.destroy();
    }
}

#ifdef DEBUG_INTERACTIVE
static void testInteractive()
{
    static float minFaceAngleDeg = 10.0f;


    static size_t numPoints = 128;
    if(ImGui::IsKeyPressed(ImGuiKey_Tab))     g_seed++;
    if(ImGui::IsKeyPressed(ImGuiKey_Space))   g_seed--;
    if(ImGui::IsKeyPressed(ImGuiKey_Left))    g_debugLastStep--;
    if(ImGui::IsKeyPressed(ImGuiKey_Right))   g_debugLastStep++;
    if(ImGui::IsKeyPressed(ImGuiKey_Up))      minFaceAngleDeg += 1.0f;
    if(ImGui::IsKeyPressed(ImGuiKey_Down))    minFaceAngleDeg -= 1.0f;
    //if(ImGui::IsKeyPressed(KEY_J))        numPoints += 32;
    //if(ImGui::IsKeyPressed(KEY_K))        numPoints -= 32;

    minFaceAngleDeg = clamp(minFaceAngleDeg, 1.0f, 15.0f);
    numPoints = clamp(numPoints, 3, 100000);
    g_debugLastStep = max(g_debugLastStep, 0);

    const Vec3 *points = genBox(numPoints, g_seed);

    g_debugCurrentStep = 0;
    TmpHull hull;
    DEBUG_TRY(hull.build(points, numPoints));
    hull.destroy();
}

static void testMesh()
{
    static TriMesh testMesh;
    if (!testMesh.m_verts)
    {
        //ASSERT(testMesh.loadFromObj("test_meshes/stanford-bunny.obj"));
        ASSERT(testMesh.loadFromObj("test_meshes/teapot.obj"));
    }
    static float minFaceAngleDeg = 10.0f;

    if(ImGui::IsKeyPressed(VK_UP))      minFaceAngleDeg += 1.0f;
    if(ImGui::IsKeyPressed(VK_DOWN))    minFaceAngleDeg -= 1.0f;
    minFaceAngleDeg = clamp(minFaceAngleDeg, 1.0f, 15.0f);

    g_debugLastStep = 1000000000;
    TmpHull hull;
    if (ImGui::IsKeyPressed(ImGuiKey_Backspace))
    {
        hull.build(testMesh.m_verts, testMesh.m_numVerts);
        hull.destroy();
    }
    //drawTriMesh(testMesh.m_verts, testMesh.m_indices, testMesh.m_numIndices, ColorLightBlue);
}
#endif

void ConvexHull::test()
{
#ifdef DEBUG_INTERACTIVE
    //testInteractive();
    testMesh();
#else
    testRandomPoints(128, 100*1000*1000, 10.0f);
#endif
}

///////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////



//#define DRAW_NORMALS(...) __VA_ARGS__
#define DRAW_NORMALS(...) 

inline bool intersectRayBox(Vec3 ray_df, Vec3 ray_f0, Vec3 box_min, Vec3 box_max, float *t_out)
{
    float dirfrac_x = 1.0f / ray_df.x;
    float dirfrac_y = 1.0f / ray_df.y;
    float dirfrac_z = 1.0f / ray_df.z;

    float t1 = (box_min.x - ray_f0.x)*dirfrac_x;
    float t2 = (box_max.x - ray_f0.x)*dirfrac_x;
    float t3 = (box_min.y - ray_f0.y)*dirfrac_y;
    float t4 = (box_max.y - ray_f0.y)*dirfrac_y;
    float t5 = (box_min.z - ray_f0.z)*dirfrac_z;
    float t6 = (box_max.z - ray_f0.z)*dirfrac_z;

    float tmin = Math::max(Math::max(Math::min(t1,t2),Math::min(t3,t4)),Math::min(t5,t6));
    float tmax = Math::min(Math::min(Math::max(t1,t2),Math::max(t3,t4)),Math::max(t5,t6));

    // if tmax < 0, ray (line) is intersecting AABB, but the whole AABB is behind us
    if(tmax < 0)
    {
        *t_out = tmax;
        return false;
    }

    // if tmin > tmax, ray doesn't intersect AABB
    if(tmin > tmax)
    {
        *t_out = tmax;
        return false;
    }

    *t_out = tmin;
    return true;
}

bool intersectSphere(Vec3 dir, Vec3 orig, Vec3 center, float radius, float *t_out) 
{
    float radius2 = radius*radius;
    Vec3 L = center - orig;
    float tca = L.dot(dir);
    // if (tca < 0) return false;
    float d2 = L.dot(L) - tca * tca;
    if(d2 > radius*radius) return false;
    float thc = sqrtf(radius2 - d2);
    float t0 = tca - thc;
    float t1 = tca + thc;
    if(t0 > t1) 
    {
        std::swap(t0,t1);
    }
    if(t0 < 0) 
    {
        t0 = t1; 
        if(t0 < 0) return false; // both t0 and t1 are negative 
    }
    *t_out = t0;
    return true;
}


inline bool intersectRayPlane(Vec3 planeNormal, float planeDist, Vec3 rayOrigin, Vec3 rayDir, float *tOut)
{
    float slope = planeNormal.dot(rayDir);
    float dist = planeNormal.dot(rayOrigin) - planeDist;
    if(slope < 0)
    {
        *tOut = -dist/slope;
        return true;
    }
    return false;
}


inline Vec3 closestPointOnSegment(Vec3 p, Vec3 a, Vec3 b)
{
    Vec3 ba = b - a; 
    float t =(p - a).dot(ba) / ba.dot(ba);
    return Vec3::lerp(Math::clamp(t,0.0f,1.0f), a, b);
}

inline void closestPointSegments(Vec3 segA, Vec3 segB, Vec3 segD, Vec3 segC, Vec3 *outCD, Vec3 *outAB)
{
    Vec3 segDC = segD - segC;
    float invLineDirSqrMag = 1.0f/segDC.dot(segDC);
    Vec3 inPlaneA = segA - segDC*(segA-segC).dot(segDC)*invLineDirSqrMag;
    Vec3 inPlaneB = segB - segDC*(segB-segC).dot(segDC)*invLineDirSqrMag;
    Vec3 inPlaneBA = inPlaneB - inPlaneA;
    float t = inPlaneA.equals(inPlaneB) ? 0 : (segC - inPlaneA).dot(inPlaneBA)/inPlaneBA.dot(inPlaneBA);
    Vec3 segABtoLineCD = Vec3::lerp(Math::clamp(t,0.0f,1.0f), segA, segB);
    Vec3 segCDtoSegAB = closestPointOnSegment(segABtoLineCD, segC, segD);
    Vec3 segABtoSegCD = closestPointOnSegment(segCDtoSegAB, segA, segB);
    *outCD = segCDtoSegAB;
    *outAB = segABtoSegCD;
}

inline void calcBestFitSegment(Vec3 points[], size_t numPoints, Vec3 *outA, Vec3 *outB)
{
    ASSERT(numPoints > 1 && numPoints < 256);
    Vec3 centroid = {0};
    for (size_t i = 0; i < numPoints; i++)
    {
        centroid += points[i];
    }
    centroid *= 1.0f/numPoints;

    Aabb bounds = Aabb::empty();
    for (size_t i = 0; i < numPoints; i++)
    {
        bounds = bounds.expand(points[i]);
    }
    size_t axis;
    bounds.getMaxExtent(axis);

    Vec3 direction = {0};
    (&direction.x)[axis] = 1;

    for (size_t i = 0; i < 50; i++)
    {
        Vec3 nextDirection ={0};
        for(size_t i = 0; i < numPoints; i++)
        {
            Vec3 centeredPoint = points[i] - centroid;
            nextDirection = nextDirection + centeredPoint*centeredPoint.dot(direction);
        }
        direction = nextDirection.normalized();
    }

    /*
    float error = 0;
    for (size_t i = 0; i < numPoints; i++)
    {
        Vec3 rel = points[i] - centroid;
        float dist = vecLength(vecClip(rel, direction));
        error = maxf(dist, error);
    }*/

    float minProj = INFINITY;
    float maxProj = -INFINITY;
    for (size_t i = 0; i < numPoints; i++)
    {
        float proj = points[i].dot(direction);
        minProj = Math::min(minProj, proj);
        maxProj = Math::max(maxProj, proj);
    }

    *outA = direction*minProj;
    *outB = direction*maxProj;
}


bool raycastSatShape(Vec3 rayOrigin, Vec3 rayDir, const SatShape *shape, Transform shapeTransform, float *tOut, Vec3 *hitOut)
{
    Transform iT = shapeTransform.inverse();
    Vec3 localRayOrigin = iT.mul(rayOrigin);
    Vec3 localRayDir = iT.rotate(rayDir);

    {
        float tHitAabb;
        float r = shape->boundingRadius;
        if(!intersectRayBox(localRayDir, localRayOrigin, {-r,-r,-r}, {r,r,r}, &tHitAabb))
        {
            return false;
        }
    }

    float tMin = -INFINITY;
    float tMax = INFINITY;
    for(size_t j = 0; j < shape->numFaces; j++)
    {
        Plane plane = shape->facePlanes[j];
        float slope = plane.normal.dot(localRayDir);
        float dist = plane.normal.dot(localRayOrigin) - plane.dist;
        if(slope < 0)
        {
            float t = -dist/slope;
            tMin = Math::max(t, tMin);
        }
        else if(slope > 0)
        {
            float t = -dist/slope;
            tMax = Math::min(t, tMax);
        }
    }

    if (tMin < tMax && tMin > 0)
    {
        *tOut = tMin;
        *hitOut = rayOrigin + rayDir*tMin;
        return true;
    }
    else
    {
        return false;
    }
}

const size_t SAT_MAX = 256;

struct BuddyAllocation
{
    size_t offset;
    size_t size;
};

#define BUDDY(root,field,num) BuddyAllocation{offsetof(root,field), sizeof(root::field[0])*(num)}

template <typename RootType>
RootType *allocateBuddies(Allocator *allocator, const std::initializer_list<BuddyAllocation> &allocations)
{
    size_t totalSize = sizeof(RootType);
    for(auto &it : allocations)
    {
        totalSize += it.size;
    }

    RootType *rootObj = (RootType*)allocator->alloc((int)totalSize);
    memset(rootObj, 0, sizeof(rootObj));
    char *ptr = (char*)rootObj+sizeof(RootType);
    for(auto &it : allocations)
    {
        *(void**)((char*)rootObj + it.offset) = ptr;
        ptr += it.size;
    }

    return rootObj;
}


SatShape *SatShape::createFromVerts(Allocator *allocator, Vec3 verts[], size_t numVerts)
{
    ConvexHull hull;
    hull.build(verts, numVerts);

    ASSERT(hull.m_numVerts + hull.m_numFaces - hull.m_numEdges/2 == 2);
    ASSERT(hull.m_numEdges < 255);
    ASSERT(hull.m_numVerts < 255);
    ASSERT(hull.m_numFaces < 255);

    SatShape *shape = allocateBuddies<SatShape>(allocator, {
        BUDDY(SatShape, faces,      hull.m_numFaces),
        BUDDY(SatShape, verts,      hull.m_numVerts),
        BUDDY(SatShape, facePlanes, hull.m_numFaces),
        BUDDY(SatShape, vertPos,    hull.m_numVerts),
        BUDDY(SatShape, faceEdges,  hull.m_numEdges),
        BUDDY(SatShape, vertEdges,  hull.m_numEdges),
        BUDDY(SatShape, edges,      hull.m_numEdges/2),
    });

    shape->numFaces   = (uint8_t)hull.m_numFaces;
    shape->numEdges   = (uint8_t)hull.m_numEdges/2;
    shape->numVerts   = (uint8_t)hull.m_numVerts;

    /*
    {
        size_t N = ROUNDUP(hull.numVerts, 4);
        shape->vertPosTransposed = arenaAlloc(arena, sizeof(float)*3*N);
        for (size_t i = 0; i < hull.numVerts; i++)
        {
            shape->vertPosTransposed[0*N + i] = hull.verts[i].x;
            shape->vertPosTransposed[1*N + i] = hull.verts[i].y;
            shape->vertPosTransposed[2*N + i] = hull.verts[i].z;
        }
        for (size_t i = hull.numVerts; i < N; i++)
        {
            shape->vertPosTransposed[0*N + i] = hull.verts[hull.numVerts-1].x;
            shape->vertPosTransposed[1*N + i] = hull.verts[hull.numVerts-1].y;
            shape->vertPosTransposed[2*N + i] = hull.verts[hull.numVerts-1].z;
        }
    }*/

    size_t numEdges = 0;
    for (size_t i = 0; i < hull.m_numEdges; i++)
    {
        size_t shapeEdgeIndex = -1;
        ConvexHull::Edge tmpEdge = hull.m_edges[i];    
        for (size_t j = 0; j < numEdges; j++)
        {
            SatShape::Edge satEdge = shape->edges[j];
            if ((satEdge.v0 == tmpEdge.v0 && satEdge.v1 == tmpEdge.v1) || (satEdge.v1 == tmpEdge.v0 && satEdge.v0 == tmpEdge.v1))
            {
                shapeEdgeIndex = j;
                break;
            }
        }

        if (shapeEdgeIndex == -1)
        {
            SatShape::Edge satEdge;
            satEdge.v0 = (uint8_t)tmpEdge.v0;
            satEdge.v1 = (uint8_t)tmpEdge.v1;
            satEdge.f0 = (uint8_t)tmpEdge.f0;
            satEdge.f1 = (uint8_t)tmpEdge.f1;
            shapeEdgeIndex = numEdges;
            shape->edges[numEdges++] = satEdge;
        }
    }
    ASSERT(numEdges == hull.m_numEdges/2);

    Vec3 centroid = {0};
    for (size_t i = 0; i < hull.m_numVerts; i++) centroid += hull.m_verts[i];
    centroid = centroid*(1.0f/hull.m_numVerts);
    for (size_t i = 0; i < hull.m_numVerts; i++) shape->vertPos[i] = hull.m_verts[i] - centroid;


    for (size_t i = 0; i < hull.m_numEdges; i++)
    {
        shape->faceEdges[i] = { 
            (uint8_t)hull.m_edges[i].v0,  
            (uint8_t)hull.m_edges[i].v1,  
            (uint8_t)hull.m_edges[i].f0,  
            (uint8_t)hull.m_edges[i].f1, 
        };
    }


    for (size_t i = 0; i < hull.m_numFaces; i++)
    {
        SatShape::EdgeList *satFace = &shape->faces[i];
        ConvexHull::Face tmpFace = hull.m_faces[i];
        satFace->first = (uint8_t)tmpFace.firstEdge;
        satFace->num = (uint8_t)tmpFace.numEdges;
        shape->facePlanes[i].normal = tmpFace.normal;
        shape->facePlanes[i].dist = tmpFace.dist - tmpFace.normal.dot(centroid);;
    }

    shape->boundingRadius = 0;
    for (size_t i = 0; i < shape->numVerts; i++)
    {
        float r = shape->vertPos[i].length();
        shape->boundingRadius = Math::max(r, shape->boundingRadius);
    }



    // Sort edges.
    {
        uint8_t openSet[SAT_MAX] = {};
        uint8_t closedSet[SAT_MAX] = {};
        uint8_t edgeStack[SAT_MAX];
        SatShape::Edge sorted[SAT_MAX];
        size_t stackSize = 0;

        edgeStack[stackSize++] = 0;
        size_t numSorted = 0;

        while (stackSize)
        {
            size_t curIdx = edgeStack[--stackSize];
            if (closedSet[curIdx])
            {
                continue;
            }
            closedSet[curIdx] = 1;
            SatShape::Edge cur = shape->edges[curIdx];
            sorted[numSorted++] = cur;

            // find neighbors
            for (size_t i = 0; i < shape->numEdges; i++)
            {
                SatShape::Edge e = shape->edges[i];
                if (!openSet[i] && !closedSet[i] && (cur.f1 == e.f0 || cur.f1 == e.f1 || cur.f0 == e.f0 || cur.f0 == e.f1))
                {
                    ASSERT(stackSize < SAT_MAX);
                    openSet[i] = 1;
                    edgeStack[stackSize++] = (uint8_t)i;
                }
            }
        }

        ASSERT(numSorted == shape->numEdges);
        for (size_t i = 0; i < shape->numEdges; i++)
        {
            SatShape::Edge a = shape->edges[i];
            bool found = false;
            for (size_t j = 0; j < numSorted; j++)
            {
                SatShape::Edge b = sorted[j];
                if (a.f0 == b.f0 && a.f1 == b.f1 && a.v0 == b.v0 && a.v1 == b.v1)
                {
                    ASSERT(!found);
                    found = true;
                }
            }
            ASSERT(found);
        }

        memcpy(shape->edges, sorted, shape->numEdges*sizeof(shape->edges[0]));
        uint8_t color[SAT_MAX] = {0};
        color[shape->edges[0].f0] = 1;
        for (size_t i = 0; i < shape->numEdges; i++)
        {
            SatShape::Edge *e = &shape->edges[i];
            if (!color[e->f0])
            {
                std::swap(e->f0, e->f1);
                std::swap(e->v0, e->v1);
            }
            ASSERT(color[e->f0]);
            color[e->f1] = 1;
        }
    }

    {
        size_t num = 0;
        for(size_t i = 0; i < shape->numVerts; i++)
        {
            SatShape::EdgeList *v = shape->verts+i;

            v->num = 0;
            v->first = (uint8_t)num;
            ASSERT(v->first < 255);
            for(size_t j = 0; j < shape->numEdges; j++)
            {
                SatShape::Edge e = shape->edges[j];

                if(e.v0 == i)
                {
                    shape->vertEdges[num++] = e;
                    v->num++;
                }
                else if(e.v1 == i)
                {
                    shape->vertEdges[num++] = {(uint8_t)e.v1, (uint8_t)e.v0, (uint8_t) e.f1, (uint8_t)e.f0};
                    v->num++;
                }
            }
        }
        ASSERT(num == shape->numEdges*2);
    }

    calcBestFitSegment(shape->vertPos, shape->numVerts, &shape->seg0, &shape->seg1);

    hull.destroy();

    return shape;
}


void getSupport(Vec3 dir,  const Vec3 verts[], size_t numVerts, float *outSupp, size_t *outIndex)
{
    float maxDot = -INFINITY;
    size_t maxIndex = -1;

    size_t quadCount = 0;
#if 1
    quadCount = numVerts/4;
    if (quadCount)
    {
        __m128 d = _mm_loadu_ps(&dir.x);
        __m128 dx = _mm_shuffle_ps(d, d, _MM_SHUFFLE(0, 0, 0, 0));
        __m128 dy = _mm_shuffle_ps(d, d, _MM_SHUFFLE(1, 1, 1, 1));
        __m128 dz = _mm_shuffle_ps(d, d, _MM_SHUFFLE(2, 2, 2, 2));

        __m128 maxDots = FLOAT4_NEGINF.value;
        __m128i maxIndices = INT4_0000.value;
        __m128i indices = INT4_0123.value;
        for(size_t i = 0; i < quadCount; i++)
        {
            __m128 vx, vy, vz;
            transposeVec3(verts+4*i, vx, vy, vz);
            __m128 dot = dot4ps(dx, dy, dz, vx, vy, vz);
            __m128 gt = _mm_cmpgt_ps(dot, maxDots);
            maxDots = _mm_blendv_ps(maxDots, dot, gt);
            maxIndices = blendvepi32(maxIndices, indices, gt);
            indices = _mm_add_epi32(indices, INT4_4444.value);
        }

        maxIndex = horizontalArgmax(maxDots, maxIndices, &maxDot);
    }
#endif

    for(size_t i = 4*quadCount; i < numVerts; i++)
    {
        float dot = verts[i].dot(dir);
        if (dot > maxDot)
        {
            maxIndex = i;
            maxDot = dot;
        }
    }

    ASSERT(maxIndex != -1 && maxIndex < numVerts);

    *outIndex = maxIndex;
    *outSupp = maxDot;
}

/*
        bv1 = -bToA.R*bv0 - bToA.p;
        bn1 = -bToA.R*bn0 
        bd1 = -bn1.dot(bToA.p) + bd0




        bn0.dot(bv1) = bn0.dot(-bToA.R*bv0 - bToA.p) 
                     = bn0.dot(-bToA.R*bv0) - bn0.dot(bToA.p)
                     = -(aToB.R*bn0).dot(bv0) - bn0.dot(bToA.p)

                     */



/*
void transformAndNegate(SatGeom *out, const SatShape *a, Transform xfA, Transform xfB)
{
    Transform bToA = xfA.inverse().mul(xfB);

    for (size_t i = 0; i < a->numVerts; i++)
    {
        out->vertPos[i] = -bToA.mul(a->vertPos[i]);
    }

    for (size_t i = 0; i < a->numFaces; i++)
    {
        Vec3 n = bToA.rotate(a->facePlanes[i].normal);
        out->facePlanes[i].dist = n.dot(bToA.p) + a->facePlanes[i].dist;
        out->facePlanes[i].normal = -n;
    }
}
*/


void satCollideReference(const SatShape *a, Transform xfA, const SatShape *b, Transform xfB, SatResult *res)
{
    ASSERT(b->numFaces != 2);
    Transform bToA = xfA.inverse().mul(xfB);
    Transform aToB = bToA.inverse();

    res->vert = -1;
    res->face = -1;
    res->support = INFINITY;
    // Test MD against planes of a
    for (size_t i = 0; i < a->numFaces; i++)
    {
        Plane plane = a->facePlanes[i];
        Vec3 normal = -aToB.rotate(plane.normal);
        float planeDist = plane.dist - plane.normal.dot(bToA.p);
        float supp;
        size_t vertex;
        getSupport(normal, b->vertPos, b->numVerts, &supp, &vertex);
        supp += planeDist;
        if (supp < res->support)
        {
            res->type = SatResult::FACE_VERT;
            res->support = supp;
            res->vert = vertex;
            res->face = i;
            res->mtv = plane.normal;
            if (supp < 0)
            {
                res->mtv = xfA.rotate(res->mtv);
                return;
            }
        }
    }
    // Test MD against planes of b
    for (size_t i = 0; i < b->numFaces; i++)
    {
        Plane plane = b->facePlanes[i];
        Vec3 normal = -bToA.rotate(plane.normal);
        float planeDist = plane.dist - normal.dot(bToA.p);
        float supp;
        size_t vertex;
        getSupport(normal, a->vertPos, a->numVerts, &supp, &vertex);
        supp += planeDist;
        if (supp < res->support)
        {
            res->support = supp;
            res->vert = vertex;
            res->face = i;
            res->type = SatResult::VERT_FACE;
            res->mtv = normal;
            if (supp < 0)
            {
                res->mtv = xfA.rotate(res->mtv);
                return;
            }
        }
    }

    //drawSphere(sphereOrigin,0.99f, ColorWhite, 20, 20);

    // Test MD against edge cross products
    for(size_t j = 0; j < b->numEdges; j++)
    {
        SatShape::Edge eB = b->edges[j];
        // it is faster to pre transform all geometry but 
        // we don't because satCollideLocal does not
        Vec3 C   = -bToA.rotate(b->facePlanes[eB.f0].normal);
        Vec3 D   = -bToA.rotate(b->facePlanes[eB.f1].normal);
        Vec3 bv0 = -bToA.mul(b->vertPos[eB.v0]);
        Vec3 bv1 = -bToA.mul(b->vertPos[eB.v1]);
        Vec3 D_x_C = bv1 - bv0;

        for(size_t i = 0; i < a->numEdges; i++)
        {
            SatShape::Edge eA = a->edges[i];
            Vec3 A  = a->facePlanes[eA.f0].normal;
            Vec3 B  = a->facePlanes[eA.f1].normal;
            Vec3 av0 = a->vertPos[eA.v0];
            Vec3 av1 = a->vertPos[eA.v1];
            Vec3 B_x_A = av1 - av0;

            float CBA = C.dot(B_x_A);
            float DBA = D.dot(B_x_A);
            float ADC = A.dot(D_x_C);
            float BDC = B.dot(D_x_C);

            //if (i == 0) drawArcBetween(sphereOrigin, C, D, 1.0f, COLOR_BLUE, true);

            float eps = -0.0001f;
            if(CBA*DBA < eps && ADC*BDC < eps && CBA*BDC < eps)
            {
                // Cast the ray CD against the plane containing the tri ABO
                // (C+(D-C)*t) dot (BxA) == 0
                // C dot BxA + (D-C) dot BxA*t == 0
                // CBA + (D dot BxA - C dot BxA) t == 0
                // t = -CBA/(DBA-CBA)
                float t = -CBA / (DBA-CBA);
                Vec3 normal = Vec3::lerp(t, C, D).normalized();
                float support = normal.dot(av0+bv0);

                //drawPoint(sphereOrigin+normal,COLOR_RED);
                //if (support < res->support-3*SOLVER_DELTA_SLOP)
                if (support < res->support)
                {
                    res->mtv = normal; 
                    res->support = support;
                    res->type = SatResult::EDGE_EDGE;
                    res->edge1 = eA;
                    res->edge2 = eB;

                    if (support < 0)
                    {
                        //drawArrowTo( tB.mul( av0),  tB.mul(av1), COLOR_RED, 1.0f);
                        //drawArrowTo( tB.mul( -bv0), tB.mul(-bv1), COLOR_RED, 1.0f);
                        //drawArcBetween(g_sphereOrigin, C, D, 1.0f, COLOR_BLUE, true);
                        //drawArcBetween(g_sphereOrigin, A, B, 1.0f, COLOR_GREEN, true);
                        //drawPoint(g_sphereOrigin+normal, COLOR_PURPLE);

                        res->mtv = xfA.rotate(res->mtv);
                        return;
                    }
                }

            }
        }
    }
    //drawPointEx(sphereOrigin+tB.inverse().rotate(res->mtv)), COLOR_PURPLE, 1.5f);

    res->mtv = xfA.rotate(res->mtv);
}

void satCollideGraph(const SatShape *a, Transform xfA, const SatShape *b, Transform xfB, SatResult *res)
{
    ASSERT(b->numFaces != 2);
    ASSERT(a->vertEdges);

    res->support = INFINITY;

    //Vec3 sphereOrigin = {3,0,0};
    //for (int i = 0; i < b->numEdges; i++) drawArcBetween(sphereOrigin, b->facePlanes[b->edges[i].f0].normal, b->facePlanes[b->edges[i].f1].normal, 1.0f, COLOR_BLUE, false);
    //drawSphere(sphereOrigin, 0.99f, ColorWhite, 20, 20);

    uint8_t bFaceToVertexRegionA[SAT_MAX];
    memset(bFaceToVertexRegionA, 0xff, sizeof(bFaceToVertexRegionA[0])*b->numFaces);

    // For a face afi this map is the min support so far of all the arcs that intersected with arcs connecting afi 
    float aFaceMaxDot[SAT_MAX];
    uint8_t aFaceToVertexRegionB[SAT_MAX];
    memset(aFaceToVertexRegionB, 0xff, sizeof(aFaceToVertexRegionB[0])*b->numFaces);
    memset(aFaceMaxDot, 0xfe, sizeof(aFaceMaxDot[0])*a->numFaces);

    // as a float the bit pattern 0xfefefefe is -1.6947395e+38

    Transform bToA = xfA.inverse().mul(xfB);
    Transform aToB = bToA.inverse();

    // Find the vertex region that contains the first face of B
    // This is the face we will start our traversal from.
    {
        size_t faceIndex = 0;
        Plane plane = b->facePlanes[faceIndex];
        Vec3 normal = -bToA.rotate(plane.normal);
        float planeDist = plane.dist - normal.dot(bToA.p);
        float supp;
        size_t vertex;
        getSupport(normal, a->vertPos, a->numVerts, &supp, &vertex);
        supp += planeDist;
        if (supp < res->support)
        {
            res->support = supp;
            res->vert = vertex;
            res->face = faceIndex;
            res->type = SatResult::VERT_FACE;
            res->mtv = normal;
            if (supp < 0)
            {
                res->mtv = xfA.rotate(res->mtv);
                return;
            }
        }

        bFaceToVertexRegionA[faceIndex] = (uint8_t)vertex; 
    }

    for(int i = 0; i < b->numEdges; i++)
    {
        SatShape::Edge eB = b->edges[i];
        Vec3 rayOrigin = -bToA.rotate(b->facePlanes[eB.f0].normal);
        Vec3 rayEnd = -bToA.rotate(b->facePlanes[eB.f1].normal);
        Vec3 rayDir = rayEnd - rayOrigin;

        int curVertA = bFaceToVertexRegionA[eB.f0];
        ASSERT(curVertA != 0xff);

        //drawPoint(sphereOrigin+rayOrigin, COLOR_ORANGE);
        //drawArcBetween(sphereOrigin, rayOrigin, rayMid, 1.0f, COLOR_GREEN, true);
        //drawArcBetween(sphereOrigin, rayMid, rayEnd, 1.0f, COLOR_RED, true);

        // Traverse the arc from f0 to f1
        for (int it = 0; it < 32; it++)
        {
            // We are in the vertex region of curVertA
            // The vertex region is bounded by vert.num planes, it forms a convex linear cone.
            // To find which plane we exit we raycast against the planes.
            SatShape::EdgeList vert = a->verts[curVertA];
            SatShape::Edge eA = { 0 };
            float tMin = INFINITY;
            for (size_t j = 0; j < vert.num; j++)
            {
                SatShape::Edge portal = a->vertEdges[vert.first + j];
                
                // this can be precomputed and laid out as SoA for SIMD
                Vec3 portalNormal = a->vertPos[curVertA] - a->vertPos[portal.v1]; // <- this normal can be innacurate if hull is bad

                //drawArrow(sphereOrigin+ Vec3::lerp(0.5f, b->facePlanes[b->edges[portal.e].f0].normal, b->facePlanes[b->edges[portal.e].f1].normal).normalize(), portalNormal, COLOR_RED, 1.0f);

                float raySlope = portalNormal.dot(rayDir);
                if (raySlope < -0.001f)
                {
                    float tHit = -rayOrigin.dot(portalNormal) / raySlope;
                    if (tHit < tMin)
                    {
                        eA = portal;
                        tMin = tHit;
                    }
                }
            }

            //ASSERT(tMin >= 0);

            if(tMin < 1)
            {
                // We hit a plane before we reached the end of the arc.
                // This means we are changing vertex regions.

                curVertA = eA.v1;
                Vec3 normal =(rayOrigin+rayDir*tMin).normalized();
                Vec3 bVert0inA = -bToA.mul(b->vertPos[eB.v0]); // vert is in -B 
                float support = normal.dot(bVert0inA + a->vertPos[curVertA]);
                if (support < res->support)
                {
                    res->mtv = normal;
                    res->type = SatResult::EDGE_EDGE;
                    res->support = support;
                    res->edge1 = eA;
                    res->edge2 = eB;

                    if (support < 0)
                    {
                        // shapes aren't overlapping.

                        //drawPoint(vecAdd(sphereOrigin, normal), COLOR_PURPLE);
                        //SatVertex v = b->verts[curVertB];
                        //for(int j = 0; j < v.numPortals; j++)
                        //{
                        //    SatEdge e = b->edges[b->vertEdges[v.firstPortal + j].e];
                        //    drawArcBetween(sphereOrigin, b->facePlanes[e.f0].normal, b->facePlanes[e.f1].normal, 1.0f, COLOR_YELLOW, true);
                        //}

                        res->mtv = xfA.rotate(res->mtv);
                        return;
                    }
                }


                // relax both faces of A connected to the arc we intersected
                Vec3 bVert1inA = -bToA.mul(b->vertPos[eB.v1]); // vert is in -B 

                // There might be a way to do less work here

                // f0
                {
                    size_t fa = eA.f0;
                    Plane aPlane = a->facePlanes[fa];
                    float dot_b0 = aPlane.normal.dot(bVert0inA);
                    float dot_b1 = aPlane.normal.dot(bVert1inA);
                    if (dot_b0 > aFaceMaxDot[fa])
                    {
                        aFaceMaxDot[fa] = dot_b0;
                        aFaceToVertexRegionB[fa] = eB.v0; 
                    }
                    if (dot_b1 > aFaceMaxDot[fa])
                    {
                        aFaceMaxDot[fa] = dot_b1;
                        aFaceToVertexRegionB[fa] = eB.v1; 
                    }
                }

                // f1
                {
                    size_t fa = eA.f1;
                    Plane aPlane = a->facePlanes[fa];
                    float dot_b0 = aPlane.normal.dot(bVert0inA);
                    float dot_b1 = aPlane.normal.dot(bVert1inA);
                    if (dot_b0 > aFaceMaxDot[fa])
                    {
                        aFaceMaxDot[fa] = dot_b0;
                        aFaceToVertexRegionB[fa] = eB.v0; 
                    }
                    if (dot_b1 > aFaceMaxDot[fa])
                    {
                        aFaceMaxDot[fa] = dot_b1;
                        aFaceToVertexRegionB[fa] = eB.v1; 
                    }
                }
                

                // Now we keep continue starting from the intersection point we just hit.
                rayOrigin = normal;
                rayDir = rayEnd - rayOrigin;

                //drawPoint(vecAdd(sphereOrigin, normal), COLOR_RED);
            }
            else
            {
                // We didn't hit anything which means f1 is in the region curVertA
                break;
            }
        }

        {
            bFaceToVertexRegionA[eB.f1] = curVertA;

            // Find the plane of f1 in A space
            Plane bPlane = b->facePlanes[eB.f1];
            Vec3 bPlaneNormalInA = -bToA.rotate(bPlane.normal);
            float bPlaneDistInA = bPlane.dist - bPlaneNormalInA.dot(bToA.p);

            float support = rayEnd.dot(a->vertPos[curVertA]) + bPlaneDistInA;
            if(support < res->support)
            {
                res->mtv = rayEnd;
                res->type = SatResult::VERT_FACE;
                res->support = support;
                res->vert = curVertA;
                res->face = eB.f1;

                if (support < 0)
                {
                    // shapes aren't overlapping.

                    //drawPoint(sphereOrigin + rayEnd, COLOR_PURPLE);

                    //SatVertex v = b->verts[curVertB];
                    //for(int j = 0; j < v.numPortals; j++)
                    //{
                    //    SatEdge e = b->edges[b->vertEdges[v.firstPortal + j].e];
                    //    drawArcBetween(sphereOrigin, b->facePlanes[e.f0].normal, b->facePlanes[e.f1].normal, 1.0f, COLOR_YELLOW, true);
                    //}

                    res->mtv = xfA.rotate(res->mtv);
                    return;
                }
            }

        }
    }


    // Color any remaining faces of A that haven't been had their arcs intersected
    uint8_t queue[SAT_MAX];
    size_t queueLo = 0;
    size_t queueHi = 0;
    for (size_t i = 0; i < a->numFaces; i++)
    {
        if (aFaceToVertexRegionB[i] != 0xff)
        {
            queue[queueHi++] = i;
        }
    }

    while (queueLo < queueHi) 
    {
        size_t f = queue[queueLo++];
        uint8_t region = aFaceToVertexRegionB[f];

        SatShape::EdgeList face = a->faces[f];
        for (size_t i = 0; i < face.num; i++) 
        {
            SatShape::Edge nbr = a->faceEdges[face.first + i];
            if (aFaceToVertexRegionB[nbr.f1] == 0xff)
            {
                Vec3 bVertInA = -bToA.mul(b->vertPos[region]);

                queue[queueHi++] = nbr.f1;
                aFaceToVertexRegionB[nbr.f1] = region;
                aFaceMaxDot[nbr.f1] = a->facePlanes[nbr.f1].normal.dot(bVertInA);


            }
        }
    }

    for (size_t faceA = 0; faceA < a->numFaces; faceA++)
    {
        size_t vertB = aFaceToVertexRegionB[faceA];
        ASSERT(vertB != 0xff);
        Plane aPlane = a->facePlanes[faceA];

        float supp = aPlane.dist + aFaceMaxDot[faceA];
        if (supp < res->support)
        {
            res->mtv = aPlane.normal;
            res->type = SatResult::FACE_VERT;
            res->support = supp;
            res->vert = vertB;
            res->face = faceA;

            if (supp < 0)
            {
                res->mtv = xfA.rotate(res->mtv);
                return;
            }
        }

    }

    res->mtv = xfA.rotate(res->mtv);
}



size_t findIncidentFace(const SatShape *shape, Vec3 dir, size_t vertIdx)
{
    ASSERT(vertIdx != -1 && vertIdx < shape->numVerts);

    size_t faceIndex = -1;
    float minProj = INFINITY;

    SatShape::EdgeList vert = shape->verts[vertIdx];

    for (size_t i = 0; i < vert.num; i++)
    {
        size_t face = shape->vertEdges[vert.first+i].f0;
        Plane p = shape->facePlanes[face];
        float proj = p.normal.dot(dir);
        if (proj < minProj)
        {
            faceIndex = face;
            minProj = proj;
        }

    }

    ASSERT(faceIndex != -1);
    return faceIndex;
}

struct TmpContact
{
    Vec3 p;
    float dist;
};

bool computeFaceContacts(
    const SatShape *shapeA, Transform xfA, size_t faceIndexA,
    const SatShape *shapeB, Transform xfB, size_t faceIndexB,
    Vec3 _normalFromBToA,
    SolverManifold *manifold,
    size_t indexA, size_t indexB)
{
    ZoneScoped;

    // TODO: make this more robust when faces are nearly coplanar.

    TmpContact contacts[32];
    size_t numContacts = 0;

    {
        ZoneScopedN("Clipping");

        SatShape::EdgeList faceA = shapeA->faces[faceIndexA];


        Transform invB = xfB.inverse();
        Transform aToB = invB.mul(xfA);

        Vec3 pointsA[32];
        for(size_t i = 0; i < faceA.num; i++)
        {
            pointsA[i] = aToB.mul(shapeA->vertPos[shapeA->faceEdges[faceA.first + i].v0]);
        }
        size_t numPointsA = faceA.num;

        const Vec3 normalFromBToA = invB.rotate(_normalFromBToA);

        //{
        //    Vec3 n =xfB.rotate(normalFromBToA);
        //    drawPlane(xfB.p+ n* planeBDist, n, COLOR_YELLOW, 1.0f);
        //}

        //for (size_t i1 = 0, i0 = numPointsA-1; i1 < numPointsA; i0=i1, i1++) drawLine(xfB.mul(pointsA[i0]), xfB.mul(pointsA[i1]), COLOR_GREEN);

        Vec3 clipped[32];
        float planeBDist = -INFINITY;
        SatShape::EdgeList faceB = shapeB->faces[faceIndexB];
        for(size_t i1 = 0, i0 = faceB.num-1; i1 < faceB.num; i0=i1, i1++)
        {
            Vec3 v0 = shapeB->vertPos[shapeB->faceEdges[faceB.first + i0].v0];
            Vec3 v1 = shapeB->vertPos[shapeB->faceEdges[faceB.first + i1].v0];

            planeBDist = Math::max(planeBDist, v0.dot(normalFromBToA));

            Vec3 e = v1 - v0;
            Vec3 n = e.cross(normalFromBToA).normalized();

            //drawArrowTo(xfB.mul(v0), xfB.mul(v1),COLOR_PURPLE,1.0f);
            //Vec3 center = vecLerp(0.5f, v0, v1);
            //drawArrow(xfB.mul(center),xfB.rotate(n)*0.2f, COLOR_PURPLE, 1.0f);

            float d = n.dot(v0);
            numPointsA = clipPolygonAgainstPlane(n, d, pointsA, numPointsA, clipped);
            memcpy(pointsA, clipped, sizeof(clipped[0])*numPointsA);
        }

        Plane planeB = shapeB->facePlanes[faceIndexB];
        for(size_t i = 0; i < numPointsA; i++)
        {
            float d0 = normalFromBToA.dot(pointsA[i])-planeBDist;
            float d1 = planeB.normal.dot(pointsA[i]) - planeB.dist;
            if(d0 <=  -SOLVER_DELTA_SLOP && d1 <= 0 /*-SOLVER_DELTA_SLOP*/)
            {
                contacts[numContacts++] ={pointsA[i], d0};
            }
        }
    }

    //for (size_t i1 = 0, i0 = numPointsA-1; i1 < numPointsA; i0=i1, i1++) drawLine(xfB.mul(pointsA[i0]), xfB.mul(pointsA[i1]), COLOR_GREEN);

    if(numContacts == 0)
    {
        return false;
    }

    if(numContacts > 4)
    {
        ZoneScopedN("Reducing");

        float maxDist = 0;
        size_t maxDistIndex = -1;
        for(size_t i = 1; i < numContacts; i++)
        {
            float dist = contacts[0].p.distSq(contacts[i].p);
            if(dist > maxDist)
            {
                maxDistIndex = i;
                maxDist = dist;
            }
        }
        std::swap(contacts[1], contacts[maxDistIndex]);

        Vec3 edge = contacts[1].p - contacts[0].p;

        {
            size_t maxAreaIndex;
            float maxArea = -INFINITY;
            for(size_t i = 2; i < numContacts; i++)
            {
                float area = edge.cross(contacts[i].p - contacts[0].p).length();
                if(area > maxArea)
                {
                    maxAreaIndex = i;
                    maxArea = area;
                }
            }
            std::swap(contacts[2], contacts[maxAreaIndex]);
        }

        {
            size_t maxAreaIndex;
            float maxArea = -INFINITY;
            for(size_t i = 3; i < numContacts; i++)
            {
                float area = edge.cross(contacts[i].p - contacts[0].p).length();
                if(area > maxArea)
                {
                    maxAreaIndex = i;
                    maxArea = area;
                }
            }
            std::swap(contacts[3], contacts[maxAreaIndex]);
        }

        numContacts = 4;
    }


    {
        ASSERT(indexA != indexB);
        manifold->normal = _normalFromBToA;
        Mat3 basis = Mat3::basis(manifold->normal);
        manifold->tangent = basis.c0;
        manifold->bitangent = basis.c1;
        manifold->normal = basis.c2;
        manifold->indexA = (uint16_t)indexA;
        manifold->indexB = (uint16_t)indexB;
        manifold->numContacts = numContacts;
        for(size_t i = 0; i < numContacts; i++)
        {
            const TmpContact tmp = contacts[i];
            float d = tmp.dist;
            Vec3 p = xfB.mul(tmp.p);
            SolverContact *c = &manifold->contacts[i];
            c->bias = d > SOLVER_DELTA_SLOP ? 0 : SOLVER_BETA*(d-SOLVER_DELTA_SLOP)/SOLVER_DT; //delta < deltaSlop ? 0.8f*(delta-deltaSlop)/DT : 0;
            c->p = p;
            c->normalLambda = 0;
            c->tangentLambda = 0;
            c->bitangentLambda = 0;

            DRAW_NORMALS(
                drawPointEx(p, COLOR_GREEN, 0.5f);
                drawArrow(p, manifold->normal, COLOR_GREEN, 0.5f);
            );
        }
    }

    //for (size_t i = 0; i < numContacts; i++) drawPoint(contacts[i].p, COLOR_GREEN);
    return true;
}

static Vec3 correctNormal(Vec3 a, Vec3 b, Vec3 n, Vec3 debugP)
{
    if (a.x == 0 && a.y == 0 && a.z == 0 
        || b.x == 0 && b.y == 0 && b.z == 0)
    {
        return n;
    }


    float tol = 0.0001f;
    if (   fabsf(a.x-b.x) < tol
        && fabsf(a.y-b.y) < tol 
        && fabsf(a.z-b.z) < tol) 
    {
        return a;
    }

    Vec3 p = a.cross(b).normalized();
    Vec3 pA = p.cross(a);
    Vec3 pB = p.cross(b);
    n = n.clip(p);

    DRAW_NORMALS(
        Vec3 p1 = debugP + a*0.4f, p2 = debugP + b*0.4f;
        drawPoint(p1, COLOR_LIGHT_BLUE);
        drawPoint(p2, COLOR_LIGHT_BLUE);
        drawLine(debugP, p1, COLOR_LIGHT_BLUE);
        drawLine(debugP, p2, COLOR_LIGHT_BLUE);
        drawLine(p1, p2, COLOR_LIGHT_BLUE);
    );

    float dA = n.dot(pA);
    float dB = n.dot(pB);
    if (dA < 0)
        return a;
    if (dB >= 0)
        return b;
    return n.normalized();
}


bool generateSatVsSatContacts(const SatShape *shapeA, Transform xfA, size_t indexA,
                             const SatShape *shapeB, Transform xfB, size_t indexB,
                             SolverManifold *manifold, bool useGraph)
{
    SatResult res;
    ASSERT(shapeA->numVerts > 3);
    ASSERT(shapeB->numVerts > 3);

    if (useGraph)
        satCollideGraph(shapeA, xfA, shapeB, xfB, &res);
    else
        satCollideReference(shapeA, xfA, shapeB, xfB, &res);

    if(res.support < 0)
    {
        return false;
    }

    manifold->numContacts = 0;

    if (res.type == SatResult::EDGE_EDGE)
    {
        Vec3 contactNormal = res.mtv;

        Vec3 contactNormalA = xfA.R.transpose()*contactNormal;
        size_t faceA = res.edge1.f1;
        if(shapeA->facePlanes[res.edge1.f0].normal.dot(contactNormalA) > shapeA->facePlanes[res.edge1.f1].normal.dot(contactNormalA))
        {
            faceA = res.edge1.f0;
        }

        size_t faceB = res.edge2.f1;
        Vec3 contactNormalB = xfB.R.transpose()*contactNormal;
        if (shapeB->facePlanes[res.edge2.f0].normal.dot(contactNormalB) < shapeB->facePlanes[res.edge2.f1].normal.dot(contactNormalB))
        {
            faceB = res.edge2.f0;
            //drawArrowTo(tB.mul(shapeB->verts[res.edge2.v0]), tB.mul(shapeB->verts[res.edge2.v1]), COLOR_LIGHT_BLUE, 1.5f);
        }


        computeFaceContacts(
            shapeB, xfB, faceB,
            shapeA, xfA, faceA,
            contactNormal, manifold,
            indexA, indexB);

    }
    else if (res.type ==  SatResult::FACE_VERT)
    {
        ASSERT(res.face != -1);
        ASSERT(res.vert != -1);
        // A Face, B Vert
        size_t faceA = res.face;
        size_t faceB = findIncidentFace(shapeB, xfB.R.transpose()*res.mtv, res.vert);

        //drawPoint(xfB.mul(shapeB->vertPos[res.vert]), COLOR_RED);
        //drawArrow(xfB.mul(shapeB->vertPos[res.vert]), -res.mtv, COLOR_RED, 1);

        computeFaceContacts(
            shapeB, xfB, faceB,
            shapeA, xfA, faceA,
            res.mtv, manifold,
            indexA, indexB);

    }
    else if (res.type == SatResult::VERT_FACE)
    {
        ASSERT(res.face != -1);
        ASSERT(res.vert != -1);
    // B Face, A Vert
        size_t faceA = findIncidentFace(shapeA, xfA.R.transpose()*-res.mtv, res.vert);
        size_t faceB = res.face;
        computeFaceContacts(shapeA, xfA, faceA,
                                    shapeB, xfB, faceB,
                                    -res.mtv, manifold,
                                    indexB, indexA);
    }


    return manifold->numContacts != 0;
}

static const TriShapeData g_triShapeData = 
{
    /*faces*/       {{0, 3}, {3, 3},},
    /*verts*/       {{0, 2}, {2, 2}, {4, 2},},
    /*edges*/       {{1, 0, 1, 0}, {2, 1, 1, 0}, {0, 2, 1, 0},},
    /*faceEdges*/   {{0, 1, 0, 1}, {1, 2, 0, 1}, {2, 0, 0, 1}, {2, 1, 1, 0}, {1, 0, 1, 0}, {0, 2, 1, 0},},
    /*facePlanes*/  {{0, 1, 0, 1}, {0, 2, 1, 0}, {1, 0, 1, 0}, {1, 2, 0, 1}, {2, 1, 1, 0}, {2, 0, 0, 1},},
};


void TriShape::setFromVerts(const Vec3 points[3])
{
    if (!sat.numFaces)
    {
        data = g_triShapeData;
        sat.numFaces   = 2;
        sat.numVerts   = 3;
        sat.numEdges   = 3;
        sat.faces      = data.faces;
        sat.verts      = data.verts;
        sat.faceEdges  = data.faceEdges;
        sat.vertEdges  = data.vertEdges;
        sat.edges      = data.edges;
        sat.vertPos    = vertPos;
        sat.facePlanes = facePlanes;
    }

    vertPos[0] = points[0];
    vertPos[1] = points[1];
    vertPos[2] = points[2];
    Vec3 triNormal = triComputeNormal(points[0], points[1], points[2]);
    float dist = points[0].dot(triNormal);
    facePlanes[0] = {triNormal,  dist};
    facePlanes[1] = {-triNormal, -dist};
}


size_t generateTriMeshVsSatContacts(const TriMeshShape  *triMesh, Transform xfA, size_t indexA,
                                 const SatShape*shapeB, Transform xfB, size_t indexB,
                                 SolverManifold *manifolds, bool useGraph)
{
    ASSERT(shapeB->numVerts > 3);

    // aToB
    Mat3 invBR = xfB.R.transpose();
    Transform bToA =xfA.inverse().mul(xfB);
    Aabb boundsB = Aabb::fromRadius(bToA.p, shapeB->boundingRadius);
    //drawWireBox(boundsB.p0, boundsB.p1, COLOR_RED);

    const size_t MAX_TRIS = 512;
    uint32_t overlappingTris[MAX_TRIS];
    size_t numOverlappingTris = triMesh->bvh.overlap(boundsB, overlappingTris, MAX_TRIS);

    if (!numOverlappingTris)
    {
        return 0;
    }

    //SatGeom geomB;
    //transformAndNegate(&geomB, shapeB, xfB, xfA);

    static TriShape triShape;

    size_t numManifolds = 0;
    for (size_t i = 0; i < numOverlappingTris; i++)
    {
        size_t tri = overlappingTris[i];

        const TriMeshShape::TriData *triData = &triMesh->tris[tri];

        Vec3 debugP = {0};
        DRAW_NORMALS(debugP = triComputeCentroid(triData->verts[0], triData->verts[1], triData->verts[2]););

        // todo: We don't need to always copy everything
        triShape.setFromVerts(triData->verts);
        Vec3 triNormal = triShape.facePlanes[0].normal;

        //drawArrow(vecLerp(0.5f,p0,p1)+ myNormal, vecCross(p1 - p0, myNormal)*0.1f, COLOR_ORANGE, 1.0f);


        SolverManifold *manifold = manifolds+numManifolds;
        {
            SatResult res;
            const SatShape *shapeA = &triShape.sat;
            triShape.sat.seg0 = triData->verts[0];
            triShape.sat.seg1 = triData->verts[1];

            if (useGraph)
            {
                satCollideGraph(shapeA, xfA, shapeB, xfB, &res);
            }
            else
            {
                satCollideReference(shapeA, xfA, shapeB, xfB, &res);
            }


            if(res.support < 0)
            {
                continue;
            }

            // Ignore "backface" collisions
            float mtvDotTriNormal = res.mtv.dot(triNormal);
            if(mtvDotTriNormal <= 0.02f)
            {
                DRAW_NORMALS(drawArrow(debugP, res.mtv*0.5f, COLOR_RED, 1.0f););
                DRAW_NORMALS(drawArrow(debugP, triNormal, COLOR_PURPLE, 1.0f););
                continue;
            }

            Vec3 correctedNormal = res.mtv;
            size_t faceB = -1;

            if (res.type == SatResult::EDGE_EDGE)
            {
                correctedNormal = correctNormal(triNormal, triData->edgeNormals[res.edge1.v1], res.mtv, debugP);
                // TODO: maybe not necessary

                faceB = res.edge2.f1;
                Vec3 mtvB = invBR*res.mtv;
                if(shapeB->facePlanes[res.edge2.f0].normal.dot(mtvB) < shapeB->facePlanes[res.edge2.f1].normal.dot(mtvB))
                {
                    faceB = res.edge2.f0;
                }
            } 
            else if (res.type == SatResult::FACE_VERT)
            {
                if (res.face == 0) faceB = findIncidentFace(shapeB, invBR*res.mtv, res.vert);
            }
            else if (res.type ==  SatResult::VERT_FACE)
            {
                if (res.mtv.dot(triNormal) >= 0.9f
                    || res.mtv.dot(triData->vertNormals[res.vert]) >= 0.9f)
                {
                    faceB = res.face;
                }
                    //correctedNormal = triNormal;
            }
            else
            {
                ASSERT(0);
            }

            if (faceB != -1)
            {
                if(correctedNormal.dot(res.mtv) < 0.0f)
                    continue;

                DRAW_NORMALS(
                    drawArrow(debugP, res.mtv, COLOR_ORANGE, 1.0f);
                    drawArrow(debugP, correctedNormal*0.5f, COLOR_YELLOW, 1.0f);
                );
                size_t faceA = 0;
                if (computeFaceContacts(shapeB, xfB, faceB, shapeA, xfA, faceA, correctedNormal, manifold, indexA, indexB))
                {
                    numManifolds++;
                }
            }
        }
    }
    return numManifolds;
}

struct TriEdge 
{
    size_t nbrId;
    size_t edge;
};

#include <unordered_map>
#include <ctype.h>
bool loadTriMeshShape(TriMeshShape *shape, const char *filename)
{
    std::vector<Vec3> verts;
    std::vector<uint32_t> face;
    std::vector<uint32_t> indices;
    {
        FILE *file;
        fopen_s(&file, filename, "r");
        if(!file)
        {
            return false;
        }

        char line[1024];
        while(fgets(line, sizeof(line), file))
        {
            if(!strncmp(line, "v ", 2))
            {
                Vec3 v;
                sscanf_s(line, "v %f %f %f", &v.x, &v.y, &v.z);
                verts.push_back(v);
            }
            else if(!strncmp(line, "f ", 2))
            {
                face.clear();
                char *at = line+1;

                while(1)
                {
                    while(isspace(*at)) at++;

                    if(!isdigit(*at))
                    {
                        break;
                    }

                    face.push_back(strtol(at, &at, 10)-1);
                    if(*at == '/')
                    {
                        at++;
                        while(isdigit(*at)) at++;
                        if(*at == '/')
                        {
                            at++;
                            while(isdigit(*at)) at++;
                        }
                    }
                }

                ASSERT(face.size() >= 3);
                if(face.size() == 3)
                {
                    indices.push_back(face[0]);
                    indices.push_back(face[1]);
                    indices.push_back(face[2]);
                }
                else
                {
                    // Triangulate face assuming it's convex
                    for(size_t i = 1; i < face.size()-1; i++)
                    {
                        indices.push_back(face[0]);
                        indices.push_back(face[i]);
                        indices.push_back(face[i+1]);
                    }
                }
            }
        }
        fclose(file);
        ASSERT(verts.size() >= 3);
        ASSERT(indices.size() >= 3);
    }

    createDebugMesh((DebugMeshId)shape, &verts[0], (int)verts.size(), &indices[0], (int)indices.size());

    shape->numTris = indices.size()/3;
    shape->tris = new TriMeshShape::TriData[shape->numTris];

    std::vector<Vec3> vertNormals;
    vertNormals.resize(verts.size(), {});

    std::unordered_map<uint64_t, TriEdge> nbrs;

    for(size_t i = 0; i < shape->numTris; i++)
    {
        shape->tris[i].verts[0] = verts[indices[3*i + 0]];
        shape->tris[i].verts[1] = verts[indices[3*i + 1]];
        shape->tris[i].verts[2] = verts[indices[3*i + 2]];
    }

    for (size_t i = 0; i < shape->numTris; i++)
    {
        Vec3 a0 = shape->tris[i].verts[0];
        Vec3 a1 = shape->tris[i].verts[1];
        Vec3 a2 = shape->tris[i].verts[2];
        Vec3 nA = triComputeNormal(a0,a1,a2);

        size_t base = i*3;
        size_t triId = i+1;
        for (size_t j0 = 0; j0 < 3; j0++)
        {
            size_t j1 = (j0 + 1)%3;
            TriEdge edge = {triId, j0};

            uint64_t v0 = indices[base + j0];
            uint64_t v1 = indices[base + j1];
            uint64_t edgeId = (v0+1) | ((v1+1) << 32);
            uint64_t twinId = (v1+1) | ((v0+1) << 32);

            nbrs.insert({edgeId, edge});
            auto it = nbrs.find(twinId);
            if (it != nbrs.end())
            {
                const TriEdge twin = it->second;
                ASSERT(twin.nbrId);
                size_t nbr = twin.nbrId-1;
                Vec3 b0 = shape->tris[nbr].verts[0];
                Vec3 b1 = shape->tris[nbr].verts[1];
                Vec3 b2 = shape->tris[nbr].verts[2];
                Vec3 nB = triComputeNormal(b0, b1, b2);
                shape->tris[i].edgeNormals[j0]  = nB;
                shape->tris[nbr].edgeNormals[twin.edge] = nA;
            }


            vertNormals[indices[base+j0]] = vertNormals[indices[base+j0]] + nA;
        }
    }
    for (size_t i = 0; i < vertNormals.size(); i++)
    {
        vertNormals[i] = vertNormals[i].normalized();
    }
    for (size_t i = 0; i < shape->numTris; i++)
    {
        for (size_t j = 0; j < 3; j++)
        {
            shape->tris[i].vertNormals[j] = vertNormals[indices[3*i+j]];
        }
    }

    // For each vertex find the triangles connected to it

    shape->bvh.buildFromTriangles(&verts[0], &indices[0], shape->numTris);

    return true;
}

static void solveConstraint(
    SolverBody *a, SolverBody *b,
    Vec3 Jva, Vec3 Jwa,
    Vec3 Jvb, Vec3 Jwb,
    float bias,
    float *accumLambdaOut, float minLambda,float maxLambda)
{
    // mC can be precalculated

    Vec3 iIA_Jwa = a->iI*Jwa;
    Vec3 iIB_Jwb = b->iI*Jwb;
    Vec3 iMA_Jva = Jva*a->iM;
    Vec3 iMB_Jvb = Jvb*b->iM;

    float mC = 1.0f/(a->iM + Jwa.dot(iIA_Jwa) + b->iM + Jwb.dot(iIB_Jwb));
    float lambda = -mC*(Jva.dot(a->v) + Jwa.dot(a->w) + Jvb.dot(b->v) + Jwb.dot(b->w) + bias);
    float accumLambda = *accumLambdaOut;
    float newLambda = Math::clamp(accumLambda + lambda, minLambda, maxLambda);
    lambda = newLambda - accumLambda;
    *accumLambdaOut = newLambda;

    a->v += iMA_Jva*lambda;
    a->w += iIA_Jwa*lambda;
    b->v += iMB_Jvb*lambda;
    b->w += iIB_Jwb*lambda;
}

void solveConstraints(SolverManifold manifolds[], size_t numManifolds, 
                      SolverBody bodies[], size_t numBodies)
{
    for(size_t outerIt = 0; outerIt < 32; outerIt++)
    {
        for(size_t i = 0; i < numManifolds; i++)
        {
            SolverManifold *m = &manifolds[i];
            SolverBody *a = &bodies[m->indexA];
            SolverBody *b = &bodies[m->indexB];

            Vec3 t = m->tangent;
            Vec3 bt = m->bitangent;
            Vec3 n = m->normal;

            Vec3 ap = a->p;
            Vec3 bp = b->p;


            for(size_t j = 0; j < m->numContacts; j++)
            {
                SolverContact *c = m->contacts+j;
                float maxLambda = 0.05f * c->normalLambda;
                Vec3 cp = c->p;
                Vec3 ac = cp-ap;
                Vec3 bc = cp-bp;
                solveConstraint(a, b, bt, ac.cross(bt),-bt, bt.cross(bc),   0,          &c->bitangentLambda,    -maxLambda, maxLambda);
                solveConstraint(a, b,  t, ac.cross(t), -t,  t.cross(bc),    0,          &c->tangentLambda,      -maxLambda, maxLambda);
                solveConstraint(a, b, -n, n.cross(ac),  n,  bc.cross(n),    c->bias,    &c->normalLambda,       0,          INFINITY);
            }
        }
    }
}


Vec3 g_sphereOrigin = {4,0,0};

void drawBvh(Bvh *bvh)
{
    size_t stack[64];
    stack[0] = 0;
    size_t stackSize = 1;

    for (size_t i = 0; i < bvh->m_numNodes; i++)
    {
        drawAabb(bvh->m_nodes[i].bounds, COLOR_RED);
    }
}

static void drawGaussEdge(SatShape::Edge edge, const SatShape *shape, Transform xf, Color color, bool drawNormal)
{
    Vec3 n0 = shape->facePlanes[edge.f0].normal;
    Vec3 n1 = shape->facePlanes[edge.f1].normal;
    Vec3 e = shape->vertPos[edge.v1] - shape->vertPos[edge.v0];
    Vec3 t0 = e.cross(n0).normalized();
    Vec3 t1 = n1.cross(e).normalized();
    Vec3 mid = Vec3::lerp(0.5f, n0+ t0, n1+ t1).normalized();


    Vec3 a = xf.rotate(n0);
    Vec3 b = xf.rotate(mid);
    Vec3 c = xf.rotate(n1);

    if (drawNormal)
    {
        Vec3 normal =(shape->vertPos[edge.v1] - shape->vertPos[edge.v0]).normalized(); 
        drawArrow(b+ g_sphereOrigin, xf.rotate(normal*0.1f),  color, 0.3f);
    }

    drawArcBetween(g_sphereOrigin, a, b, 1.0f, color, false);
    drawArcBetween(g_sphereOrigin, b, c, 1.0f, color, false);
}

void drawGaussRegion(size_t vertex, const SatShape *shape, Transform xf, Color color)
{
    for (size_t i = 0; i < shape->verts[vertex].num; i++)
    {
        SatShape::Edge edge = shape->vertEdges[shape->verts[vertex].first + i];
        drawGaussEdge(edge, shape, xf, color, true);
    }
}

void drawGaussPoint(Vec3 p, Color color)
{
    drawPointEx(g_sphereOrigin+ p, color, 0.2f);
}

void drawGaussMap(const SatShape *shape, Transform xf, Color color)
{
    drawSphere(g_sphereOrigin, 0.9999f, COLOR_WHITE, 20, 20);
    for (size_t i = 0; i < shape->numEdges; i++)
    {
        drawGaussEdge(shape->edges[i], shape, xf, color, false);
    }
}

void drawConvexity(const TriMeshShape *shape)
{
    for (size_t i = 0; i < shape->numTris; i++)
    {
        TriMeshShape::TriData tri = shape->tris[i];
        for (size_t j = 0; j < 3; j++)
        {
            size_t j0 = j;
            size_t j1 = (j+1)%3;

            Vec3 mid = Vec3::lerp(0.5f, tri.verts[j0], tri.verts[j1]);
            drawArrow(mid, tri.edgeNormals[j0], COLOR_PURPLE, 1.0f);
        }
    }
}

void drawSatShape(const SatShape * shape, Transform transform, Color color)
{
    float det = transform.R.det();

    for(size_t i = 0; i < shape->numFaces; i++)
    {
        SatShape::EdgeList f = shape->faces[i];
        Vec3 v0 = transform.mul(shape->vertPos[shape->faceEdges[f.first].v0]);
        for(size_t j = 1; j < f.num-1; j++)
        {
            Vec3 v1 = transform.mul(shape->vertPos[shape->faceEdges[f.first+j].v0]);
            Vec3 v2 = transform.mul(shape->vertPos[shape->faceEdges[f.first+j+1].v0]);
            if (det < 0)
            {
                _drawWireTri(v0, v2, v1, color);
            }
            else
            {
                _drawWireTri(v0, v1, v2, color);
            }
        }
    }

#if 0
    for (size_t i = 0; i < shape->numEdges; i++)
    {
        SatShape::Edge e = shape->edges[i];
        Vec3 v0 = transform.mul(shape->vertPos[e.v0]);
        Vec3 v1 = transform.mul(shape->vertPos[e.v1]);
        drawLine(v0, v1, COLOR_BLUE);
    }
#endif
}

