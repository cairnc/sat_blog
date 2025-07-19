#include "collision.h"

#define _CRT_SECURE_NO_WARNINGS
#include <string.h>
#include <stdarg.h>
#include <stdio.h>
#include <vector>
#include <unordered_map>

//
// Graphics stuff
//

typedef void* VboId;

struct DebugMesh 
{
    DebugMeshId id;
    VboId vbo;
    size_t numVerts;
};

static std::unordered_map<DebugMeshId, DebugMesh> g_debugMeshes;

struct DebugVertex
{
    Vec3 pos;
    Vec3 normal;
    Vec2 texCoord;
    Color color;
};


struct DrawMeshCommand
{
    DebugMeshId meshId;
    Mat4 transform; 
    Color color;
};

struct DebugDrawData
{
    Mat4 screenProj;
    Mat4 worldViewProj;
    Vec3 lightDir;
    Vec3 eyePos;

    std::vector<DrawMeshCommand> meshes;
    std::vector<DebugVertex> tris;
    std::vector<DebugVertex> lines;
    std::vector<DebugVertex> wireTris;
};




template <typename T> size_t sizeInBytes(const std::vector<T> &v) { return (int)(sizeof(T)*v.size()); }
template <typename T> size_t capacityInBytes(const std::vector<T> &v) { return (int)(sizeof(T)*v.capacity()); }

DebugDrawData g_debugDrawData;

void drawMesh(DebugMeshId meshId, Transform transform, Color color)
{
    DrawMeshCommand cmd;
    cmd.meshId = meshId;
    cmd.transform = Mat4::setTR(transform.p, transform.R);
    cmd.color = color;
    g_debugDrawData.meshes.push_back(cmd);
}

// TODO: use Array::expand
void _drawTriVert(Vec3 p, Vec3 n,Color color)
{
    g_debugDrawData.tris.push_back({p,n,{0},color});
}

void _drawWireTriVert(Vec3 p, Vec3 n,Color color)
{
    g_debugDrawData.wireTris.push_back({p,n,{0},color});
}

void _drawTriVertV(DebugVertex vertex)
{
    g_debugDrawData.tris.push_back(vertex);
}

void _drawLineVert(Vec3 p, Color c)
{
    g_debugDrawData.lines.push_back({p,{0},{0},c});
}

void _drawLineVertV(DebugVertex vertex)
{
    g_debugDrawData.lines.push_back(vertex);
}

void _drawQuad(Vec3 a, Vec3 b, Vec3 c, Vec3 d, Vec3 n, Color color)
{
    _drawTriVert(a, n, color);
    _drawTriVert(c, n, color);
    _drawTriVert(b, n, color);
    _drawTriVert(a, n, color);
    _drawTriVert(d, n, color);
    _drawTriVert(c, n, color);
}

void _drawLine(Vec3 a, Vec3 b, Color color)
{
    _drawLineVert(a,color);
    _drawLineVert(b,color);
}

void _drawTri(Vec3 a, Vec3 b, Vec3 c, Color color)
{
    Vec3 n = triComputeNormal(a,b,c);
    _drawTriVert(a,n,color);
    _drawTriVert(b,n,color);
    _drawTriVert(c,n,color);
}

void _drawWireTri(Vec3 a, Vec3 b, Vec3 c, Color color)
{
    Vec3 n = triComputeNormal(a,b,c);
    _drawWireTriVert(a,n,color);
    _drawWireTriVert(b,n,color);
    _drawWireTriVert(c,n,color);
}

void drawIndexedTris(const Vec3 points[], const uint32_t indices[], size_t count, Color color)
{
    for (size_t i = 0; i < count; i += 3)
    {
        Vec3 a = points[indices[i]];
        Vec3 b = points[indices[i+1]];
        Vec3 c = points[indices[i+2]];

        Vec3 n = triComputeNormal(a,b,c);
        _drawTriVert(a,n,color);
        _drawTriVert(b,n,color);
        _drawTriVert(c,n,color);
    }
}

void drawBox(Vec3 a, Vec3 b, Color color)
{
    Vec3 p0 = {a.x,a.y,a.z};
    Vec3 p1 = {b.x,a.y,a.z};
    Vec3 p2 = {b.x,b.y,a.z};
    Vec3 p3 = {a.x,b.y,a.z};
    Vec3 p4 = {a.x,a.y,b.z};
    Vec3 p5 = {b.x,a.y,b.z};
    Vec3 p6 = {b.x,b.y,b.z};
    Vec3 p7 = {a.x,b.y,b.z};
    Vec3 n0 = { 0, 0,-1};
    Vec3 n1 = { 0,-1, 0};
    Vec3 n2 = {-1, 0, 0};
    Vec3 n3 = { 0, 1, 0};
    Vec3 n4 = { 1, 0, 0};
    Vec3 n5 = { 0, 0, 1};
    _drawQuad(p0,p1,p2,p3,n0,color);
    _drawQuad(p1,p0,p4,p5,n1,color);
    _drawQuad(p0,p3,p7,p4,n2,color);
    _drawQuad(p6,p7,p3,p2,n3,color);
    _drawQuad(p2,p1,p5,p6,n4,color);
    _drawQuad(p4,p7,p6,p5,n5,color);
}

void drawWireBox(Vec3 a, Vec3 b, Color color)
{
    Vec3 p0 = {a.x,a.y,a.z};
    Vec3 p1 = {b.x,a.y,a.z};
    Vec3 p2 = {b.x,b.y,a.z};
    Vec3 p3 = {a.x,b.y,a.z};
    Vec3 p4 = {a.x,a.y,b.z};
    Vec3 p5 = {b.x,a.y,b.z};
    Vec3 p6 = {b.x,b.y,b.z};
    Vec3 p7 = {a.x,b.y,b.z};
    _drawLine(p0,p1,color);
    _drawLine(p1,p2,color);
    _drawLine(p2,p3,color);
    _drawLine(p3,p0,color);
    _drawLine(p4,p5,color);
    _drawLine(p5,p6,color);
    _drawLine(p6,p7,color);
    _drawLine(p7,p4,color);
    _drawLine(p0,p4,color);
    _drawLine(p1,p5,color);
    _drawLine(p2,p6,color);
    _drawLine(p3,p7,color);
}
 void drawAabb(Aabb aabb, Color color)
 {
     drawWireBox(aabb.getMin(), aabb.getMax(), color);
 }

void drawLine(Vec3 a, Vec3 b, Color color)
{
    _drawLine(a,b,color);
}

void drawWireSphere(Vec3 p, float radius, Color color)
{
    const size_t NLon = 5;
    const size_t NLat = 15;
    for (size_t i = 0; i < NLon; i++)
    {
        float ti = 2*Math::PI/NLon*i;
        Vec3 d = {cosf(ti), sinf(ti), 0};
        for (size_t j = 0; j <= NLat; j++)
        {
            float tj0 = 2*Math::PI/NLat*j;
            float tj1 = 2*Math::PI/NLat*(j+1);

            float c0 = cosf(tj0);
            float s0 = cosf(tj0);
            Vec3 p0 = {p.x+d.x*c0, p.y+d.y*c0, p.z+d.z*c0 + s0*radius};

            float c1 = cosf(tj1);
            float s1 = cosf(tj1);
            Vec3 p1 = {p.x+d.x*c1, p.y+d.y*c1, p.z+d.z*c1 + s1*radius};

            _drawLine(p0,p1,color);
        }
    }
}

void drawSphere(Vec3 origin, float radius, Color color, size_t numLon, size_t numLat)
{
    enum {
        MaxLon = 24,
        MaxLat = 24
    };

    ASSERT(numLon < MaxLon && numLat < MaxLat);
    static DebugVertex verts[(MaxLon+1)*(MaxLat+1)]; 
    size_t numVerts = 0;
    for(size_t i = 0; i <= numLat; ++i)
    {
        float lat = Math::PI / 2 - i * (Math::PI/numLat);
        float cLat = cosf(lat);
        float sLat = sinf(lat);
        for(size_t j = 0; j <= numLon; ++j)
        {
            float lon = j * (2*Math::PI/numLon);
            Vec3 d = {cLat*cosf(lon),cLat*sinf(lon),sLat};
            Vec3 pos = {origin.x+radius*d.x, origin.y+radius*d.y, origin.z+radius*d.z};
            DebugVertex vert = {pos, d, {0}, color};
            verts[numVerts++] = vert;
        }
    }

    for(size_t i = 0; i < numLat; ++i)
    {
        size_t k1 = i * (numLon + 1);
        size_t k2 = k1 + numLon + 1;
        for(size_t j = 0; j < numLon; ++j, ++k1, ++k2)
        {
            if(i != 0)
            {
                _drawTriVertV(verts[k1]);
                _drawTriVertV(verts[k2]);
                _drawTriVertV(verts[k1+1]);
            }
            if(i != (numLat-1))
            {
                _drawTriVertV(verts[k1+1]);
                _drawTriVertV(verts[k2]);
                _drawTriVertV(verts[k2+1]);
            }
        }
    }
}


void drawWireCylinder(Vec3 a, Vec3 b, float eX, float eY, Color color)
{
    const size_t NumSeg = 28;
    Vec3 delta = b - a;
    Mat3 basis = Mat3::basis(delta).scale(eY, eY, 1.0f);
    Transform xf = {basis, a};
	for (size_t i = 0; i < NumSeg; i++)
	{
        float t0 = 2*Math::PI/NumSeg*i;
        float t1 = 2*Math::PI/NumSeg*(i+1);
        Vec3 p0 = {cosf(t0), sinf(t0), 0};
        Vec3 p1 = {cosf(t1), sinf(t1), 0};
        Vec3 a0 = xf.mul(p0);
        Vec3 a1 = xf.mul(p1);
        Vec3 b0 = a0 + delta;
        Vec3 b1 = a1 + delta;
        _drawLine(a0, a1, color);
        _drawLine(b0, b1, color);
        if ((i % 4) == 0)
        {
            _drawLine(a0,b0,color);
        }
	}
}

void drawArrow(Vec3 origin, Vec3 arrow, Color color, float scale)
{
    const float eXY = 2*0.007f*scale;
    const float eZ = 2*0.04f*scale;

    Mat3 basis = Mat3::basis(arrow).scale(eXY, eXY, eZ);

    Vec3 q = origin + arrow;
    Vec3 p = q - basis.c2;

    Transform t = {basis, p};
    Vec3 a = t.mul(Vec3{ 1,  1, 0});
    Vec3 b = t.mul(Vec3{ 1, -1, 0});
    Vec3 c = t.mul(Vec3{-1, -1, 0});
    Vec3 d = t.mul(Vec3{-1,  1, 0});

    Vec3 points[] = { a, b, c, d, q };
    uint32_t indices[] = { 0,4,1, 1,4,2, 2,4,3, 3,4,0, 0,1,2, 0,2,3 };
    drawIndexedTris(points, indices, 18, color);
    drawLine(origin, p, color);
}

void drawArrowTo(Vec3 a, Vec3 b, Color color, float scale)
{
    drawArrow(a, b - a, color, scale);
}

void drawArcBetween(Vec3 origin, Vec3 a, Vec3 b, float radius, Color color, bool directed)
{
    a = a.normalized();
    b = b.normalized();
    const size_t NumSeg = 32;
    float dist = a.dist(b);
    size_t subdivisions = (size_t)ceilf(dist/0.1f);
    //ASSERT(vecDist(a,b) < 1.999f);

    for (size_t i = 0; i < NumSeg; i++)
    {
        float t0 = i/(float)NumSeg;
        float t1 = (i+1)/(float)NumSeg;
        Vec3 p0 = origin + Vec3::lerp(t0,a,b).normalized()*radius;
        Vec3 p1 = origin + Vec3::lerp(t1,a,b).normalized()*radius;
        if (directed)
        {
            if (i == NumSeg-2 || i == 3 || ((i % 8) == 0 && (i+8<NumSeg) && (i > 8)))
            {
                drawArrow(p0, p1 - p0, color, 1.0f);
            }
            else
            {
                drawLine(p0, p1, color);
            }
        }
        else
        {
            drawLine(p0, p1, color);
        }
    }
}

void drawWireCone(Vec3 a, Vec3 b, float eX, float eY, Color color)
{
    const size_t NumSeg = 20;
    Mat3 basis = Mat3::basis(b - a).scale(eX, eY, 1.0f);
    Transform xf = {basis,a};
	for (size_t i = 0; i < NumSeg; i++)
	{
        float t0 = 2*Math::PI/NumSeg*i;
        float t1 = 2*Math::PI/NumSeg*(i+1);
        Vec3 p0 = {cosf(t0), sinf(t0), 0};
        Vec3 p1 = {cosf(t1), sinf(t1), 0};
        Vec3 a0 = xf.mul(p0);
        Vec3 a1 = xf.mul(p1);
        _drawLine(a0, a1, color);
        _drawLine(a0, b, color);
	}
}

void drawCone(Vec3 a, Vec3 b, float eX, float eY, Color color)
{
    const size_t NumSeg = 20;
    Mat3 basis = Mat3::basis(b-a).scale(eX, eY, 1.0f);
    Vec3 base = a + basis.c0*eX;
    Transform xf = {basis, a};
	for (size_t i = 0; i < NumSeg+1; i++)
	{
        float t0 = 2*Math::PI/NumSeg*i;
        float t1 = 2*Math::PI/NumSeg*(i+1);
        Vec3 p0 = {cosf(t0), sinf(t0), 0};
        Vec3 p1 = {cosf(t1), sinf(t1), 0};
        Vec3 a0 = xf.mul(p0);
        Vec3 a1 = xf.mul(p1);
        _drawTri(b, a0, a1, color);
        if (i > 0)
        {
            _drawTri(a0, base, a1, color);
        }
	}
}

void drawWireCapsule(Vec3 a, Vec3 b, float radius, Color color)
{

}

void drawCapsule(Vec3 a, Vec3 b, float radius, Color color)
{
}

void drawCoordinateSystem(Vec3 origin, float scale)
{
}

void drawPointEx(Vec3 origin, Color color, float scale)
{
#if 0
    const float e = 0.1f*scale;

    Vec3 c0 = { e,  e, 0};
    Vec3 c1 = { e, -e, 0};
    Vec3 c2 = {-e, -e, 0};
    Vec3 c3 = {-e,  e, 0};

    Vec3 w0 = Transform(origin, _drawData.cameraMat, c0); 
    Vec3 w1 = Transform(origin, _drawData.cameraMat, c1); 
    Vec3 w2 = Transform(origin, _drawData.cameraMat, c2); 
    Vec3 w3 = Transform(origin, _drawData.cameraMat, c3); 

    _drawQuad(w0, w1, w2, w3, color);
#else
    Vec3 extent = {0.02f*scale,0.02f*scale,0.02f*scale};
    drawBox(origin - extent, origin + extent, color);
#endif
}

void drawPoint(Vec3 origin, Color color)
{
    drawPointEx(origin, color,1.0f);
}

void drawPlane(Vec3 origin, Vec3 normal, Color color, float scale)
{
    Mat3 b = Mat3::basis(normal);
    Vec3 v0 = origin - b.c0*scale - b.c1*scale;
    Vec3 v1 = origin + b.c0*scale - b.c1*scale;
    Vec3 v2 = origin + b.c0*scale + b.c1*scale;
    Vec3 v3 = origin - b.c0*scale + b.c1*scale;
    _drawLine(v0, v1, color);
    _drawLine(v1, v2, color);
    _drawLine(v2, v3, color);
    _drawLine(v3, v0, color);

}

void drawDisk(Vec3 origin, Vec3 xAxis, Vec3 yAxis, float eX, float eY)
{
}

void drawCircle(Vec3 origin, Vec3 normal, float extent)
{
}

void drawOrientedWireBox(Vec3 origin, Vec3 xAxis, Vec3 yAxis, Vec3 zAxis, Vec3 extent)
{
    drawBox({-0.5f,-0.5f,-0.5f}, {0.5f,0.5f,0.5f},COLOR_BLUE);
}

void drawDemo(void)
{
    drawBox({-0.5f,-0.5f,-0.5f}, {0.5f,0.5f,0.5f},COLOR_BLUE);
    drawArrow({0,1,0}, {1,0,0},COLOR_GREEN, 1.0f);
    drawArcBetween({0.0f,0,0}, {-1,0,0}, {0,1,0}, 1.0f, COLOR_YELLOW,true);
    drawSphere({0,-1,0},0.5f,COLOR_LIGHT_BLUE,8,8);
    drawWireCylinder({2,0,0}, {1,1,0}, 0.5f,0.25f,COLOR_BLUE);
    drawWireCone({-1,-1.0f,0}, {-1.0f,0,0}, 0.4f, 0.3f, COLOR_GREEN);
    drawCone({-1,-1.0f,0}, {-1.0f,-2,0}, 0.4f, 0.3f,COLOR_LIGHT_BLUE);
}




inline Vec3 screenToWorldDir(Mat3 rot, float fovdeg, float znear, Vec2 screenPos, Vec2 screenSize)
{
    ASSERT(screenSize.x && screenSize.y);
    float sx = -1 + 2*screenPos.x/screenSize.x;
    float sy =  1 - 2*screenPos.y/screenSize.y; 

    float halfH = tanf(fovdeg*Math::DEG2RAD*0.5f)*znear;
    float halfW = halfH*screenSize.x/screenSize.y;

    Vec3 cameraSpacePoint = {halfW*sx, halfH*sy, znear};
    Vec3 worldSpacePoint = rot*cameraSpacePoint;
    Vec3 dir = worldSpacePoint.normalized();
    return dir;
}


static Mat3 g_camRot;
static Vec3 g_camPos;

const float CAMERA_FOV_DEG = 70.0f;
const float CAMERA_ZNEAR = 0.1f;
const float CAMERA_ZFAR = 1000;

void getPickRay(Vec3 &origin, Vec3 &dir)
{
    Vec2 screenPos = ImGui::GetMousePos();
    Vec2 screenSize = ImGui::GetMainViewport()->Size;
    if (screenPos.x == -FLT_MAX)
    {
        origin = {};
        dir = {0,0,1};
        return;
    }
    dir = screenToWorldDir(g_camRot, CAMERA_FOV_DEG, CAMERA_ZNEAR, screenPos, screenSize);
    origin = g_camPos;
}

void setCamera(Mat3 rot, Vec3 pos)
{
    g_camRot = rot;
    g_camPos = pos;
    Vec2 screenSize = ImGui::GetMainViewport()->Size;
    g_debugDrawData.lightDir = -g_camRot.c2;
    Mat4 proj = Mat4::perspectiveLH(CAMERA_FOV_DEG, screenSize.y/screenSize.x, CAMERA_ZNEAR, CAMERA_ZFAR);
    Mat4 view = Mat4::setInvTR(g_camPos, g_camRot);
    g_debugDrawData.worldViewProj = proj.mul(view); 
}



#define NOMINMAX
#define WIN32_LEAN_AND_MEAN
#include <d3d11.h>
#include <d3dcompiler.h>
#include <dxgi.h>

#pragma comment(lib, "d3d11.lib")
#pragma comment(lib, "d2d1.lib")
#pragma comment(lib, "dwrite.lib")
#pragma comment(lib, "d3dcompiler.lib")

#include "imgui\imgui.h"
#include "imgui\imgui_impl_dx11.h"
#include "imgui\imgui_impl_win32.h"

#define SAFE_RELEASE(x) do { if (x) { (x)->Release(); (x) = NULL; } } while (0)
#define CHECK_HR(x) ASSERT((x) == S_OK)

static ID3D11Device             *g_device;
static ID3D11DeviceContext      *g_deviceContext;

ID3D11Buffer *updateDynamicBuffer(ID3D11Buffer *buf, const void *data, size_t size, size_t capacity, D3D11_BIND_FLAG usage)
{
    ASSERT(capacity > 0);

    if (buf)
    {
        D3D11_BUFFER_DESC desc;
        buf->GetDesc(&desc);
        if ((int)desc.ByteWidth < capacity)
        {
            buf->Release();
            buf = NULL;
        }
    }

    if (!buf)
    {
        D3D11_BUFFER_DESC desc ={(UINT)capacity, D3D11_USAGE_DYNAMIC, (UINT)usage, D3D11_CPU_ACCESS_WRITE};
        CHECK_HR(g_device->CreateBuffer(&desc, NULL, &buf));
    }

    D3D11_MAPPED_SUBRESOURCE mapped;
    CHECK_HR(g_deviceContext->Map(buf,0, D3D11_MAP_WRITE_DISCARD,0,&mapped));
    memcpy(mapped.pData, data, size);
    g_deviceContext->Unmap(buf, 0);
    return buf;
}


static ID3D11Buffer *updateConstantBuffer(ID3D11Buffer *buf, const void *data, size_t size)
{
    size_t capacity = ((size+15)/16)*16;
    return updateDynamicBuffer(buf, data, size, capacity, D3D11_BIND_CONSTANT_BUFFER);
}

const size_t MAX_CBUFFERS = 8;

struct VertexShader
{
    ID3D11VertexShader *vertexShader;
    ID3D11InputLayout *inputLayout;
    ID3D11Buffer *cbuffers[MAX_CBUFFERS];

    template <typename T>
    void setVar(size_t cb, const T &v)
    {
        ASSERT(cb >= 0 && cb < MAX_CBUFFERS);
        cbuffers[cb] = updateConstantBuffer(cbuffers[cb], &v, sizeof(v));
    }

    void bind()
    {
        g_deviceContext->IASetInputLayout(inputLayout);
        g_deviceContext->VSSetShader(vertexShader, NULL, 0);
        g_deviceContext->VSSetConstantBuffers(0, MAX_CBUFFERS, cbuffers);
    }

    void compile(const char *source, size_t sourceLen, const char *entry, D3D11_INPUT_ELEMENT_DESC elems[], size_t numElems)
    {
        ID3D10Blob *code;
        CHECK_HR(D3DCompile(source, sourceLen, NULL, NULL, NULL, entry, "vs_5_0", 0, 0, &code, NULL));
        CHECK_HR(g_device->CreateVertexShader(code->GetBufferPointer(), code->GetBufferSize(), NULL, &vertexShader));
        CHECK_HR(g_device->CreateInputLayout(elems, (UINT)numElems, code->GetBufferPointer(), code->GetBufferSize(), &inputLayout));
        code->Release();
    }
};

struct PixelShader
{
    ID3D11PixelShader *pixelShader;
    ID3D11Buffer *cbuffers[MAX_CBUFFERS];

    template <typename T>
    void setVar(size_t cb, const T &v)
    {
        ASSERT(cb >= 0 && cb < MAX_CBUFFERS);
        cbuffers[cb] = updateConstantBuffer(cbuffers[cb], &v, sizeof(v));
    }

    void compile(const char *source, size_t sourceLen, const char *entry)
    {
        ID3D10Blob *code;
        CHECK_HR(D3DCompile(source, sourceLen, NULL, NULL, NULL, entry, "ps_5_0", 0, 0, &code, NULL));
        CHECK_HR(g_device->CreatePixelShader(code->GetBufferPointer(), code->GetBufferSize(), NULL, &pixelShader));
        code->Release();
    }

    void bind()
    {
        g_deviceContext->PSSetShader(pixelShader, NULL, 0);
        g_deviceContext->PSSetConstantBuffers(0, MAX_CBUFFERS, cbuffers);
    }
};




const size_t DEFAULT_SCREEN_WIDTH = 1600;
const size_t DEFAULT_SCREEN_HEIGHT = 900;

static HWND                     g_window;
static int                      g_clientSizeX;
static int                      g_clientSizeY;

static ID3D11Texture2D          *g_backbufTex;
static ID3D11RenderTargetView   *g_backbufRtv;
static ID3D11Texture2D          *g_depthbufTex;
static ID3D11DepthStencilView   *g_depthbufDsv;
static IDXGISwapChain           *g_swapchain;
static ID3D11DepthStencilState  *g_depthState;
static ID3D11BlendState         *g_premulBlend;
static ID3D11RasterizerState    *g_triRaster;
static ID3D11RasterizerState    *g_wireframeRaster;
static ID3D11RasterizerState    *g_flatRaster;
static ID3D11SamplerState       *g_linearSampler;
static ID3D11SamplerState       *g_nearestSampler;
static VertexShader              g_commonVs;
static PixelShader               g_lambertPs;
static PixelShader               g_flatPs;
static PixelShader               g_gridPs;
static ID3D11Buffer             *g_tmpVbo;

struct SVarCommon
{
    Mat4 mvp;
    Mat4 model;
};

struct SVarGrid
{
    Vec3 pos;
};

struct SVarFlat
{
    float dimFactor;
};

struct SVarLambert
{
    Vec3 lightDir;
};

extern IMGUI_IMPL_API LRESULT ImGui_ImplWin32_WndProcHandler(HWND hWnd, UINT msg, WPARAM wParam, LPARAM lParam);
static LRESULT CALLBACK windowProc(HWND hwnd, UINT message, WPARAM wParam, LPARAM lParam)
{
    if (ImGui_ImplWin32_WndProcHandler(hwnd, message, wParam, lParam))
        return true;

    if (message == WM_CLOSE || message == WM_DESTROY)
    {
        ExitProcess(0);
    }

    return DefWindowProcA(hwnd, message, wParam, lParam);
}


void renderDebugDrawData()
{
    DebugDrawData &data = g_debugDrawData;
    g_deviceContext->OMSetBlendState(g_premulBlend, NULL, 0xffffffff);

    g_gridPs.setVar(0, SVarGrid{data.eyePos});
    g_lambertPs.setVar(0, data.lightDir);
    g_commonVs.setVar(0, SVarCommon{data.worldViewProj, Mat4::identity()});
    g_flatPs.setVar(0, SVarFlat{1.0f});

    const UINT vertexStride = sizeof(DebugVertex);
    const UINT vertexOffset = 0;

    g_deviceContext->OMSetDepthStencilState(g_depthState, 0);

    if (data.lines.size())
    {
        g_tmpVbo = updateDynamicBuffer(g_tmpVbo, &data.lines[0], sizeInBytes(data.lines), capacityInBytes(data.lines), D3D11_BIND_VERTEX_BUFFER);
        g_deviceContext->IASetVertexBuffers(0, 1, &g_tmpVbo, &vertexStride, &vertexOffset);
        g_deviceContext->RSSetState(g_flatRaster);
        g_deviceContext->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_LINELIST);
        g_deviceContext->PSSetSamplers(0, 1, &g_linearSampler);
        g_commonVs.bind();
        g_flatPs.bind();
        g_deviceContext->Draw((UINT)data.lines.size(), 0);
        data.lines.clear();

    }

    if (data.tris.size())
    {
        g_tmpVbo = updateDynamicBuffer(g_tmpVbo, &data.tris[0], sizeInBytes(data.tris), capacityInBytes(data.tris), D3D11_BIND_VERTEX_BUFFER);
        g_deviceContext->IASetVertexBuffers(0, 1, &g_tmpVbo, &vertexStride, &vertexOffset);
        g_deviceContext->RSSetState(g_triRaster);
        g_deviceContext->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
        g_commonVs.bind();
        g_lambertPs.bind();
        g_deviceContext->Draw((UINT)data.tris.size(), 0);
        data.tris.clear();
    }

    g_flatPs.setVar(0, SVarFlat{0.0f});

    if (data.wireTris.size())
    {
        g_tmpVbo = updateDynamicBuffer(g_tmpVbo, &data.wireTris[0], sizeInBytes(data.wireTris), capacityInBytes(data.wireTris), D3D11_BIND_VERTEX_BUFFER);
        g_deviceContext->IASetVertexBuffers(0, 1, &g_tmpVbo, &vertexStride, &vertexOffset);
        g_commonVs.bind();
        g_lambertPs.bind();
        g_deviceContext->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST);

        g_deviceContext->RSSetState(g_triRaster);
        g_deviceContext->Draw((UINT)data.wireTris.size(), 0);

        g_flatPs.bind();
        g_deviceContext->RSSetState(g_wireframeRaster);
        g_deviceContext->Draw((UINT)data.wireTris.size(), 0);

        data.wireTris.clear();
    }

    if (data.meshes.size())
    {
        g_commonVs.bind();
        g_deviceContext->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
        g_deviceContext->RSSetState(g_triRaster);
        for(size_t i = 0; i < data.meshes.size(); i++)
        {
            DrawMeshCommand &cmd = data.meshes[i];
            auto it = g_debugMeshes.find(cmd.meshId);
            DebugMesh &mesh = it->second;
            g_deviceContext->IASetVertexBuffers(0, 1, (ID3D11Buffer**)&mesh.vbo, &vertexStride, &vertexOffset);
            g_commonVs.setVar(0, SVarCommon{data.worldViewProj.mul(cmd.transform), cmd.transform});
            g_lambertPs.bind();
            g_deviceContext->RSSetState(g_triRaster);
            g_deviceContext->Draw((UINT)mesh.numVerts, 0);

            g_flatPs.bind();
            g_deviceContext->RSSetState(g_wireframeRaster);
            g_deviceContext->Draw((UINT)mesh.numVerts, 0);

        }
        data.meshes.clear();
    }


    {
        ID3D11Buffer *dummy = NULL;
        g_deviceContext->IASetVertexBuffers(0, 1, (ID3D11Buffer**)&dummy, &vertexStride, &vertexOffset);
    }
}

void createDebugMesh(DebugMeshId id, const Vec3 vertices[], size_t numVertices, const uint32_t indices[], size_t numIndices)
{
    ASSERT(g_debugMeshes.find(id) == g_debugMeshes.end());

    std::vector<DebugVertex> debugVerts;
    debugVerts.resize(numIndices*3);
    for (size_t i = 0; i < numIndices; i += 3)
    {
        Vec3 v0 = vertices[indices[i+0]];
        Vec3 v1 = vertices[indices[i+1]];
        Vec3 v2 = vertices[indices[i+2]];
        Vec3 n = triComputeNormal(v0, v1, v2);
        debugVerts[i+0] = {v0, n, {}, COLOR_WHITE};
        debugVerts[i+1] = {v1, n, {}, COLOR_WHITE};
        debugVerts[i+2] = {v2, n, {}, COLOR_WHITE};
    }

    ID3D11Buffer *buf;
    size_t size = sizeInBytes(debugVerts);
    buf = updateDynamicBuffer(NULL, &debugVerts[0], size, size, D3D11_BIND_VERTEX_BUFFER);

    DebugMesh mesh;
    mesh.vbo = buf;
    mesh.numVerts = (int)debugVerts.size();
    mesh.id = id;
    g_debugMeshes.insert({id, mesh});
}

void destroyDebugMesh(DebugMeshId id)
{
    auto it = g_debugMeshes.find(id);
    ID3D11Buffer *buf = (ID3D11Buffer*)it->second.vbo;
    buf->Release();
    g_debugMeshes.erase(id);
}

extern const char *g_shaderSource;

void demoTick();
void demoInit();

int WINAPI WinMain(HINSTANCE hInstance, HINSTANCE hPrevInstance, LPSTR lpCmdLine, int nCmdShow)
{
    WNDCLASSEXA wc = {sizeof(wc), CS_HREDRAW|CS_VREDRAW, windowProc, 0, 0, GetModuleHandleA(NULL), NULL, NULL, NULL, NULL, "SatBlogD3D11"};
    RegisterClassExA(&wc);
    RECT rect = {0,0,DEFAULT_SCREEN_WIDTH,DEFAULT_SCREEN_HEIGHT}; 
    AdjustWindowRect(&rect,WS_OVERLAPPEDWINDOW,FALSE);
    g_window = CreateWindowExA(0, wc.lpszClassName, "SatBlogD3D11", WS_OVERLAPPEDWINDOW | WS_VISIBLE, CW_USEDEFAULT, CW_USEDEFAULT, rect.right-rect.left, rect.bottom-rect.top, NULL, NULL, wc.hInstance, NULL);

    // TODO: handle device lost
    {
        D3D_FEATURE_LEVEL wantFeature = D3D_FEATURE_LEVEL_11_0;
        D3D_FEATURE_LEVEL gotFeature;
        UINT flags = D3D11_CREATE_DEVICE_DEBUG;
        CHECK_HR(D3D11CreateDevice(NULL, D3D_DRIVER_TYPE_HARDWARE, NULL, flags | D3D11_CREATE_DEVICE_BGRA_SUPPORT, &wantFeature, 1, D3D11_SDK_VERSION, &g_device, &gotFeature, &g_deviceContext));
    }


    {
        {
            D3D11_RASTERIZER_DESC desc ={D3D11_FILL_SOLID, D3D11_CULL_BACK, FALSE, 100, 0, 1.5};
            desc.MultisampleEnable = true;
            CHECK_HR(g_device->CreateRasterizerState(&desc, &g_triRaster));
        }
        {
            D3D11_RASTERIZER_DESC desc ={D3D11_FILL_WIREFRAME, D3D11_CULL_BACK};
            desc.MultisampleEnable = true;
            CHECK_HR(g_device->CreateRasterizerState(&desc, &g_wireframeRaster));
        }
        {
            D3D11_RASTERIZER_DESC desc ={D3D11_FILL_SOLID, D3D11_CULL_NONE};
            desc.MultisampleEnable = true;
            CHECK_HR(g_device->CreateRasterizerState(&desc, &g_flatRaster));
        }
        {
            D3D11_DEPTH_STENCIL_DESC desc ={TRUE, D3D11_DEPTH_WRITE_MASK_ALL, D3D11_COMPARISON_LESS};
            CHECK_HR(g_device->CreateDepthStencilState(&desc, &g_depthState));
        }
        {
            D3D11_BLEND_DESC desc ={FALSE, FALSE, {{TRUE,  D3D11_BLEND_ONE,  D3D11_BLEND_INV_SRC_ALPHA, D3D11_BLEND_OP_ADD,  D3D11_BLEND_INV_DEST_ALPHA,  D3D11_BLEND_ONE, D3D11_BLEND_OP_ADD, D3D11_COLOR_WRITE_ENABLE_ALL, }}};
            CHECK_HR(g_device->CreateBlendState(&desc, &g_premulBlend));
        }
        {
            D3D11_SAMPLER_DESC desc ={D3D11_FILTER_MIN_MAG_MIP_LINEAR, D3D11_TEXTURE_ADDRESS_CLAMP, D3D11_TEXTURE_ADDRESS_CLAMP, D3D11_TEXTURE_ADDRESS_CLAMP, 0, 0, D3D11_COMPARISON_NEVER, {1, 1, 1, 1}};
            CHECK_HR(g_device->CreateSamplerState(&desc, &g_linearSampler));
        }
        {
            D3D11_SAMPLER_DESC desc ={D3D11_FILTER_MIN_MAG_MIP_POINT, D3D11_TEXTURE_ADDRESS_BORDER, D3D11_TEXTURE_ADDRESS_BORDER, D3D11_TEXTURE_ADDRESS_BORDER, 0, 0, D3D11_COMPARISON_NEVER, {1, 1, 1, 1}};
            CHECK_HR(g_device->CreateSamplerState(&desc, &g_nearestSampler));
        }

        D3D11_INPUT_ELEMENT_DESC vertexPntcLayout[] =
        {
            {"POS",     0, DXGI_FORMAT_R32G32B32_FLOAT, 0, 0,                            D3D11_INPUT_PER_VERTEX_DATA, 0},
            {"NORMAL",  0, DXGI_FORMAT_R32G32B32_FLOAT, 0, D3D11_APPEND_ALIGNED_ELEMENT, D3D11_INPUT_PER_VERTEX_DATA, 0},
            {"TEX",     0, DXGI_FORMAT_R32G32_FLOAT,    0, D3D11_APPEND_ALIGNED_ELEMENT, D3D11_INPUT_PER_VERTEX_DATA, 0},
            {"COL",     0, DXGI_FORMAT_R8G8B8A8_UNORM,  0, D3D11_APPEND_ALIGNED_ELEMENT, D3D11_INPUT_PER_VERTEX_DATA, 0},
        };


        size_t sourceSize = strlen(g_shaderSource);
        g_commonVs.compile(g_shaderSource, sourceSize, "commonVs", vertexPntcLayout, _countof(vertexPntcLayout));
        g_lambertPs.compile(g_shaderSource, sourceSize, "lambertPs");
        g_gridPs.compile(g_shaderSource, sourceSize, "gridPs");
        g_flatPs.compile(g_shaderSource, sourceSize, "flatPs");
    }

    ImGui::CreateContext();
    ImGuiIO imguiIo = ImGui::GetIO();
    ImGui::StyleColorsDark();
    ImGui_ImplWin32_Init(g_window);
    ImGui_ImplDX11_Init(g_device, g_deviceContext);

    demoInit();

    while(true)
    {
        MSG msg;
        while(PeekMessageA(&msg, NULL, 0, 0, PM_REMOVE))
        {
            TranslateMessage(&msg);
            DispatchMessageA(&msg);
        }

        ImGui_ImplDX11_NewFrame();
        ImGui_ImplWin32_NewFrame();
        ImGui::NewFrame();


        {
            ASSERT(g_device);
            ASSERT(g_deviceContext);

            RECT newRect;
            GetClientRect(g_window, &newRect);
            int newSizeX = newRect.right-newRect.left;
            int newSizeY = newRect.bottom-newRect.top;


            if((newSizeX && newSizeY) && (g_clientSizeX != newSizeX || g_clientSizeY != newSizeY))
            {
                if(!g_swapchain)
                {
                    DXGI_SWAP_CHAIN_DESC desc ={{(UINT)newSizeX, (UINT)newSizeY, {60, 1}, DXGI_FORMAT_R8G8B8A8_UNORM}, {8, 0}, DXGI_USAGE_RENDER_TARGET_OUTPUT, 2, g_window, TRUE, DXGI_SWAP_EFFECT_DISCARD};
                    IDXGIDevice* dxgiDevice = 0;
                    IDXGIAdapter* adapter = 0;
                    IDXGIFactory* factory = 0;
                    CHECK_HR(g_device->QueryInterface(IID_PPV_ARGS(&dxgiDevice)));
                    CHECK_HR(dxgiDevice->GetParent(IID_PPV_ARGS(&adapter)));
                    CHECK_HR(adapter->GetParent(IID_PPV_ARGS(&factory)));
                    CHECK_HR(factory->CreateSwapChain(g_device, &desc, &g_swapchain));
                    CHECK_HR(factory->MakeWindowAssociation(g_window, DXGI_MWA_NO_ALT_ENTER));
                    SAFE_RELEASE(dxgiDevice);
                    SAFE_RELEASE(adapter);
                    SAFE_RELEASE(factory);
                }
                else
                {
                    SAFE_RELEASE(g_depthbufDsv);
                    SAFE_RELEASE(g_depthbufTex);
                    SAFE_RELEASE(g_backbufRtv);
                    SAFE_RELEASE(g_backbufTex);
                    // TODO: handle device lost
                    CHECK_HR(g_swapchain->ResizeBuffers(2, newSizeX, newSizeY, DXGI_FORMAT_UNKNOWN, 0));
                }
                CHECK_HR(g_swapchain->GetBuffer(0, IID_PPV_ARGS(&g_backbufTex)));
                CHECK_HR(g_device->CreateRenderTargetView(g_backbufTex, NULL, &g_backbufRtv));
                D3D11_TEXTURE2D_DESC desc;
                g_backbufTex->GetDesc(&desc);
                desc.Format = DXGI_FORMAT_D24_UNORM_S8_UINT;
                desc.BindFlags = D3D11_BIND_DEPTH_STENCIL;
                CHECK_HR(g_device->CreateTexture2D(&desc, NULL, &g_depthbufTex));
                CHECK_HR(g_device->CreateDepthStencilView(g_depthbufTex, NULL, &g_depthbufDsv));

                g_clientSizeX = newSizeX;
                g_clientSizeY = newSizeY;
            }
        }

        demoTick();

        ImGui::Render();

        FLOAT bgColor[4] ={0.3, 0.3, 0.3, 0};
        D3D11_VIEWPORT viewport ={0.0f, 0.0f, (float)g_clientSizeX, (float)g_clientSizeY, 0.0f, 1.0f};
        g_deviceContext->ClearRenderTargetView(g_backbufRtv, bgColor);
        g_deviceContext->ClearDepthStencilView(g_depthbufDsv, D3D11_CLEAR_DEPTH, 1.0f, 0);
        g_deviceContext->RSSetViewports(1, &viewport);
        g_deviceContext->OMSetRenderTargets(1, &g_backbufRtv, g_depthbufDsv);

        renderDebugDrawData();
        ImGui_ImplDX11_RenderDrawData(ImGui::GetDrawData());
        // TODO: handle device lost
        CHECK_HR(g_swapchain->Present(1, 0));

    }
    return 0;
}


#if 0
typedef void *TextureId;
static TextureId createTexture(void *bytes, size_t width, size_t height, bool genMips)
{
    ID3D11ShaderResourceView *srv;
    ID3D11Texture2D *tex;
    if(genMips)
    {
        {
            D3D11_TEXTURE2D_DESC desc;
            desc.Width = (UINT)width;
            desc.Height = (UINT)height;
            desc.MipLevels = 0;
            desc.ArraySize = 1;
            desc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
            desc.SampleDesc ={1, 0};
            desc.Usage = D3D11_USAGE_DEFAULT;
            desc.BindFlags = D3D11_BIND_SHADER_RESOURCE | D3D11_BIND_RENDER_TARGET;
            desc.CPUAccessFlags = 0;
            desc.MiscFlags = D3D11_RESOURCE_MISC_GENERATE_MIPS;
            CHECK_HR(g_device->CreateTexture2D(&desc, NULL, &tex));
        }

        {
            D3D11_SHADER_RESOURCE_VIEW_DESC desc ={};
            desc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
            desc.ViewDimension =  D3D11_SRV_DIMENSION_TEXTURE2D;
            desc.Texture2D.MipLevels = -1;
            CHECK_HR(g_device->CreateShaderResourceView(tex, &desc, &srv));
        }

        if(bytes)
        {
            g_deviceContext->UpdateSubresource(tex, 0, NULL, bytes, width*4, 0);
            g_deviceContext->GenerateMips(srv);
        }
    }
    else
    {
        D3D11_TEXTURE2D_DESC desc ={(UINT)width, (UINT)height, 1, 1, DXGI_FORMAT_R8G8B8A8_UNORM, {1, 0}, D3D11_USAGE_DEFAULT, D3D11_BIND_SHADER_RESOURCE};
        desc.Width = (UINT)width;
        desc.Height = (UINT)height;
        desc.MipLevels = 1;
        desc.ArraySize = 1;
        desc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
        desc.SampleDesc ={1, 0};
        desc.Usage = D3D11_USAGE_DEFAULT;
        desc.BindFlags = D3D11_BIND_SHADER_RESOURCE | D3D11_BIND_RENDER_TARGET;
        desc.CPUAccessFlags = 0;
        desc.MiscFlags = 0;

        D3D11_SUBRESOURCE_DATA data ={bytes, (UINT)4*width, 0};
        CHECK_HR(g_device->CreateTexture2D(&desc, &data, &tex));
        CHECK_HR(g_device->CreateShaderResourceView(tex, NULL, &srv));
    }

    tex->Release();

    return (TextureId)srv;
}
static TextureId tryLoadTexture(const char* filename, bool genmips)
{
    int channels = 4;
    int width,height;
    void *pixels = stbi_load(filename,&width,&height,&channels,channels);
    if (!pixels) return false;
    TextureId res = createTexture(pixels,width,height,genmips);
    free(pixels);
    return res;
}
#endif

const char *g_shaderSource = R"HLSL(
cbuffer BasicVsUniforms : register(b0)
{
    float4x4 mvp;
    float4x4 model;
};

struct VertexIn
{
    float4 pos      : POS;
    float3 normal   : NORMAL;
    float2 texcoord : TEX;
    float4 color    : COL;
};

struct VertexOut
{
    float4 pos      : SV_POSITION;
    float4 wpos     : WPOS;
    float3 wnormal  : WNORMAL;
    float2 texcoord : TEX;
    float4 color    : COL;
};

VertexOut commonVs(VertexIn input)
{
    VertexOut output; 
    output.pos = mul(mvp,input.pos);
    output.wnormal = mul(model, float4(input.normal,0)).xyz;
    output.color = input.color;
    output.texcoord = input.texcoord;
    output.wpos = mul(model, input.pos);
    return output;
}

cbuffer LambertPsUniforms : register(b0)
{
    float3 lightDir;
}

float4 lambertPs(VertexOut input) : SV_TARGET
{
    float light1 = max(dot(normalize(input.wnormal),normalize(lightDir)),0);
    float4 intensity = 0.4f+0.6f*light1;
    float4 texcolor = float4(1,1,1,1);
    return float4(intensity.xyz*input.color.xyz*texcolor.xyz,1);
}

cbuffer FlatPsUniforms : register(b0)
{
    float flatDimFactor;
}

float4 flatPs(VertexOut input) : SV_TARGET
{
    return float4(input.color.xyz*flatDimFactor, input.color.w);
}

cbuffer GridPsUniforms : register(b0)
{
    float3 gridCamera;
    float pad;
}

float4 gridPs(VertexOut input) : SV_TARGET
{
    float dist = distance(input.wpos.xyz, gridCamera);
    float maxD = 80;
    float t = max(1.0f-dist/maxD, 0);
    return float4(input.color.xyz, t);
}

float4 dummy(float4 pos : POS) : SV_POSITION
{
    return float4(0,0,0,0);
}
)HLSL";
