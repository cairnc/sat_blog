#include "collision.h"

#include "tracy\Tracy.hpp"

enum Demo {
    DEMO_BROWSER,
    DEMO_SAT,
    DEMO_MESH
};

Demo g_demo;


struct Flycam
{
    Vec3 pos;
    float yaw;
    float pitch;
    Mat3 rot;
};

Flycam g_flycam;
float g_flySpeed = 2.0f;

struct Body
{
    Transform T;
    Mat3 iI;
    Vec3 v;
    Vec3 w;
    SatShape *shape;
    float iM;
    uint32_t id;
    uint32_t firstBodyPair;
    uint16_t numBodyPairs;
};

uint32_t g_nextBodyId = 1;

struct ArenaAllocator : Allocator
{
    size_t m_size;
    size_t m_capacity;
    uint8_t *m_memory;

    ArenaAllocator() : m_size(0), m_capacity(0), m_memory(0) {}

    void init(size_t capacity)
    {
        m_memory = (uint8_t*)malloc(capacity);
        m_capacity = capacity;
        m_size = 0;
    }

    virtual void *alloc(size_t size) 
    { 
        void *ptr = m_memory+m_size;
        m_size += size;
        ASSERT(m_size <= m_capacity);
        return ptr;
    };

    virtual void free(void *pointer) {};

    void reset()
    {
        m_size = 0;
    }


};

static Aabb g_bodyAabbs[SOLVER_MAX_BODIES];
static Body g_bodies[SOLVER_MAX_BODIES];
static uint16_t g_bodyPairs[SOLVER_MAX_MANIFOLDS];
static size_t g_numBodies;
static SolverManifold g_manifolds[SOLVER_MAX_MANIFOLDS];
static size_t g_numManifolds;
static SolverBody g_solverBodies[SOLVER_MAX_BODIES];
static ArenaAllocator g_shapeAllocator;
static Bvh g_bodyBvh;

static TriMeshShape testTriMesh;

void setBoxMass(Mat3 *invI, float *invM, float density, Vec3 extent)
{
    float mass = extent.x*extent.y*extent.z*density;
    *invM = 1.0f/mass;
    float c = 1.0f/12.0f * mass;
    float x2 = c*extent.x*extent.x;
    float y2 = c*extent.y*extent.y;
    float z2 = c*extent.z*extent.z;
    *invI =  {
        {1.0f/(y2+z2), 0, 0},
        {0, 1.0f/(x2+z2), 0},
        {0, 0, 1.0f/(x2+y2)}
    };
}

SatShape *makeBoxShape(Vec3 extent, float density)
{
    ASSERT(extent.x > 0 && extent.y > 0 && extent.z > 0);
    size_t numPoints = 8;
    Vec3 points[] ={
        {-extent.x, -extent.y, -extent.z},
        {-extent.x, -extent.y,  extent.z},
        {-extent.x,  extent.y, -extent.z},
        {-extent.x,  extent.y,  extent.z},
        { extent.x, -extent.y, -extent.z},
        { extent.x, -extent.y,  extent.z},
        { extent.x,  extent.y, -extent.z},
        { extent.x,  extent.y,  extent.z},
    };


    SatShape *shape = SatShape::createFromVerts(&g_shapeAllocator, points, numPoints);
    setBoxMass(&shape->invInertia, &shape->invMass, density, extent);

    return shape;
}

SatShape *makeRandomHull(Vec3 extent, float density, size_t numPoints)
{
    Vec3 points[256];
    ASSERT(numPoints < 256);
    for (size_t i = 0; i < numPoints; i++)
    {
        points[i] = Vec3::rand11();
        points[i].x *= extent.x;
        points[i].y *= extent.y;
        points[i].z *= extent.z;
    }
    SatShape *shape = SatShape::createFromVerts(&g_shapeAllocator, points, numPoints);
    setBoxMass(&shape->invInertia, &shape->invMass, density, extent);
    return shape;
}

void spawnRandomBodies(Vec3 origin, size_t numX, size_t numY, size_t numZ)
{
    for(size_t x = 0; x < numX; x++)
    {
        for(size_t y = 0; y < numY; y++)
        {
            for(size_t z = 0; z < numZ; z++)
            {
                SatShape *shape = NULL;

                Vec3 extent = Vec3{0.1f, 0.1f, 0.1f} + Vec3::rand01()*0.5f;

                size_t type = rand() % 2;
                if (1 || type == 0)
                {
                    shape = makeRandomHull(extent, 1.0f, 128);
                }
                else if (type == 1)
                {
                    shape = makeBoxShape(extent, 1.0f);
                }

                Body *body = g_bodies+g_numBodies++;
                body->shape = shape;
                body->v = {};
                body->w = {};
                body->id = g_nextBodyId;
                body->T = {Mat3::identity(), Vec3{ 4.0f*x, 4.0f*y, 4.0f*z } + origin};
                body->T.R = Mat3::rotY(Math::DEG2RAD*90)*Mat3::rotZ(Math::DEG2RAD*45.0f);
                body->iM = shape->invMass;
                body->iI = shape->invInertia;
                g_nextBodyId++;
            }
        }
    }
}

void demoInit()
{
    g_shapeAllocator.init(1024*1024*8);
    g_flycam.pos ={};
    g_flycam.rot = Mat3::identity();
}

struct GrabState
{
    bool grabbing;
    Vec3 bodyP;
    float dist;
    uint32_t bodyId;
    uint32_t hoverId;
};


GrabState grab;



#define TIMER(...)

void grabBodies()
{
    Vec3 pickOrigin, pickDir;
    getPickRay(pickOrigin, pickDir);


    Body *hoverBody = NULL;
    float hoverDist = INFINITY;
    grab.hoverId = -1;

    if (!ImGui::GetIO().WantCaptureMouse)
    {
        for(size_t i = 0; i < g_numBodies; i++)
        {
            Body *body = g_bodies+i;
            float tHit;
            Vec3 hit;
            if(raycastSatShape(pickOrigin, pickDir, body->shape, body->T, &tHit, &hit))
            {
                if(tHit < hoverDist)
                {
                    grab.hoverId = body->id;
                    hoverBody = body;
                    hoverDist = tHit;
                }
            }
        }

        Vec3 hoverP;
        if(hoverBody && !grab.grabbing)
        {
            hoverP = pickOrigin + pickDir*hoverDist;
            drawPointEx(hoverP, COLOR_RED, 0.5f);
            if(ImGui::IsKeyPressed('B'))
            {
                __debugbreak();
            }

            if(ImGui::IsKeyPressed(KEY_SPACE))
            {
                grab.grabbing = true;
                grab.bodyP = hoverBody->T.inverse().mul(hoverP);
                grab.bodyId = hoverBody->id;
                grab.dist = hoverDist;
            }
            drawPointEx(hoverP, COLOR_RED, 0.5f);
        }
    }


    if (!ImGui::IsKeyDown(KEY_SPACE))
    {
        grab.grabbing = false;
    }

    if (ImGui::IsKeyDown('J')) grab.dist -= 0.1f;
    if (ImGui::IsKeyDown('K')) grab.dist += 0.1f;
    grab.dist = Math::max(grab.dist, 0.1f);

    if (grab.grabbing)
    {

        Body *grabbedBody = NULL;

        for (size_t i = 0; i < g_numBodies; i++) 
        {
            if (g_bodies[i].id == grab.bodyId) 
            { 
                grabbedBody = g_bodies+i; 
                break; 
            }
        }

        if (grabbedBody)
        {
            float wheelDelta = ImGui::GetIO().MouseWheel;
            if(wheelDelta)
            {
                Mat3 rot = Quat::fromAxisAngle(g_flycam.rot.c2, wheelDelta*Math::PI*0.01f).toMat();
                grabbedBody->T.R = rot*grabbedBody->T.R;
            }

            Vec3 worldP = grabbedBody->T.mul(grab.bodyP);
            Vec3 targetP = g_flycam.pos + pickDir*grab.dist;
            drawLine(worldP, worldP, COLOR_RED);
            drawLine(worldP, targetP, COLOR_RED);
            drawPointEx(worldP, COLOR_RED, 0.5f);
            drawPointEx(targetP, COLOR_RED, 0.5f);

            const float deltaSlop = 0.01f;
            if(worldP.dist(targetP) > deltaSlop)
            {
                Vec3 d = worldP - targetP;
                float dist =d.length();
                d = d/dist;
                Vec3 Jva = d;
                Vec3 Jwa =(worldP - grabbedBody->T.p).cross(d);
                float bias =  dist < deltaSlop ? 0 : 0.1f*(dist-deltaSlop)/SOLVER_DT;

                float iM = grabbedBody->iM;
                Mat3 iI = grabbedBody->T.R*grabbedBody->iI*grabbedBody->T.R.transpose();

                float mC = 1.0f/(iM + Jwa.dot(iI*Jwa));
                float lambda = -mC*(Jva.dot(grabbedBody->v) + Jwa.dot(grabbedBody->w) + bias);
                lambda = Math::clamp(lambda,-10.0f,0.0f);

                grabbedBody->v += Jva*iM*lambda;
                grabbedBody->w += iI*Jwa*lambda;
                grabbedBody->v *= 0.95f;
                grabbedBody->w *= 0.6f;
            }
        }
    }
}

void demoTriMesh()
{
    ZoneScoped;

    static bool enableGravity = true;
    static bool useGraph = false;

    if (!testTriMesh.numTris)
    {
        ASSERT(loadTriMeshShape(&testTriMesh, "test_meshes/ascension.obj"));

        g_flycam.pos = { 10.091136f,6.8672866f,8.7661008f };
        g_flycam.yaw = -2.2599978f;
        g_flycam.pitch = 0.34799960f;
        g_flycam.rot = Mat3::rotY(g_flycam.yaw)*Mat3::rotX(g_flycam.pitch);
    }




    if (enableGravity)
    {
        for(size_t i = 0; i < g_numBodies; i++)
        {
            Body *body = g_bodies+i;
            float gravity = 20.8f;
            body->v.y -= gravity*SOLVER_DT;
        }
    }

    {
        ZoneScopedN("Draw Bodies");

        for (size_t i = 0; i < g_numBodies; i++)
        {
            Body *body = g_bodies+i;
            drawSatShape(body->shape, body->T, COLOR_LIGHT_BLUE);
            drawPoint(body->T.p, COLOR_ORANGE);
        }
        drawMesh(&testTriMesh, Transform::identity(), COLOR_WHITE);
    }

    grabBodies();


    double timer0 = getTime();
    if (g_numBodies)
    {
        ZoneScopedN("Broadphase");


        for(size_t i = 0; i < g_numBodies; i++)
        {
            Body *b = g_bodies+i;
            g_bodyAabbs[i] = Aabb::fromRadius(g_bodies[i].T.p, g_bodies[i].shape->boundingRadius);
        }

        g_bodyBvh.buildFromAabbs(g_bodyAabbs, g_numBodies);
        //bvhDraw(&g_bodyBvh);

        size_t numBodyPairs = 0;

        for (size_t i = 0; i < g_numBodies; i++)
        {
            Body *b = g_bodies+i;

            const size_t MAX_BODIES = 512;
            uint32_t overlapBodies[512];
            size_t numOverlapBodies = g_bodyBvh.overlap(g_bodyAabbs[i], overlapBodies, MAX_BODIES);
            
            b->numBodyPairs = 0;
            b->firstBodyPair = (uint32_t)numBodyPairs;

            if(!numOverlapBodies)
            {
                continue;
            }

            if(numBodyPairs+numOverlapBodies > SOLVER_MAX_MANIFOLDS)
            {
                ods("WARNING: Too many body pairs!\n");
                continue;
            }

            for(size_t j = 0; j < numOverlapBodies; j++)
            {
                if (overlapBodies[j] >= i)
                {
                    continue;
                }
                g_bodyPairs[numBodyPairs++] = overlapBodies[j];
                b->numBodyPairs++;
            }
        }

    }


    TriMeshShape *staticShape = &testTriMesh;
    size_t staticShapeIndex = SOLVER_MAX_BODIES-1;
    g_numManifolds = 0;

    double timer1 = getTime();
    {
        ZoneScopedN("Convex vs Convex Narrowphase");
        for(size_t i = 0; i < g_numBodies; i++)
        {
            Body *bodyA = g_bodies+i;
            size_t indexA = i;

            for(size_t j = 0; j < bodyA->numBodyPairs; j++)
            {
                size_t indexB = g_bodyPairs[bodyA->firstBodyPair + j];
                Body *bodyB = g_bodies+indexB;

                g_numManifolds += generateSatVsSatContacts(bodyA->shape, bodyA->T, indexA,
                                                           bodyB->shape, bodyB->T, indexB, g_manifolds+g_numManifolds, useGraph);
            }
        }
    }

    {
        ZoneScopedN("Convex vs Tri Narrowphase");
        for(size_t i = 0; i < g_numBodies; i++)
        {
            Body *bodyA = g_bodies+i;
            size_t indexA = i;
            g_numManifolds += generateTriMeshVsSatContacts(staticShape, Transform::identity(), staticShapeIndex,
                                                           bodyA->shape, bodyA->T, indexA,
                                                           g_manifolds+g_numManifolds, useGraph);
        }
    }
    double timer2 = getTime();

    {
        ZoneScopedN("Solve");

        g_solverBodies[staticShapeIndex] = {};
        for(size_t i = 0; i < g_numBodies; i++)
        {
            const Body *srcBody = &g_bodies[i];
            SolverBody *dstBody = &g_solverBodies[i];

            dstBody->iI = srcBody->T.R*srcBody->iI*srcBody->T.R.transpose();
            dstBody->iM = srcBody->iM;
            dstBody->p = srcBody->T.p;
            dstBody->v = srcBody->v;
            dstBody->w = srcBody->w;
        }

        solveConstraints(g_manifolds, g_numManifolds, g_solverBodies, g_numBodies);
    }
    double timer3 = getTime();

    for (size_t i = 0; i < g_numBodies; i++)
    {
        const SolverBody *srcBody = &g_solverBodies[i];
        Body *dstBody = &g_bodies[i];
        dstBody->v = srcBody->v;
        dstBody->w = srcBody->w;
    }

    for (size_t i = 0; i < g_numBodies; i++)
    {
        Body *body = &g_bodies[i];

        
        bool isSleeping = body->v.length() < 0.05f && body->w.length() < 0.02f;
        if (isSleeping)
        {
            body->v = {};
            body->w = {};
        }
        else 
        {
            Vec3 w = body->w;
            Mat3 S ={
                {0, w.z, -w.y},
                {-w.z, 0, w.x},
                {w.y, -w.x, 0}
            };

            body->T.R = body->T.R + S*body->T.R*SOLVER_DT;
            body->T.p += body->v*SOLVER_DT;

            body->T.R =body->T.R.orthonormalized();
            body->v *= 0.99f;
            body->w *= 0.99f;
        }

    }

    for (size_t i = 0; i < g_numBodies; i++)
    {
        Body *body = &g_bodies[i];
        if (body->T.p.y < -100)
        {
            *body = g_bodies[--g_numBodies];
            i--;
        }
    }

    ImGui::Begin("Mesh Demo");

    if (ImGui::Button("Back to Demo Browser")) {
        g_demo = DEMO_BROWSER;
    }

    ImGui::LabelText("Hovered Body", "%d", grab.hoverId);
    ImGui::LabelText("Body Count", "%d", g_numBodies);

    static int numBodiesToSpawn = 1;
    ImGui::SliderInt("Num Bodies To Spawn", &numBodiesToSpawn, 1, 8);
    ImGui::SliderFloat("Fly Speed", &g_flySpeed, 5.0f, 100.0f); 
    ImGui::Checkbox("Test Mode", &enableGravity);
    ImGui::Checkbox("Use Optimized SAT", &useGraph);
    float dt = ImGui::GetIO().DeltaTime;
    ImGui::LabelText("FPS", "%.2f", 1.0f / dt);
    ImGui::LabelText("Frame Time", "%.02f ms", dt * 1000);

    ImGui::LabelText("Broadphase",  "%.02f ms", (timer1 - timer0) * 1000);
    ImGui::LabelText("Narrowphase", "%.02f ms", (timer2 - timer1) * 1000);
    ImGui::LabelText("Constraints", "%.02f ms", (timer3 - timer2) * 1000);

    if (ImGui::Button("Spawn Random Bodies"))
    {
        Vec3 dir = g_flycam.rot.c2;
        Vec3 spawnPos = g_flycam.pos + dir*10;
        spawnRandomBodies(spawnPos, numBodiesToSpawn, numBodiesToSpawn, numBodiesToSpawn);
    }

    if (ImGui::Button("Remove All Bodies"))
    {
        g_numBodies = 0;
        g_shapeAllocator.reset();
    }

    ImGui::End();

    //drawConvexity(&testTriMesh);
}


void demoSatTest()
{
    static SatShape *shapeA;
    static SatShape *shapeB;
    static int scrollAmount;
    static int selectedFace;
    static float offset = 0.5f;
    static float angle1 = 221; // 98.76
    static float angle2 = 221; // 0

    static bool showReference = true;
    static TriShape tri;
    static SatShape *hull;
    static bool testTriangle;

    if (!shapeA)
    {
        Vec3 v[] = {
            { 0.0f, 0.0f, 0.0f}, 
            { 1.0f, 0.0f, 0.0f}, 
            { 0.0f, 1.0f, 0.0f}
        };
        tri.setFromVerts(v);
        hull = makeRandomHull({0.2f, 2.2f, 0.5f}, 1.0f, 128);
        shapeA = hull;
        shapeB = makeRandomHull({0.5f, 0.5f, 1.0f}, 1.0f, 32);
        g_flycam.pos = {0,0,-5};


    }

    if (testTriangle)
    {
        shapeA = &tri.sat;
    }
    else
    {
        shapeA = hull;
    }


    Transform xfB = {Mat3::rotY(30*Math::DEG2RAD), { 1.5f, 0, 0}};
    Transform xfA = {Mat3::rotY(angle1*Math::DEG2RAD)*Mat3::rotZ(angle2*Math::DEG2RAD), {offset, 0,0}};

    Transform bToA = Transform::composeNeg(xfA, xfB);

    drawSatShape(shapeA, xfA, COLOR_LIGHT_BLUE);
    drawSatShape(shapeB, xfB, COLOR_LIGHT_GREEN);


    ImGui::Begin("SAT Demo ");
    if (ImGui::Button("Back to Demo Browser")) {
        g_demo = DEMO_BROWSER;
    }
    ImGui::SliderFloat("Offset", &offset, -1, 1);
    ImGui::SliderFloat("Angle1", &angle1, 0, 360);
    ImGui::SliderFloat("Angle2", &angle2, 0, 360);
    ImGui::Checkbox("Test Triangle", &testTriangle);
    ImGui::Separator();


    {
        float mouseSupp = -INFINITY;
        float tHit;
        Vec3 pickOrigin, pickDir;
        getPickRay(pickOrigin, pickDir);
        if(intersectSphere(pickDir, pickOrigin, g_sphereOrigin, 1.0f, &tHit))
        {

            Vec3 hitPoint = pickOrigin + pickDir*tHit;
            Vec3 mouseN = hitPoint - g_sphereOrigin;

            Vec3 nA = mouseN;
            Vec3 nB =bToA.R.transpose()*nA;

            size_t ignore;
            float suppA, suppB;
            getSupport(nA, shapeA->vertPos, shapeA->numVerts, &suppA, &ignore);
            getSupport(nB, shapeB->vertPos, shapeB->numVerts, &suppB, &ignore);

            mouseSupp = suppA+suppB+ bToA.p.dot(nA);

            drawPointEx(hitPoint, COLOR_RED, 0.5f);
        }

        ImGui::Text("Mouse Support (red point) = %f", mouseSupp);
    }

    drawPoint({}, COLOR_WHITE);

    ImVec4 color = ImColor(128,255,128);
    {
        SatResult res;
        satCollideReference(shapeA, xfA, shapeB, xfB, &res);
        if (res.support < 0)
        {
            color = ImColor(255,128,128);
            ImGui::TextColored(color, "Not colliding so values won't match (since exact version earlys out).");
        }

        Vec3 mtv = xfA.inverse().rotate(res.mtv);
        drawPointEx(g_sphereOrigin + mtv, COLOR_ORANGE, 0.4f);
        ImGui::TextColored(color, "Reference Support (orange point) = %f", res.support);
    }

    drawGaussMap(shapeA, Transform::identity(), COLOR_BLUE);
    drawGaussMap(shapeB, bToA, COLOR_GREEN);

    {
        SatResult res;
        satCollideGraph(shapeA, xfA, shapeB, xfB, &res);
        Vec3 mtv = xfA.inverse().rotate(res.mtv);
        drawPointEx(g_sphereOrigin + mtv*1.02f, COLOR_YELLOW, 0.4f);

        ImGui::TextColored(color, "Graph Support (yellow point) = %f", res.support);
    }


    ImGui::End();
}

bool g_rotating = false;

void demoTick()
{
    //ZoneScoped;

    //ConvexHull::test();

    if(ImGui::IsKeyDown(KEY_ESCAPE))
    {
        exit(0);
    }

    setCamera(g_flycam.rot, g_flycam.pos);


    if (!ImGui::GetIO().WantCaptureMouse && ImGui::IsMouseClicked(ImGuiMouseButton_Left))
    {
        g_rotating = true;
    }

    if(!ImGui::IsMouseDown(ImGuiMouseButton_Left)) g_rotating = false;

    if(g_rotating)
    {
        Vec2 mouseDelta = ImGui::GetIO().MouseDelta;

        g_flycam.yaw += 0.004f*mouseDelta.x;
        g_flycam.pitch += 0.004f*mouseDelta.y;
        g_flycam.pitch = Math::clamp(g_flycam.pitch, -Math::PI/2, Math::PI/2);


        g_flycam.rot = Mat3::rotY(g_flycam.yaw)*Mat3::rotX(g_flycam.pitch);

        Vec3 strafe ={};
        if(ImGui::IsKeyDown('W')) strafe.z = 1;
        if(ImGui::IsKeyDown('S')) strafe.z =-1;
        if(ImGui::IsKeyDown('D')) strafe.x = 1;
        if(ImGui::IsKeyDown('A')) strafe.x =-1;
        if(ImGui::IsKeyDown('Q')) strafe.y = 1;
        if(ImGui::IsKeyDown('Z')) strafe.y =-1;
        if(strafe.x||strafe.y||strafe.z) strafe = strafe.normalizedSafe();
        Vec3 flyDir = g_flycam.rot*strafe;
        g_flycam.pos = g_flycam.pos + flyDir*SOLVER_DT*g_flySpeed;
    }


    g_flySpeed += ImGui::GetIO().MouseWheel;
    g_flySpeed = Math::clamp(g_flySpeed, 0.1f, 100.0f);


    {
        if (g_demo == DEMO_BROWSER)
        {
            ImGui::Begin("Demo Browser");
            if (ImGui::Button("SAT Demo"))
            {
                g_demo = DEMO_SAT;
            }
            if (ImGui::Button("Mesh Demo"))
            {
                g_demo = DEMO_MESH;
            }
            ImGui::End();
        }
        if (g_demo == DEMO_SAT) demoSatTest();
        if (g_demo == DEMO_MESH) demoTriMesh();
    }

    //demoSatTest();



    FrameMark;
}

