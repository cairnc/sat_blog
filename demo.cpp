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

ArenaAllocator g_shapeAllocator;

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


void demoInit()
{
    g_shapeAllocator.init(1024*1024*8);
    g_flycam.pos ={};
    g_flycam.rot = Mat3::identity();
}


#define TIMER(...)

void dumpShape(const SatShape *s, Transform T)
{
    ods("constexpr uint8_t numFaces = %d;\n", s->numFaces);
    ods("constexpr uint8_t numVerts = %d;\n", s->numVerts);
    ods("constexpr uint8_t numEdges = %d;\n", s->numEdges);
    ods("static SatShape::EdgeList faces[numFaces] = {");
    for (int i = 0; i < s->numFaces; i++)
    {
        ods("{%d, %d}, ", s->faces[i].first, s->faces[i].num);
    }
    ods("};\n");

    ods("static SatShape::EdgeList verts[numVerts] = {");
    for (int i = 0; i < s->numVerts; i++)
    {
        ods("{%d, %d}, ", s->verts[i].first, s->verts[i].num);
    }
    ods("};\n");

    ods("static Vec3 vertPos[numVerts] = {");
    for (int i = 0; i < s->numVerts; i++)
    {
        ods("{%a, %a, %a}, ", s->vertPos[i].x, s->vertPos[i].y, s->vertPos[i].z);
    }
    ods("};\n");

    ods("static Plane facePlanes[numFaces] = {");
    for (int i = 0; i < s->numFaces; i++)
    {
        ods("{{%a, %a, %a}, %a}, ", s->facePlanes[i].normal.x, s->facePlanes[i].normal.y, s->facePlanes[i].normal.z, s->facePlanes[i].dist);
    }
    ods("};\n");

    ods("static SatShape::Edge faceEdges[numEdges * 2] = {");
    for (int i = 0; i < s->numEdges * 2; i++)
    {
        ods("{%d, %d, %d, %d}, ", s->faceEdges[i].v0, s->faceEdges[i].v1, s->faceEdges[i].f0, s->faceEdges[i].f1);
    }
    ods("};\n");

    ods("static SatShape::Edge vertEdges[numEdges * 2] = {");
    for (int i = 0; i < s->numEdges * 2; i++)
    {
        ods("{%d, %d, %d, %d}, ", s->vertEdges[i].v0, s->vertEdges[i].v1, s->vertEdges[i].f0, s->vertEdges[i].f1);
    }
    ods("};\n");

    ods("static SatShape::Edge edges[numEdges] = {");
    for (int i = 0; i < s->numEdges; i++)
    {
        ods("{%d, %d, %d, %d}, ", s->edges[i].v0, s->edges[i].v1, s->edges[i].f0, s->edges[i].f1);
    }
    ods("};\n");

    ods(R"FOO(
static SatShape shape;
static Transform T;
shape.numFaces = numFaces;
shape.numVerts = numVerts;
shape.numEdges = numEdges;
shape.faces = faces;
shape.verts = verts;
shape.vertPos = vertPos;
shape.facePlanes = facePlanes;
shape.faceEdges = faceEdges;
shape.vertEdges = vertEdges;
shape.edges = edges;
T.R = {{%a, %a, %a}, {%a, %a, %a}, {%a, %a, %a}};
T.p = {%a, %a, %a};
)FOO",
        T.R.c0.x, T.R.c0.y, T.R.c0.z,
        T.R.c1.x, T.R.c1.y, T.R.c1.z,
        T.R.c2.x, T.R.c2.y, T.R.c2.z,
        T.p.x, T.p.y, T.p.z);


}


void SetFlyCam(Vec3 pos, float yaw, float pitch)
{
    g_flycam.pos = pos;
    g_flycam.yaw = yaw;
    g_flycam.pitch = pitch;
    g_flycam.rot = Mat3::rotY(g_flycam.yaw) * Mat3::rotX(g_flycam.pitch);
}

struct GrabState
{
    bool grabbing;
    Vec3 bodyP;
    float dist;
    uint32_t bodyId;
    uint32_t hoverId;
};


struct DemoTriMesh
{
    bool enableGravity;
    bool useGraph;

    GrabState grab;
    Aabb bodyAabbs[SOLVER_MAX_BODIES];
    Body bodies[SOLVER_MAX_BODIES];
    uint16_t bodyPairs[SOLVER_MAX_MANIFOLDS];
    size_t numBodies;
    SolverManifold manifolds[SOLVER_MAX_MANIFOLDS];
    size_t numManifolds;
    SolverBody solverBodies[SOLVER_MAX_BODIES];
    Bvh bodyBvh;
    TriMeshShape testTriMesh;

    void init()
    {
        g_shapeAllocator.reset();
        enableGravity = true;
        useGraph = false;

        if (testTriMesh.numTris == 0)
        {
            loadTriMeshShape(&testTriMesh, "test_meshes/ascension.obj");
        }

        SetFlyCam({ 10.091136f,6.8672866f,8.7661008f }, -2.2599978f, 0.34799960f);
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

                    Body *body = bodies+numBodies++;
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

    void grabBodies()
    {
        Vec3 pickOrigin, pickDir;
        getPickRay(pickOrigin, pickDir);


        Body *hoverBody = NULL;
        float hoverDist = INFINITY;
        grab.hoverId = -1;

        if (!ImGui::GetIO().WantCaptureMouse)
        {
            for(size_t i = 0; i < numBodies; i++)
            {
                Body *body = bodies+i;
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

            if (hoverBody && ImGui::IsKeyPressed('P'))
            {
                dumpShape(hoverBody->shape, hoverBody->T);
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

            for (size_t i = 0; i < numBodies; i++) 
            {
                if (bodies[i].id == grab.bodyId) 
                { 
                    grabbedBody = bodies+i; 
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

    void tick()
    {
        ZoneScoped;


        if (!testTriMesh.numTris)
        {
        }

        if (enableGravity)
        {
            for (size_t i = 0; i < numBodies; i++)
            {
                Body *body = bodies + i;
                float gravity = 20.8f;
                body->v.y -= gravity * SOLVER_DT;
            }
        }

        {
            ZoneScopedN("Draw Bodies");

            for (size_t i = 0; i < numBodies; i++)
            {
                Body *body = bodies + i;
                drawSatShape(body->shape, body->T, COLOR_LIGHT_BLUE);
                drawPoint(body->T.p, COLOR_ORANGE);
            }
            drawMesh(&testTriMesh, Transform::identity(), COLOR_WHITE);
        }

        grabBodies();


        double timer0 = getTime();
        if (numBodies)
        {
            ZoneScopedN("Broadphase");


            for (size_t i = 0; i < numBodies; i++)
            {
                Body *b = bodies + i;
                bodyAabbs[i] = Aabb::fromRadius(bodies[i].T.p, bodies[i].shape->boundingRadius);
            }

            bodyBvh.buildFromAabbs(bodyAabbs, numBodies);
            //bvhDraw(&bodyBvh);

            size_t numBodyPairs = 0;

            for (size_t i = 0; i < numBodies; i++)
            {
                Body *b = bodies + i;

                const size_t MAX_BODIES = 512;
                uint32_t overlapBodies[512];
                size_t numOverlapBodies = bodyBvh.overlap(bodyAabbs[i], overlapBodies, MAX_BODIES);

                b->numBodyPairs = 0;
                b->firstBodyPair = (uint32_t)numBodyPairs;

                if (!numOverlapBodies)
                {
                    continue;
                }

                if (numBodyPairs + numOverlapBodies > SOLVER_MAX_MANIFOLDS)
                {
                    ods("WARNING: Too many body pairs!\n");
                    continue;
                }

                for (size_t j = 0; j < numOverlapBodies; j++)
                {
                    if (overlapBodies[j] >= i)
                    {
                        continue;
                    }
                    bodyPairs[numBodyPairs++] = overlapBodies[j];
                    b->numBodyPairs++;
                }
            }

        }


        TriMeshShape *staticShape = &testTriMesh;
        size_t staticShapeIndex = SOLVER_MAX_BODIES - 1;
        numManifolds = 0;

        double timer1 = getTime();
        {
            ZoneScopedN("Convex vs Convex Narrowphase");
            for (size_t i = 0; i < numBodies; i++)
            {
                Body *bodyA = bodies + i;
                size_t indexA = i;

                for (size_t j = 0; j < bodyA->numBodyPairs; j++)
                {
                    size_t indexB = bodyPairs[bodyA->firstBodyPair + j];
                    Body *bodyB = bodies + indexB;

                    numManifolds += generateSatVsSatContacts(bodyA->shape, bodyA->T, indexA,
                                                               bodyB->shape, bodyB->T, indexB, manifolds + numManifolds, useGraph);
                }
            }
        }

        {
            ZoneScopedN("Convex vs Tri Narrowphase");
            for (size_t i = 0; i < numBodies; i++)
            {
                Body *bodyA = bodies + i;
                size_t indexA = i;
                numManifolds += generateTriMeshVsSatContacts(staticShape, Transform::identity(), staticShapeIndex,
                                                               bodyA->shape, bodyA->T, indexA,
                                                               manifolds + numManifolds, useGraph);
            }
        }
        double timer2 = getTime();

        {
            ZoneScopedN("Solve");

            solverBodies[staticShapeIndex] = {};
            for (size_t i = 0; i < numBodies; i++)
            {
                const Body *srcBody = &bodies[i];
                SolverBody *dstBody = &solverBodies[i];

                dstBody->iI = srcBody->T.R * srcBody->iI * srcBody->T.R.transpose();
                dstBody->iM = srcBody->iM;
                dstBody->p = srcBody->T.p;
                dstBody->v = srcBody->v;
                dstBody->w = srcBody->w;
            }

            solveConstraints(manifolds, numManifolds, solverBodies, numBodies);
        }
        double timer3 = getTime();

        for (size_t i = 0; i < numBodies; i++)
        {
            const SolverBody *srcBody = &solverBodies[i];
            Body *dstBody = &bodies[i];
            dstBody->v = srcBody->v;
            dstBody->w = srcBody->w;
        }

        for (size_t i = 0; i < numBodies; i++)
        {
            Body *body = &bodies[i];


            bool isSleeping = body->v.length() < 0.05f && body->w.length() < 0.02f;
            if (isSleeping)
            {
                body->v = {};
                body->w = {};
            }
            else
            {
                Vec3 w = body->w;
                Mat3 S = {
                    {0, w.z, -w.y},
                    {-w.z, 0, w.x},
                    {w.y, -w.x, 0}
                };

                body->T.R = body->T.R + S * body->T.R * SOLVER_DT;
                body->T.p += body->v * SOLVER_DT;

                body->T.R = body->T.R.orthonormalized();
                body->v *= 0.99f;
                body->w *= 0.99f;
            }

        }

        for (size_t i = 0; i < numBodies; i++)
        {
            Body *body = &bodies[i];
            if (body->T.p.y < -100)
            {
                *body = bodies[--numBodies];
                i--;
            }
        }

        ImGui::Begin("Mesh Demo");

        if (ImGui::Button("Back to Demo Browser"))
        {
            g_demo = DEMO_BROWSER;
        }

        ImGui::LabelText("Hovered Body", "%d", grab.hoverId);
        ImGui::LabelText("Body Count", "%d", numBodies);

        static int numBodiesToSpawn = 1;
        ImGui::SliderInt("Num Bodies To Spawn", &numBodiesToSpawn, 1, 8);
        ImGui::SliderFloat("Fly Speed", &g_flySpeed, 5.0f, 100.0f);
        ImGui::Checkbox("Gravity", &enableGravity);
        ImGui::Checkbox("Use Graph SAT", &useGraph);
        float dt = ImGui::GetIO().DeltaTime;
        ImGui::LabelText("FPS", "%.2f", 1.0f / dt);
        ImGui::LabelText("Frame Time", "%.02f ms", dt * 1000);

        ImGui::LabelText("Broadphase", "%.02f ms", (timer1 - timer0) * 1000);
        ImGui::LabelText("Narrowphase", "%.02f ms", (timer2 - timer1) * 1000);
        ImGui::LabelText("Constraints", "%.02f ms", (timer3 - timer2) * 1000);

        if (ImGui::Button("Spawn Random Bodies"))
        {
            Vec3 dir = g_flycam.rot.c2;
            Vec3 spawnPos = g_flycam.pos + dir * 10;
            spawnRandomBodies(spawnPos, numBodiesToSpawn, numBodiesToSpawn, numBodiesToSpawn);
        }

        if (ImGui::Button("Remove All Bodies"))
        {
            numBodies = 0;
            g_shapeAllocator.reset();
        }

        ImGui::End();

        //drawConvexity(&testTriMesh);
    }


};


struct DemoSatTest
{
    SatShape *shapeA;
    SatShape *shapeB;
    float offset;
    float angle1; // 98.76
    float angle2; // 0

    TriShape tri;
    SatShape *hull;
    bool testTriangle;

    Vec3 shapeABounds;
    Vec3 shapeBBounds;
    int numVertsA;
    int numVertsB;

    void init()
    {
        g_shapeAllocator.reset();

        offset = 0.23f;
        angle1 = 180; // 98.76
        angle2 = 243.71f; // 0

        testTriangle = false;

        shapeABounds = { 0.5f, 0.5f, 0.5f };
        shapeBBounds = { 1.0f, 0.5f, 0.3f };
        numVertsA = 32;
        numVertsB = 64;

        Vec3 v[] = {
            { 0.0f, 0.0f, 0.0f}, 
            { 1.0f, 0.0f, 0.0f}, 
            { 0.0f, 1.0f, 0.0f}
        };
        tri.setFromVerts(v);

        shapeA = makeRandomHull(shapeABounds, 1.0f, numVertsA);
        shapeB = makeRandomHull(shapeBBounds, 1.0f, numVertsB);
        hull = shapeB;

        SetFlyCam({0, 0, -5}, 0, 0);
    }

    void tick()
    {
        Transform xfB = { Mat3::rotY(30 * Math::DEG2RAD), { 1.5f, 0, 0} };
        Transform xfA = { Mat3::rotY(angle1 * Math::DEG2RAD) * Mat3::rotZ(angle2 * Math::DEG2RAD), {offset, 0,0} };


        Transform bToA = Transform::composeNeg(xfA, xfB);

        drawSatShape(shapeA, xfA, COLOR_LIGHT_BLUE);
        drawSatShape(shapeB, xfB, COLOR_LIGHT_GREEN);


        ImGui::Begin("SAT Demo ");
        if (ImGui::Button("Back to Demo Browser"))
        {
            g_demo = DEMO_BROWSER;
        }
        ImGui::SliderFloat("Offset", &offset, -1, 1);
        ImGui::SliderFloat("Angle1", &angle1, 0, 360);
        ImGui::SliderFloat("Angle2", &angle2, 0, 360);

        ImGui::Separator();
        ImGui::DragFloat3("Shape A Bounds", &shapeABounds.x, 0.1f, 0.1f, 5.0f);
        ImGui::DragInt("Shape A Num Verts", &numVertsA, 1.0f, 5, 128);


        ImGui::DragFloat3("Shape B Bounds", &shapeBBounds.x, 0.1f, 0.1f, 5.0f);
        ImGui::DragInt("Shape B Num Verts", &numVertsB, 1.0f, 5, 128);


        if (ImGui::Button("Generate Shapes"))
        {
            shapeA = makeRandomHull(shapeABounds, 1.0f, numVertsA);
            shapeB = makeRandomHull(shapeBBounds, 1.0f, numVertsB);
            hull = shapeB;
        }

        ImGui::Separator();

        if (ImGui::Checkbox("Test Triangle", &testTriangle))
        {
            if (testTriangle)
            {
                shapeA = &tri.sat;
            }
            else
            {
                shapeA = hull;
            }
        }



        {
            float mouseSupp = -INFINITY;
            float tHit;
            Vec3 pickOrigin, pickDir;
            getPickRay(pickOrigin, pickDir);
            if (intersectSphere(pickDir, pickOrigin, g_sphereOrigin, 1.0f, &tHit))
            {

                Vec3 hitPoint = pickOrigin + pickDir * tHit;
                Vec3 mouseN = hitPoint - g_sphereOrigin;

                Vec3 nA = mouseN;
                Vec3 nB = bToA.R.transpose() * nA;

                size_t ignore;
                float suppA, suppB;
                getSupport(nA, shapeA->vertPos, shapeA->numVerts, &suppA, &ignore);
                getSupport(nB, shapeB->vertPos, shapeB->numVerts, &suppB, &ignore);

                mouseSupp = suppA + suppB + bToA.p.dot(nA);

                drawPointEx(hitPoint, COLOR_RED, 0.5f);
            }

            ImGui::Text("Mouse Support (red point) = %f", mouseSupp);
        }

        drawPoint({}, COLOR_WHITE);

        SatResult res1;
        SatResult res2;
        satCollideGraph(shapeA, xfA, shapeB, xfB, &res2);
        satCollideReference(shapeA, xfA, shapeB, xfB, &res1);

        ImVec4 color = ImColor(128, 255, 128);

        drawGaussMap(shapeA, Transform::identity(), COLOR_BLUE);
        drawGaussMap(shapeB, bToA, COLOR_GREEN);

        if (res1.support < 0)
        {
            color = ImColor(255, 128, 128);
            ImGui::TextColored(color, "Not colliding so values won't match (early out conditions are different).");
        }


        Vec3 mtv1 = xfA.inverse().rotate(res1.mtv);
        drawPointEx(g_sphereOrigin + mtv1, COLOR_ORANGE, 0.4f);

        Vec3 mtv2 = xfA.inverse().rotate(res2.mtv);
        drawPointEx(g_sphereOrigin + mtv2 * 1.02f, COLOR_YELLOW, 0.4f);


        const char *featureNames[] = {
            "NONE",
            "FACE_VERT",
            "VERT_FACE",
            "EDGE_EDGE",
        };


        if (ImGui::BeginTable("Result Comparison", 3, ImGuiTableFlags_RowBg))
        {
            ImGui::TableNextColumn();
            ImGui::TableNextColumn(); ImGui::Text("Reference");
            ImGui::TableNextColumn(); ImGui::Text("Graph");

            ImGui::TableNextColumn(); ImGui::Text("Support");
            ImGui::TableNextColumn(); ImGui::Text("%f", res1.support);
            ImGui::TableNextColumn(); ImGui::Text("%f", res2.support);

            ImGui::TableNextColumn(); ImGui::Text("MTV");
            ImGui::TableNextColumn(); ImGui::Text("%.04f, %.04f, %.04f", res1.mtv.x, res1.mtv.y, res1.mtv.z);
            ImGui::TableNextColumn(); ImGui::Text("%.04f, %.04f, %.04f", res2.mtv.x, res2.mtv.y, res2.mtv.z);

            ImGui::TableNextColumn(); ImGui::Text("Features");
            ImGui::TableNextColumn(); ImGui::Text("%s", featureNames[res1.type]);
            ImGui::TableNextColumn(); ImGui::Text("%s", featureNames[res2.type]);

            ImGui::EndTable();
        }

        ImGui::End();
    }



};



bool g_rotating = false;
static DemoSatTest g_satDemo;
static DemoTriMesh g_meshDemo;

void demoTick()
{
    //ZoneScoped;

    //ConvexHull::test();

    if(ImGui::IsKeyDown(KEY_ESCAPE))
    {
        exit(0);
    }

    if(!ImGui::GetIO().WantCaptureMouse && ImGui::IsMouseDown(ImGuiMouseButton_Left))
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
                g_satDemo.init();
            }
            if (ImGui::Button("Mesh Demo"))
            {
                g_demo = DEMO_MESH;
                g_meshDemo.init();
            }
            ImGui::End();
        }
        if (g_demo == DEMO_SAT) g_satDemo.tick();
        if (g_demo == DEMO_MESH) g_meshDemo.tick();
    }

    setCamera(g_flycam.rot, g_flycam.pos);



    FrameMark;
}

