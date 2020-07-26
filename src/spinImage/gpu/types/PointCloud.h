#pragma once

#include <spinImage/gpu/types/DeviceVertexList.cuh>

namespace SpinImage {
    namespace gpu {
        struct PointCloud {
            DeviceVertexList vertices;
            DeviceVertexList normals;
            size_t pointCount;

            PointCloud(size_t pointCount) : vertices(pointCount), normals(pointCount), pointCount(pointCount) {}

            void free() {
                vertices.free();
                normals.free();
            }
        };
    }


}