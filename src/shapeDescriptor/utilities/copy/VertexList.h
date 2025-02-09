#pragma once

#include <shapeDescriptor/cpu/types/float3.h>
#include <shapeDescriptor/gpu/types/VertexList.cuh>
#include <shapeDescriptor/cpu/types/array.h>

namespace ShapeDescriptor {
    namespace copy {
        ShapeDescriptor::gpu::VertexList hostVertexListToDevice(ShapeDescriptor::cpu::array<ShapeDescriptor::cpu::float3> hostArray);
        ShapeDescriptor::cpu::array<ShapeDescriptor::cpu::float3> deviceVertexListToHost(ShapeDescriptor::gpu::VertexList vertexList);
    }
}