#pragma once
#include <shapeDescriptor/cpu/types/Mesh.h>
#include <shapeDescriptor/gpu/types/Mesh.h>
#include <shapeDescriptor/cpu/radialIntersectionCountImageGenerator.h>
#include <shapeDescriptor/gpu/radialIntersectionCountImageGenerator.cuh>
#include <shapeDescriptor/utilities/copy/mesh.h>
#include <vector>
#include <map>
#include <variant>
#include <ctime>
#include <chrono>

namespace Benchmarking
{
    namespace utilities
    {
        namespace descriptor
        {
            ShapeDescriptor::cpu::array<ShapeDescriptor::RICIDescriptor> generateRICIDescriptor(
                ShapeDescriptor::cpu::Mesh mesh,
                std::string hardware,
                float supportRadius,
                std::chrono::duration<double> &elapsedTime);
        }
    }
}