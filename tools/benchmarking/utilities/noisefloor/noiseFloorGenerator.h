#pragma once
#include <shapeDescriptor/cpu/types/Mesh.h>
#include <shapeDescriptor/utilities/read/MeshLoader.h>
#include <shapeDescriptor/gpu/radialIntersectionCountImageSearcher.cuh>
#include <shapeDescriptor/gpu/quickIntersectionCountImageSearcher.cuh>
#include <shapeDescriptor/gpu/spinImageSearcher.cuh>
#include <shapeDescriptor/gpu/3dShapeContextSearcher.cuh>
#include <shapeDescriptor/gpu/fastPointFeatureHistogramSearcher.cuh>
#include <shapeDescriptor/utilities/free/mesh.h>
#include <shapeDescriptor/utilities/free/array.h>
#include <benchmarking/utilities/descriptor/RICI.h>
#include <benchmarking/utilities/descriptor/QUICCI.h>
#include <benchmarking/utilities/descriptor/spinImage.h>
#include <benchmarking/utilities/descriptor/3dShapeContext.h>
#include <benchmarking/utilities/descriptor/FPFH.h>
#include <benchmarking/utilities/metadata/generateFakeMetadata.h>
#include <benchmarking/utilities/metadata/transformDescriptor.h>
#include <json.hpp>
#include <vector>

using json = nlohmann::json;

namespace Benchmarking
{
    namespace utilities
    {
        namespace noisefloor
        {
            json generateNoiseFloor(
                std::string objectOnePath,
                std::string objectTwoPath,
                int descriptor,
                std::string hardware,
                float supportRadius = 2.5f,
                float supportAngleDegrees = 60.0f,
                float pointDensityRadius = 0.2f,
                float minSupportRadius = 0.1f,
                float maxSupportRadius = 2.5f,
                size_t pointCloudSampleCount = 200000,
                size_t randomSeed = 4917133789385064);
        }
    }
}