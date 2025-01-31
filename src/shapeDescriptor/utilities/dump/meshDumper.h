#pragma once

#include <shapeDescriptor/cpu/types/Mesh.h>
#include <filesystem>
#include <shapeDescriptor/cpu/types/array.h>
#include <shapeDescriptor/gpu/types/float2.h>

namespace ShapeDescriptor {
    namespace dump {
        void mesh(cpu::Mesh mesh, const std::filesystem::path outputFile);
        void mesh(cpu::Mesh mesh, const std::filesystem::path outputFilePath,
                size_t highlightStartVertex, size_t highlightEndVertex);
        void mesh(cpu::Mesh mesh, const std::filesystem::path &outputFilePath,
                ShapeDescriptor::cpu::array<float2> vertexTextureCoordinates, std::string textureMapPath);
    }
}