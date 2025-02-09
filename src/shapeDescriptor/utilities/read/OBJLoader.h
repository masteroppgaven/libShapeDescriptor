#pragma once

#include <filesystem>
#include "shapeDescriptor/cpu/types/Mesh.h"

namespace ShapeDescriptor {
    namespace utilities {
        cpu::Mesh loadOBJ(std::filesystem::path, bool recomputeNormals = false);
    }
}