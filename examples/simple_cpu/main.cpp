#include <shapeDescriptor/common/types/OrientedPoint.h>
#include <shapeDescriptor/cpu/types/Mesh.h>
#include <shapeDescriptor/cpu/radialIntersectionCountImageGenerator.h>
#include <shapeDescriptor/utilities/read/MeshLoader.h>
#include <shapeDescriptor/utilities/spinOriginsGenerator.h>
#include <shapeDescriptor/utilities/free/array.h>
#include <shapeDescriptor/utilities/free/mesh.h>
#include <shapeDescriptor/utilities/dump/descriptorImages.h>

int main(int argc, char** argv) {
    if(argc == 1) {
        std::cout << "Usage: simple_cpu [file_to_read.obj/.ply/.off]" << std::endl;
        return 1;
    }

    // Load mesh
    const bool recomputeNormals = false;
    std::string fileToRead = std::string(argv[1]);
    ShapeDescriptor::cpu::Mesh mesh = ShapeDescriptor::utilities::loadMesh(fileToRead, recomputeNormals);
        
    // Define and upload descriptor origins
    ShapeDescriptor::cpu::array<ShapeDescriptor::OrientedPoint> descriptorOrigins = ShapeDescriptor::utilities::generateUniqueSpinOriginBuffer(mesh);

    // Compute the descriptor(s)
    float supportRadius = 1.0;
    ShapeDescriptor::cpu::array<ShapeDescriptor::RICIDescriptor> descriptors = 
        ShapeDescriptor::cpu::generateRadialIntersectionCountImages(
                mesh,
                descriptorOrigins,
                supportRadius);

                    
    // Do something with descriptors here, for example write the first 5000 to an image file
    descriptors.length = std::min<size_t>(descriptors.length, 5000);
    ShapeDescriptor::dump::descriptors(descriptors, "output_image.png");

    // Free memory
    ShapeDescriptor::free::array(descriptorOrigins);
    ShapeDescriptor::free::array(descriptors);
    ShapeDescriptor::free::mesh(mesh);
}