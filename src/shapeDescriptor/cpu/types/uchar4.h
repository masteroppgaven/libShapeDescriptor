#pragma once

#include <string>

namespace ShapeDescriptor {
    namespace cpu {
        struct uchar4 {

            // Aliases for addressing the same fields with different names
            union { unsigned char x = 0; unsigned char r; };
            union { unsigned char y = 0; unsigned char g; };
            union { unsigned char z = 0; unsigned char b; };
            union { unsigned char w = 0; unsigned char a; };

            uchar4() = default;
            uchar4(unsigned char x, unsigned char y, unsigned char z, unsigned char w) : x(x), y(y), z(z), w(w) {}

            std::string to_string() {
                return "(" + std::to_string(x) + ", " + std::to_string(y) + ", " + std::to_string(z) + ", " + std::to_string(w) + ")";
            }
        };
    }
}