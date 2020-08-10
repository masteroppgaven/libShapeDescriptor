#pragma once

#include <shapeDescriptor/libraryBuildSettings.h>
#include <cstddef>

namespace ShapeDescriptor {
    namespace gpu {
        struct SpinImageSearchResults {
            size_t resultIndices[SEARCH_RESULT_COUNT];
            float resultScores[SEARCH_RESULT_COUNT];
        };

        struct RadialIntersectionCountImageSearchResults {
            size_t resultIndices[SEARCH_RESULT_COUNT];
            int resultScores[SEARCH_RESULT_COUNT];
        };

        struct QUICCISearchResults {
            size_t resultIndices[SEARCH_RESULT_COUNT];
            int resultScores[SEARCH_RESULT_COUNT];
        };

        struct ShapeContextSearchResults {
            size_t resultIndices[SEARCH_RESULT_COUNT];
            float resultScores[SEARCH_RESULT_COUNT];
        };
    }
}

