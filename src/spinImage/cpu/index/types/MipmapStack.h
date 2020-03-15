#pragma once

#include <spinImage/libraryBuildSettings.h>
#include <iostream>
#include <cassert>
#include <bitset>
#include <spinImage/cpu/types/QuiccImage.h>

constexpr int uintsPerRow = spinImageWidthPixels / 32;

// bits per pixel:                                   2 -> 4      4 -> 8      8 -> 16     16 -> 32
const unsigned int groupingShiftBitMasks_2bits[4] = {0x33333333, 0x0F0F0F0F, 0x00FF00FF, 0x0000FFFF};

// Shift all bits to the end, so they are all collected together.
inline unsigned int compressChunk_2bits(unsigned int chunk) {
    unsigned int shiftIndex = 0;
    for(unsigned int shiftDistance = 1; shiftDistance < 16; shiftDistance *= 2) {
        chunk = (chunk | (chunk >> shiftDistance)) & groupingShiftBitMasks_2bits[shiftIndex];
        shiftIndex++;
    }
    return chunk;
}

inline unsigned long bitwiseTranspose8x8(unsigned long in) {
    unsigned long mask = 0x8040201008040201ULL;
    unsigned long out = in & mask;
    for(unsigned int s = 7; s <= 49; s += 7) {
        mask = mask >> 8U;
        out = out | ((in & mask) << s) | ((in >> s) & mask);
    }
    return out;
}

inline void bitwiseTranspose32x32(std::array<unsigned int, 32> &in) {
    unsigned int mask = 0x0000FFFF;
    unsigned int temp = 0;
    for(unsigned int j = 16; j != 0; j = j >> 1U, mask = mask ^ (mask << j)) {
        for(unsigned int k = 0; k < 32; k = (k + j + 1) & ~j) {
            temp = (in[k] ^ (in[k + j] >> j)) & mask;
            in[k] = in[k] ^ temp;
            in[k + j] = in[k + j] ^ (temp << j);
        }
    }
}

struct MipMapLevel3 {
    // const left out on purpose because assignments require it to be non-const.
    // Just C++ things..
    std::array<unsigned int, 32> image;

    unsigned int computeLevelByte(unsigned int level) const {
        const unsigned short levelByteIndexWithinLevel3 = level - 8 - (16 * 2);
        const unsigned short level3UintIndex = levelByteIndexWithinLevel3 / 4;
        const unsigned short level3ByteIndex = levelByteIndexWithinLevel3 % 4;
        const unsigned int level3Uint = image[level3UintIndex];
        return (level3Uint >> (32U - 8U * (level3ByteIndex + 1))) & 0xFFU;
    }

    // 64x64 -> 32x32 image
    static std::array<unsigned int, 32> computeMipmapLevel3(const QuiccImage &quiccImage) {
        std::array<unsigned int, 32> level3;

        for(int row = 0; row < spinImageWidthPixels; row += 2) {
            unsigned int topLeftChunk = quiccImage[row * uintsPerRow + 0];
            unsigned int topRightChunk = quiccImage[row * uintsPerRow + 1];
            unsigned int bottomLeftChunk = quiccImage[(row + 1) * uintsPerRow + 0];
            unsigned int bottomRightChunk = quiccImage[(row + 1) * uintsPerRow + 1];

            unsigned int topLeftCombined = (topLeftChunk | (topLeftChunk >> 1U)) & 0x55555555U;
            unsigned int bottomLeftCombined = (bottomLeftChunk | (bottomLeftChunk >> 1U)) & 0x55555555U;
            unsigned int topRightCombined = (topRightChunk | (topRightChunk >> 1U)) & 0x55555555U;
            unsigned int bottomRightCombined = (bottomRightChunk | (bottomRightChunk >> 1U)) & 0x55555555U;

            unsigned int compressedLeftChunk = compressChunk_2bits(topLeftCombined | bottomLeftCombined);
            unsigned int compressedRightChunk = compressChunk_2bits(topRightCombined | bottomRightCombined);

            level3[row / 2] = (compressedLeftChunk << 16U) | compressedRightChunk;
        }

        // Compute matrix transpose
        bitwiseTranspose32x32(level3);

        // Rearranging bits for more balanced tree referencing
        for(int columnIndex = 0; columnIndex < 32; columnIndex++) {
            unsigned int column = level3[columnIndex];
            level3[columnIndex] =
                ((column << 12U) & 0xFF000000U) |
                ((column       ) & 0x00F00000U) |
                ((column << 8U ) & 0x000F0000U) |
                ((column >> 12U) & 0x0000F000U) |
                ((column << 8U ) & 0x00000F00U) |
                ((column >> 24U) & 0x000000F0U) |
                ((column       ) & 0x0000000FU);
        }

        return level3;
    }

    MipMapLevel3(const QuiccImage &quicciImage) : image(computeMipmapLevel3(quicciImage)) {}

    // Constructor intended for array initialisation
    MipMapLevel3() : image({0}) {}
};

struct MipMapLevel2 {
    const std::array<unsigned int, 8> image;

    unsigned int computeLevelByte(unsigned int level) const {
        assert(level >= 8);
        assert(level < 8 + (16 * 2));
        const unsigned short levelByteIndexWithinLevel2 = level - 8;
        const unsigned short level2UintIndex = levelByteIndexWithinLevel2 / 4;
        const unsigned short level2ByteIndex = levelByteIndexWithinLevel2 % 4;
        const unsigned int level2Uint = image[level2UintIndex];
        return (level2Uint >> (32U - 8U * (level2ByteIndex + 1))) & 0xFFU;
    }

    // 32x32 -> 16x16 image
    static std::array<unsigned int, 8> computeMipmapLevel2(MipMapLevel3 level3) {
        unsigned int combinedCompressedChunk = 0;
        std::array<unsigned int, 8> level2;

        for(unsigned int row = 0; row < 32; row += 2) {
            unsigned int topChunk = level3.image[row];
            unsigned int bottomChunk = level3.image[row + 1];

            unsigned int topCombined = (topChunk | (topChunk >> 1U)) & 0x55555555U;
            unsigned int bottomCombined = (bottomChunk | (bottomChunk >> 1U)) & 0x55555555U;

            unsigned int compressedChunk = compressChunk_2bits(topCombined | bottomCombined);

            // We need 2 16 bit compressed chunks to make 1 32-bit output chunk.
            // So we save every other of these chunks, and only write when we need to.
            if((row & 0x2U) == 0) {
                combinedCompressedChunk = (compressedChunk << 16U);
            } else {
                level2[row / 4] = combinedCompressedChunk | compressedChunk;
            }
        }

        return level2;
    }

    MipMapLevel2(MipMapLevel3 level3Image) : image(computeMipmapLevel2(level3Image)) {}
};

struct MipMapLevel1 {
    const unsigned long image;

    unsigned char computeLevelByte(const unsigned char level) const {
        assert(level < 8);
        return (image >> (64U - 8U * (level + 1))) & 0xFFU;
    }

    // 16x16 -> 8x8 image
    static unsigned long computeMipmapLevel1(MipMapLevel2 level2) {
        unsigned long level1 = 0;

        for(unsigned int chunk = 0; chunk < 8; chunk++) {
            unsigned int doubleRowChunk = level2.image[chunk];

            unsigned int combined = (doubleRowChunk | (doubleRowChunk >> 1U)) & 0x55555555U;
            combined = (combined | (combined >> 16U)) & 0x00005555U;

            // Only 8 rightmost bits can be set at this point.
            // Unsigned long is needed because next operation requires space for shifting
            unsigned long compressedChunk = compressChunk_2bits(combined);

            // Every 4 chunks produces one output chunk
            // (8 bits per processed chunk)
            level1 |= compressedChunk << (64U - 8U * (chunk + 1U));
        }

        return level1;
    }

    MipMapLevel1(MipMapLevel2 higherLevelImage) : image(computeMipmapLevel1(higherLevelImage)) {}
};

struct MipmapStack {
    //   level   mipmap size    pixel count   area per pixel   space needed
    //   1       8x8 pixels     64            8x8 pixels       2 unsigned ints (transposed!)
    //   2       16x16 pixels   256           4x4 pixels       8 unsigned ints
    //   3       32x32 pixels   1024          2x2 pixels       32 unsigned ints
    //   -- 64x64: source --

    // The order in which these are defined matters due to the initialiser list of the constructor
    MipMapLevel3 level3;
    MipMapLevel2 level2;
    MipMapLevel1 level1;

    MipmapStack(const QuiccImage &quiccImage) :
            level3(quiccImage),
            level2(level3),
            level1(level2) {
        static_assert(spinImageWidthPixels == 64, "The MipmapStack class assumes the original input image has dimensions 64x64.");
    }

    unsigned char computeLevelByte(const unsigned short level) const {
        // Level 1 contains 8 1-byte chunks
        if(level < 8) {
            return level1.computeLevelByte(level);
        }

        // Level 2 starts after level 1, and contains 16 columns of 2 chunks each
        const unsigned short level3StartChunk = 8 + (16 * 2);
        if(level < level3StartChunk) {
            return level2.computeLevelByte(level);
        }

        // Level 3 starts after level 2, and contains 32 columns of 4 chunks each
        // Repeating the final byte is a hack
        return level3.computeLevelByte(std::min<unsigned int>(level, level3StartChunk + (32 * 4) - 1));
    }

    template<typename printedType, int intCount> void printBitwiseImage(const std::array<printedType, intCount> &image, int size) {
        unsigned int bitIndex = 0;
        unsigned int byteIndex = 0;
        const unsigned int bitsPerType = sizeof(printedType) * 8;
        for(int row = 0; row < size; row++) {
            for(int col = 0; col < size; col++) {
                printedType currentBits = image[byteIndex];
                std::cout << (((currentBits >> (bitsPerType - 1U - bitIndex)) & 0x1U) == 1 ? "1" : ".");
                bitIndex++;
                if(bitIndex == bitsPerType) {
                    byteIndex++;
                    bitIndex = 0;
                }
            }
            std::cout << std::endl;
        }
    }

    void print() {
        std::cout << std::endl << "Level 1" << std::endl;
        std::array<unsigned int, 2> temp = {(unsigned int) (level1.image >> 32), (unsigned int) level1.image};
        printBitwiseImage<unsigned int, 2>(temp, 8);

        std::cout << std::endl << "Level 2" << std::endl;
        printBitwiseImage<unsigned int, 8>(level2.image, 16);

        std::cout << std::endl << "Level 3" << std::endl;
        printBitwiseImage<unsigned int, 32>(level3.image, 32);
    }

    static QuiccImage combine(
            const QuiccImage &image1,
            const QuiccImage &image2) {
        QuiccImage combinedImage;
        for(unsigned int i = 0; i < 128; i++) {
            combinedImage[i] = image1[i] | image2[i];
        }
        return combinedImage;
    }
};