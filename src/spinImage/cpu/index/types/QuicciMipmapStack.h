#pragma once

struct QuicciMipmapStack {

    //   level   mipmap size    pixel count   area per pixel   value range   space needed
    //   0       4x4 images     16            16x16 pixels     0-256         16 bytes, 8 bits/pixel
    //   1       8x8 images     64            8x8 pixels       0-64          64 bytes, 8 (6) bits/pixel
    //   2       16x16 images   256           4x4 pixels       0-16          128 bytes, 4 bits/pixel
    //   3       32x32 images   1024          2x2 pixels       0-4           256 bytes, 2 bits/pixel
    //   -- 64x64: source --

    unsigned int[4]  level0;
    unsigned int[16] level1;
    unsigned int[32] level2;
    unsigned int[64] level3;

    unsigned int[128] quiccImage;
};