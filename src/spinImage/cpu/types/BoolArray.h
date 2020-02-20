#pragma once

#include <cstddef>
#include <bitset>

template<size_t length> class BoolArray {

public:
    static constexpr size_t computeArrayLength() {
        return (length / 32) + (length % 32 > 0 ? 1 : 0);
    }

private:
    unsigned int arrayContents[computeArrayLength()];

public:
    class BoolReference {
    private:
        unsigned int* referenceToBitVector;
        unsigned char bitOffset;

        unsigned int genBitMask(unsigned char offset) {
            return (0x1U << (31U - offset));
        }
    public:
        BoolReference();

        BoolReference(unsigned int* reference, unsigned char offset)
            : referenceToBitVector(reference), bitOffset(offset) {}

        // array[i] = x;
        BoolReference& operator= (bool value)
        {
            if (value) {
                *referenceToBitVector |= genBitMask(bitOffset);
            } else {
                *referenceToBitVector &= ~genBitMask(bitOffset);
            }
            return *this;
        }

        // array[i] = array[j];
        BoolReference& operator= (const BoolReference& j)
        {
            if ((*(j.referenceToBitVector) & genBitMask(j.bitOffset))) {
                *referenceToBitVector |= genBitMask(j.bitOffset);
            } else {
                *referenceToBitVector &= ~genBitMask(bitOffset);
            }
            return *this;
        }

        // x = array[i];
        operator bool()
        {
            return (*(referenceToBitVector) & genBitMask(bitOffset)) != 0;
        }
    };

    BoolReference &operator[](size_t index) {
        BoolReference ref = BoolReference(arrayContents + (index / 32), index % 32);
        return ref;
    }


    BoolArray(bool initialValue) {
        for(size_t i = 0; i < computeArrayLength(); i++) {
            arrayContents[i] = initialValue ? 0xFFFFFFFF : 0x00000000;
        }
    }

    const unsigned int* data() const {
        return arrayContents;
    }

    BoolArray() {}
};