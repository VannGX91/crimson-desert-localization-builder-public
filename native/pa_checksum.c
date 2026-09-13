#define WIN32_LEAN_AND_MEAN
#include <errno.h>
#include <stdint.h>
#include <stdio.h>
#include <wchar.h>

#define PA_MAGIC UINT32_C(0x2145E233)
#define BLOCKS_PER_CHUNK 4096

#if defined(_WIN32)
#define PA_EXPORT __declspec(dllexport)
#else
#define PA_EXPORT __attribute__((visibility("default")))
#endif

static uint32_t rotate_left(uint32_t value, unsigned count) {
    return (value << count) | (value >> (32U - count));
}

static uint32_t rotate_right(uint32_t value, unsigned count) {
    return (value >> count) | (value << (32U - count));
}

static uint32_t read_u32_le(const unsigned char *data) {
    return ((uint32_t)data[0]) |
           ((uint32_t)data[1] << 8U) |
           ((uint32_t)data[2] << 16U) |
           ((uint32_t)data[3] << 24U);
}

static void mix_block(
    uint32_t *a,
    uint32_t *b,
    uint32_t *c,
    const unsigned char *block
) {
    *a += read_u32_le(block);
    *b += read_u32_le(block + 4);
    *c += read_u32_le(block + 8);

    *a -= *c; *a ^= rotate_left(*c, 4);  *c += *b;
    *b -= *a; *b ^= rotate_left(*a, 6);  *a += *c;
    *c -= *b; *c ^= rotate_left(*b, 8);  *b += *a;
    *a -= *c; *a ^= rotate_left(*c, 16); *c += *b;
    *b -= *a; *b ^= rotate_left(*a, 19); *a += *c;
    *c -= *b; *c ^= rotate_left(*b, 4);  *b += *a;
}

static uint32_t finish_checksum(
    uint32_t a,
    uint32_t b,
    uint32_t c,
    const unsigned char *tail,
    size_t tail_size
) {
    size_t index;
    for (index = 0; index < tail_size; ++index) {
        if (index < 4) {
            a += (uint32_t)tail[index] << (8U * (unsigned)index);
        } else if (index < 8) {
            b += (uint32_t)tail[index] << (8U * (unsigned)(index - 4));
        } else {
            c += (uint32_t)tail[index] << (8U * (unsigned)(index - 8));
        }
    }

    uint32_t value_82 = (b ^ c) - rotate_left(b, 14);
    uint32_t value_83 = (a ^ value_82) - rotate_left(value_82, 11);
    uint32_t value_84 = (value_83 ^ b) - rotate_right(value_83, 7);
    uint32_t value_85 = (value_84 ^ value_82) - rotate_left(value_84, 16);
    uint32_t mixed = (value_83 ^ value_85) - rotate_left(value_85, 4);
    uint32_t value_87 = (mixed ^ value_84) - rotate_left(mixed, 14);
    return (value_87 ^ value_85) - rotate_right(value_87, 8);
}

PA_EXPORT uint32_t pa_checksum_file_w(
    const wchar_t *path,
    int *error_code
) {
    unsigned char buffer[BLOCKS_PER_CHUNK * 12];
    unsigned char tail[12];
    FILE *stream;
    __int64 signed_length;
    uint64_t length;
    uint64_t block_count;
    uint64_t processed = 0;
    uint32_t a;
    uint32_t b;
    uint32_t c;

    if (error_code != NULL) {
        *error_code = 0;
    }
    if (path == NULL) {
        if (error_code != NULL) {
            *error_code = EINVAL;
        }
        return 0;
    }

    stream = _wfopen(path, L"rb");
    if (stream == NULL) {
        if (error_code != NULL) {
            *error_code = errno != 0 ? errno : EIO;
        }
        return 0;
    }
    if (_fseeki64(stream, 0, SEEK_END) != 0) {
        if (error_code != NULL) {
            *error_code = errno != 0 ? errno : EIO;
        }
        fclose(stream);
        return 0;
    }
    signed_length = _ftelli64(stream);
    if (signed_length < 0) {
        if (error_code != NULL) {
            *error_code = errno != 0 ? errno : EIO;
        }
        fclose(stream);
        return 0;
    }
    length = (uint64_t)signed_length;
    if (_fseeki64(stream, 0, SEEK_SET) != 0) {
        if (error_code != NULL) {
            *error_code = errno != 0 ? errno : EIO;
        }
        fclose(stream);
        return 0;
    }
    if (length == 0) {
        fclose(stream);
        return 0;
    }

    a = b = c = (uint32_t)(length - PA_MAGIC);
    block_count = (length - 1U) / 12U;
    while (block_count > 0) {
        uint64_t current_blocks = block_count < BLOCKS_PER_CHUNK
            ? block_count
            : BLOCKS_PER_CHUNK;
        size_t chunk_size = (size_t)(current_blocks * 12U);
        size_t offset;
        if (fread(buffer, 1, chunk_size, stream) != chunk_size) {
            if (error_code != NULL) {
                *error_code = ferror(stream) && errno != 0 ? errno : EIO;
            }
            fclose(stream);
            return 0;
        }
        for (offset = 0; offset < chunk_size; offset += 12) {
            mix_block(&a, &b, &c, buffer + offset);
        }
        processed += chunk_size;
        block_count -= current_blocks;
    }

    size_t tail_size = (size_t)(length - processed);
    if (fread(tail, 1, tail_size, stream) != tail_size) {
        if (error_code != NULL) {
            *error_code = ferror(stream) && errno != 0 ? errno : EIO;
        }
        fclose(stream);
        return 0;
    }
    fclose(stream);
    return finish_checksum(a, b, c, tail, tail_size);
}
