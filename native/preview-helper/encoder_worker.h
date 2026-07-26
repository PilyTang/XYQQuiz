#pragma once

#include <cstdint>

struct EncoderWorkerArguments {
    unsigned adapter = 0;
    unsigned width = 0;
    unsigned height = 0;
    unsigned frame_rate = 30;
};

constexpr std::uint32_t kEncoderWorkerMagic = 0x51455748U;

#pragma pack(push, 1)
struct EncoderWorkerInputHeader {
    std::uint32_t magic;
    std::uint32_t payload_size;
    std::uint64_t frame_id;
    std::int64_t timestamp_100ns;
    std::uint8_t force_key_frame;
    std::uint8_t reserved[7];
};

struct EncoderWorkerOutputHeader {
    std::uint32_t magic;
    std::uint32_t payload_size;
    std::uint64_t frame_id;
    std::int64_t timestamp_100ns;
    std::uint8_t key_frame;
    std::uint8_t reserved[7];
};
#pragma pack(pop)

static_assert(sizeof(EncoderWorkerInputHeader) == 32);
static_assert(sizeof(EncoderWorkerOutputHeader) == 32);

int RunEncoderWorker(const EncoderWorkerArguments& arguments);
