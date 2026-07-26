#include "server.h"
#include "encoder_worker.h"

#include <windows.h>

#include <codecapi.h>
#include <d3d11.h>
#include <d3d11_1.h>
#include <dxgi1_2.h>
#include <mfapi.h>
#include <mferror.h>
#include <mfidl.h>
#include <mftransform.h>
#include <icodecapi.h>
#include <windows.graphics.capture.interop.h>
#include <windows.graphics.directx.direct3d11.interop.h>
#include <wrl/client.h>

#include <winrt/Windows.Foundation.h>
#include <winrt/Windows.Graphics.Capture.h>
#include <winrt/Windows.Graphics.DirectX.Direct3D11.h>
#include <winrt/Windows.Graphics.DirectX.h>
#include <winrt/base.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <cwctype>
#include <cstring>
#include <exception>
#include <memory>
#include <mutex>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <utility>
#include <vector>

#include "preview_ps.h"
#include "preview_vs.h"

using Microsoft::WRL::ComPtr;
using namespace winrt::Windows::Graphics;
using namespace winrt::Windows::Graphics::Capture;
using namespace winrt::Windows::Graphics::DirectX;
using namespace winrt::Windows::Graphics::DirectX::Direct3D11;

struct __declspec(uuid("A9B3D012-3DF2-4EE3-B8D1-8695F457D3C1"))
IDirect3DDxgiInterfaceAccess : IUnknown {
    virtual HRESULT STDMETHODCALLTYPE GetInterface(REFIID iid, void** object) = 0;
};

namespace {

constexpr std::uint32_t kPipeMagic = 0x51595848U; // "HXYQ" in LE.
constexpr std::uint16_t kProtocolVersion = 1;
constexpr std::uint32_t kMappingMagic = 0x5146524DU; // "MRFQ" in LE.
constexpr std::uint32_t kMappingVersion = 1;
constexpr std::size_t kMaximumPipePayload = 16U * 1024U * 1024U;

enum class MessageType : std::uint16_t {
    hello = 1,
    ready = 2,
    video = 3,
    error = 4,
    force_key_frame = 5,
    stop = 6,
    debug = 7,
    preview_layout = 8,
    preview_overlay = 9,
};

#pragma pack(push, 1)
struct PipeHeader {
    std::uint32_t magic;
    std::uint16_t version;
    std::uint16_t type;
    std::uint32_t payload_size;
};

struct VideoHeader {
    std::uint64_t frame_id;
    std::int64_t timestamp_100ns;
    std::uint32_t width;
    std::uint32_t height;
    std::uint8_t key_frame;
    std::uint8_t reserved[7];
};

struct PreviewLayoutMessage {
    std::uint64_t owner_hwnd;
    std::int32_t x;
    std::int32_t y;
    std::int32_t width;
    std::int32_t height;
    float scale;
    std::uint32_t visible;
};

struct PreviewOverlayMessage {
    float x;
    float y;
    float width;
    float height;
    float score;
    std::uint32_t level;
};
#pragma pack(pop)

static_assert(sizeof(PipeHeader) == 12);
static_assert(sizeof(VideoHeader) == 32);
static_assert(sizeof(PreviewLayoutMessage) == 32);
static_assert(sizeof(PreviewOverlayMessage) == 24);

struct alignas(64) SharedFrameSlot {
    std::uint64_t frame_id;
    std::int64_t captured_qpc;
    std::uint32_t width;
    std::uint32_t height;
    std::uint32_t stride;
    std::uint32_t payload_size;
    std::array<std::byte, 32> reserved{};
};

struct alignas(64) SharedFrameHeader {
    std::uint32_t magic;
    std::uint32_t version;
    std::uint32_t header_size;
    std::uint32_t slot_header_size;
    std::uint32_t slot_capacity;
    volatile LONG active_slot;
    std::array<std::byte, 40> reserved{};
};

static_assert(sizeof(SharedFrameSlot) == 64);
static_assert(sizeof(SharedFrameHeader) == 64);

class HResultError final : public std::runtime_error {
public:
    HResultError(std::string message, HRESULT result)
        : std::runtime_error(std::move(message)), result_(result) {}

    [[nodiscard]] HRESULT result() const noexcept { return result_; }

private:
    HRESULT result_;
};

void Check(HRESULT result, std::string_view operation) {
    if (FAILED(result)) {
        char buffer[160]{};
        const int count = _snprintf_s(
            buffer,
            sizeof(buffer),
            _TRUNCATE,
            "%.*s failed with HRESULT 0x%08lX",
            static_cast<int>(operation.size()),
            operation.data(),
            static_cast<unsigned long>(result));
        throw HResultError(
            count > 0 ? std::string(buffer, static_cast<std::size_t>(count))
                      : std::string(operation),
            result);
    }
}

class UniqueHandle final {
public:
    UniqueHandle() = default;
    explicit UniqueHandle(HANDLE value) : value_(value) {}
    ~UniqueHandle() { reset(); }
    UniqueHandle(const UniqueHandle&) = delete;
    UniqueHandle& operator=(const UniqueHandle&) = delete;
    UniqueHandle(UniqueHandle&& other) noexcept : value_(other.release()) {}
    UniqueHandle& operator=(UniqueHandle&& other) noexcept {
        if (this != &other) {
            reset(other.release());
        }
        return *this;
    }
    [[nodiscard]] HANDLE get() const noexcept { return value_; }
    [[nodiscard]] explicit operator bool() const noexcept {
        return value_ != nullptr && value_ != INVALID_HANDLE_VALUE;
    }
    HANDLE release() noexcept {
        const HANDLE value = value_;
        value_ = nullptr;
        return value;
    }
    void reset(HANDLE value = nullptr) noexcept {
        if (*this) {
            CloseHandle(value_);
        }
        value_ = value;
    }

private:
    HANDLE value_ = nullptr;
};

class MappedView final {
public:
    explicit MappedView(void* value = nullptr) : value_(value) {}
    ~MappedView() {
        if (value_ != nullptr) {
            UnmapViewOfFile(value_);
        }
    }
    MappedView(const MappedView&) = delete;
    MappedView& operator=(const MappedView&) = delete;
    [[nodiscard]] void* get() const noexcept { return value_; }

private:
    void* value_;
};

void ReadExact(HANDLE pipe, void* destination, std::size_t size) {
    auto* cursor = static_cast<std::byte*>(destination);
    while (size != 0) {
        DWORD read = 0;
        const DWORD chunk = static_cast<DWORD>((std::min)(
            size,
            static_cast<std::size_t>((std::numeric_limits<DWORD>::max)())));
        if (!ReadFile(pipe, cursor, chunk, &read, nullptr) || read == 0) {
            throw std::runtime_error("preview helper pipe disconnected");
        }
        cursor += read;
        size -= read;
    }
}

void WriteExact(HANDLE pipe, const void* source, std::size_t size) {
    const auto* cursor = static_cast<const std::byte*>(source);
    while (size != 0) {
        DWORD written = 0;
        const DWORD chunk = static_cast<DWORD>((std::min)(
            size,
            static_cast<std::size_t>((std::numeric_limits<DWORD>::max)())));
        if (!WriteFile(pipe, cursor, chunk, &written, nullptr) || written == 0) {
            throw std::runtime_error("preview helper pipe disconnected");
        }
        cursor += written;
        size -= written;
    }
}

struct IncomingMessage {
    MessageType type{};
    std::vector<std::byte> payload;
};

IncomingMessage ReadMessage(HANDLE pipe) {
    PipeHeader header{};
    ReadExact(pipe, &header, sizeof(header));
    if (header.magic != kPipeMagic || header.version != kProtocolVersion) {
        throw std::runtime_error("preview helper pipe protocol mismatch");
    }
    if (header.payload_size > kMaximumPipePayload) {
        throw std::runtime_error("preview helper pipe payload is too large");
    }
    IncomingMessage result;
    result.type = static_cast<MessageType>(header.type);
    result.payload.resize(header.payload_size);
    if (!result.payload.empty()) {
        ReadExact(pipe, result.payload.data(), result.payload.size());
    }
    return result;
}

class PipeWriter final {
public:
    explicit PipeWriter(HANDLE pipe) : pipe_(pipe) {}

    void Send(MessageType type, const void* payload, std::size_t size) {
        if (size > (std::numeric_limits<std::uint32_t>::max)()) {
            throw std::invalid_argument("preview helper message is too large");
        }
        const PipeHeader header{
            kPipeMagic,
            kProtocolVersion,
            static_cast<std::uint16_t>(type),
            static_cast<std::uint32_t>(size),
        };
        std::scoped_lock lock(mutex_);
        WriteExact(pipe_, &header, sizeof(header));
        if (size != 0) {
            WriteExact(pipe_, payload, size);
        }
    }

    void SendText(MessageType type, std::string_view value) {
        Send(type, value.data(), value.size());
    }

private:
    HANDLE pipe_;
    std::mutex mutex_;
};

class SharedFrameWriter final {
public:
    SharedFrameWriter(std::wstring_view name, std::uint32_t slot_capacity)
        : capacity_(slot_capacity) {
        if (capacity_ == 0) {
            throw std::invalid_argument("shared frame capacity must be positive");
        }
        const std::uint64_t total = sizeof(SharedFrameHeader)
            + 2ULL * (sizeof(SharedFrameSlot) + capacity_);
        if (total > (std::numeric_limits<SIZE_T>::max)()) {
            throw std::invalid_argument("shared frame mapping is too large");
        }
        mapping_.reset(CreateFileMappingW(
            INVALID_HANDLE_VALUE,
            nullptr,
            PAGE_READWRITE,
            static_cast<DWORD>(total >> 32U),
            static_cast<DWORD>(total & 0xFFFFFFFFU),
            std::wstring(name).c_str()));
        if (!mapping_) {
            Check(HRESULT_FROM_WIN32(GetLastError()), "CreateFileMappingW");
        }
        view_ = std::make_unique<MappedView>(MapViewOfFile(
            mapping_.get(),
            FILE_MAP_ALL_ACCESS,
            0,
            0,
            static_cast<SIZE_T>(total)));
        if (view_->get() == nullptr) {
            Check(HRESULT_FROM_WIN32(GetLastError()), "MapViewOfFile");
        }
        base_ = static_cast<std::byte*>(view_->get());
        std::memset(base_, 0, static_cast<std::size_t>(total));
        auto* header = reinterpret_cast<SharedFrameHeader*>(base_);
        header->magic = kMappingMagic;
        header->version = kMappingVersion;
        header->header_size = sizeof(SharedFrameHeader);
        header->slot_header_size = sizeof(SharedFrameSlot);
        header->slot_capacity = capacity_;
        header->active_slot = -1;
    }

    void Publish(
        std::uint64_t frame_id,
        std::int64_t captured_qpc,
        std::uint32_t width,
        std::uint32_t height,
        std::uint32_t stride,
        const std::byte* pixels,
        std::uint32_t payload_size) {
        if (payload_size > capacity_) {
            throw std::runtime_error("captured frame exceeds shared memory capacity");
        }
        auto* header = reinterpret_cast<SharedFrameHeader*>(base_);
        const LONG current = InterlockedCompareExchange(&header->active_slot, -1, -1);
        const LONG next = current == 0 ? 1 : 0;
        auto* slot = Slot(next);
        std::byte* destination = reinterpret_cast<std::byte*>(slot) + sizeof(*slot);
        std::memcpy(destination, pixels, payload_size);
        slot->frame_id = frame_id;
        slot->captured_qpc = captured_qpc;
        slot->width = width;
        slot->height = height;
        slot->stride = stride;
        slot->payload_size = payload_size;
        MemoryBarrier();
        InterlockedExchange(&header->active_slot, next);
    }

private:
    SharedFrameSlot* Slot(LONG index) const noexcept {
        const std::size_t offset = sizeof(SharedFrameHeader)
            + static_cast<std::size_t>(index) * (sizeof(SharedFrameSlot) + capacity_);
        return reinterpret_cast<SharedFrameSlot*>(base_ + offset);
    }

    std::uint32_t capacity_;
    UniqueHandle mapping_;
    std::unique_ptr<MappedView> view_;
    std::byte* base_ = nullptr;
};

ComPtr<IMFMediaType> CreateVideoType(
    const GUID& subtype,
    UINT32 width,
    UINT32 height,
    UINT32 frame_rate,
    UINT32 bitrate = 0) {
    ComPtr<IMFMediaType> type;
    Check(MFCreateMediaType(&type), "MFCreateMediaType");
    Check(type->SetGUID(MF_MT_MAJOR_TYPE, MFMediaType_Video), "MF_MT_MAJOR_TYPE");
    Check(type->SetGUID(MF_MT_SUBTYPE, subtype), "MF_MT_SUBTYPE");
    Check(MFSetAttributeSize(type.Get(), MF_MT_FRAME_SIZE, width, height), "MF_MT_FRAME_SIZE");
    Check(MFSetAttributeRatio(type.Get(), MF_MT_FRAME_RATE, frame_rate, 1), "MF_MT_FRAME_RATE");
    Check(MFSetAttributeRatio(type.Get(), MF_MT_PIXEL_ASPECT_RATIO, 1, 1), "MF_MT_PIXEL_ASPECT_RATIO");
    Check(type->SetUINT32(MF_MT_INTERLACE_MODE, MFVideoInterlace_Progressive), "MF_MT_INTERLACE_MODE");
    if (bitrate != 0) {
        Check(type->SetUINT32(MF_MT_AVG_BITRATE, bitrate), "MF_MT_AVG_BITRATE");
        Check(type->SetUINT32(MF_MT_MPEG2_PROFILE, eAVEncH264VProfile_Base), "MF_MT_MPEG2_PROFILE");
    }
    return type;
}

struct EncoderOutput {
    std::vector<std::byte> bytes;
    std::int64_t timestamp_100ns = 0;
    bool key_frame = false;
};

EncoderOutput ReadEncoderOutput(IMFTransform* encoder) {
    MFT_OUTPUT_STREAM_INFO stream_info{};
    Check(encoder->GetOutputStreamInfo(0, &stream_info), "GetOutputStreamInfo");
    ComPtr<IMFSample> owned_sample;
    MFT_OUTPUT_DATA_BUFFER output{};
    output.dwStreamID = 0;
    if ((stream_info.dwFlags & MFT_OUTPUT_STREAM_PROVIDES_SAMPLES) == 0) {
        Check(MFCreateSample(&owned_sample), "MFCreateSample(output)");
        ComPtr<IMFMediaBuffer> buffer;
        Check(MFCreateMemoryBuffer(
            (std::max)(stream_info.cbSize, static_cast<DWORD>(2U * 1024U * 1024U)),
            &buffer), "MFCreateMemoryBuffer(output)");
        Check(owned_sample->AddBuffer(buffer.Get()), "AddBuffer(output)");
        output.pSample = owned_sample.Get();
    }
    DWORD status = 0;
    const HRESULT processed = encoder->ProcessOutput(0, 1, &output, &status);
    if (output.pEvents != nullptr) {
        output.pEvents->Release();
    }
    if (processed == MF_E_TRANSFORM_NEED_MORE_INPUT) {
        return {};
    }
    Check(processed, "ProcessOutput");
    ComPtr<IMFSample> provided_sample;
    IMFSample* sample = output.pSample;
    if (!owned_sample && sample != nullptr) {
        provided_sample.Attach(sample);
    }
    if (sample == nullptr) {
        return {};
    }
    EncoderOutput result;
    LONGLONG timestamp = 0;
    if (SUCCEEDED(sample->GetSampleTime(&timestamp))) {
        result.timestamp_100ns = timestamp;
    }
    UINT32 clean_point = FALSE;
    result.key_frame = SUCCEEDED(sample->GetUINT32(MFSampleExtension_CleanPoint, &clean_point))
        && clean_point != FALSE;
    ComPtr<IMFMediaBuffer> contiguous;
    Check(sample->ConvertToContiguousBuffer(&contiguous), "ConvertToContiguousBuffer");
    BYTE* data = nullptr;
    DWORD length = 0;
    Check(contiguous->Lock(&data, nullptr, &length), "IMFMediaBuffer::Lock");
    try {
        result.bytes.resize(length);
        if (length != 0) {
            std::memcpy(result.bytes.data(), data, length);
        }
    } catch (...) {
        contiguous->Unlock();
        throw;
    }
    Check(contiguous->Unlock(), "IMFMediaBuffer::Unlock");
    return result;
}

class HardwareEncoder final {
public:
    HardwareEncoder(
        ID3D11Device* capture_device,
        ID3D11DeviceContext* capture_context,
        UINT32 width,
        UINT32 height,
        UINT32 frame_rate,
        PipeWriter& writer)
        : width_(width),
          height_(height),
          frame_rate_(frame_rate),
          writer_(writer) {
        capture_device_ = capture_device;
        capture_context_ = capture_context;
        ComPtr<IDXGIDevice> dxgi_device;
        Check(capture_device->QueryInterface(IID_PPV_ARGS(&dxgi_device)), "IDXGIDevice(capture)");
        ComPtr<IDXGIAdapter> adapter;
        Check(dxgi_device->GetAdapter(&adapter), "IDXGIDevice::GetAdapter");
        DXGI_ADAPTER_DESC adapter_description{};
        Check(adapter->GetDesc(&adapter_description), "IDXGIAdapter::GetDesc");
        vendor_id_ = adapter_description.VendorId;
        ActivateEncoder();
        Configure();
        ComPtr<IMFMediaEvent> initial_event;
        Check(event_generator_->GetEvent(0, &initial_event), "GetEvent(initial input)");
        MediaEventType initial_type{};
        Check(initial_event->GetType(&initial_type), "GetType(initial input)");
        if (initial_type != METransformNeedInput) {
            throw std::runtime_error("H.264 encoder did not request initial input");
        }
        input_requests_ = 1;
        need_input_reported_ = true;
        writer_.SendText(MessageType::debug, "encoder_need_input");
        writer_.SendText(MessageType::debug, "encoder_configured");
    }

    ~HardwareEncoder() {
        if (encoder_) {
            encoder_->ProcessMessage(MFT_MESSAGE_NOTIFY_END_OF_STREAM, 0);
            encoder_->ProcessMessage(MFT_MESSAGE_NOTIFY_END_STREAMING, 0);
        }
        if (activation_) {
            activation_->ShutdownObject();
        }
    }

    void Submit(ID3D11Texture2D* texture, std::uint64_t frame_id, std::int64_t timestamp_100ns) {
        if (!submit_called_reported_) {
            writer_.SendText(MessageType::debug, "encoder_submit_called");
            submit_called_reported_ = true;
        }
        D3D11_TEXTURE2D_DESC description{};
        texture->GetDesc(&description);
        D3D11_TEXTURE2D_DESC staging_description = description;
        staging_description.BindFlags = 0;
        staging_description.MiscFlags = 0;
        staging_description.Usage = D3D11_USAGE_STAGING;
        staging_description.CPUAccessFlags = D3D11_CPU_ACCESS_READ;
        ComPtr<ID3D11Texture2D> staging;
        Check(
            capture_device_->CreateTexture2D(&staging_description, nullptr, &staging),
            "CreateTexture2D(NV12 staging)");
        capture_context_->CopyResource(staging.Get(), texture);
        D3D11_MAPPED_SUBRESOURCE mapped{};
        Check(
            capture_context_->Map(staging.Get(), 0, D3D11_MAP_READ, 0, &mapped),
            "Map(NV12 staging)");
        std::vector<std::byte> packed(
            static_cast<std::size_t>(description.Width) * description.Height * 3U / 2U);
        const auto* mapped_bytes = static_cast<const std::byte*>(mapped.pData);
        for (UINT32 row = 0; row < description.Height * 3U / 2U; ++row) {
            std::memcpy(
                packed.data() + static_cast<std::size_t>(row) * description.Width,
                mapped_bytes + static_cast<std::size_t>(row) * mapped.RowPitch,
                description.Width);
        }
        capture_context_->Unmap(staging.Get(), 0);
        {
            std::scoped_lock lock(queue_mutex_);
            pending_input_ = PendingInput{std::move(packed), frame_id, timestamp_100ns};
        }
    }

    void RequestKeyFrame() noexcept { force_key_frame_.store(true); }

    void PumpOnce() {
        if (!streaming_started_) {
            return;
        }
        if (input_requests_ == 0) {
            {
                std::scoped_lock lock(queue_mutex_);
                if (!pending_input_) return;
            }
            DrainEvents(last_submitted_frame_id_);
        }
        std::optional<PendingInput> input;
        {
            std::scoped_lock lock(queue_mutex_);
            if (pending_input_) {
                input = std::move(pending_input_);
                pending_input_.reset();
            }
        }
        if (!input) return;
        if (force_key_frame_.exchange(false)) {
            SetCodecBoolean(CODECAPI_AVEncVideoForceKeyFrame, true);
        }
        ComPtr<IMFMediaBuffer> buffer;
        Check(
            MFCreateMemoryBuffer(static_cast<DWORD>(input->bytes.size()), &buffer),
            "MFCreateMemoryBuffer(input)");
        BYTE* destination = nullptr;
        DWORD capacity = 0;
        Check(buffer->Lock(&destination, &capacity, nullptr), "IMFMediaBuffer::Lock(input)");
        if (capacity < input->bytes.size()) {
            buffer->Unlock();
            throw std::runtime_error("H.264 input buffer is unexpectedly small");
        }
        std::memcpy(destination, input->bytes.data(), input->bytes.size());
        Check(buffer->Unlock(), "IMFMediaBuffer::Unlock(input)");
        Check(
            buffer->SetCurrentLength(static_cast<DWORD>(input->bytes.size())),
            "SetCurrentLength(input)");
        ComPtr<IMFSample> sample;
        Check(MFCreateSample(&sample), "MFCreateSample(input)");
        Check(sample->AddBuffer(buffer.Get()), "AddBuffer(input)");
        Check(sample->SetSampleTime(input->timestamp_100ns), "SetSampleTime");
        Check(sample->SetSampleDuration(10'000'000LL / frame_rate_), "SetSampleDuration");
        const HRESULT input_result = encoder_->ProcessInput(0, sample.Get(), 0);
        if (input_result == MF_E_NOTACCEPTING) {
            std::scoped_lock lock(queue_mutex_);
            if (!pending_input_) pending_input_ = std::move(input);
            return;
        }
        Check(input_result, "ProcessInput");
        if (!submitted_reported_) {
            writer_.SendText(MessageType::debug, "encoder_submitted");
            submitted_reported_ = true;
        }
        if (input_requests_ != 0) --input_requests_;
        last_submitted_frame_id_ = input->frame_id;
        pending_frame_ids_.push_back(input->frame_id);
        DrainEvents(input->frame_id);
    }

private:
    struct PendingInput {
        std::vector<std::byte> bytes;
        std::uint64_t frame_id;
        std::int64_t timestamp_100ns;
    };

    void ActivateEncoder() {
        MFT_REGISTER_TYPE_INFO output_type{MFMediaType_Video, MFVideoFormat_H264};
        IMFActivate** raw = nullptr;
        UINT32 count = 0;
        Check(MFTEnumEx(
            MFT_CATEGORY_VIDEO_ENCODER,
            MFT_ENUM_FLAG_HARDWARE | MFT_ENUM_FLAG_SORTANDFILTER,
            nullptr,
            &output_type,
            &raw,
            &count), "MFTEnumEx(H.264 hardware encoder)");
        struct Candidate {
            ComPtr<IMFActivate> activation;
            std::wstring name;
        };
        std::vector<Candidate> candidates;
        candidates.reserve(count);
        for (UINT32 index = 0; index < count; ++index) {
            ComPtr<IMFActivate> activation(raw[index]);
            WCHAR* friendly_name = nullptr;
            UINT32 friendly_name_length = 0;
            std::wstring name;
            if (SUCCEEDED(activation->GetAllocatedString(
                    MFT_FRIENDLY_NAME_Attribute,
                    &friendly_name,
                    &friendly_name_length))) {
                name.assign(friendly_name, friendly_name_length);
                CoTaskMemFree(friendly_name);
            }
            candidates.push_back({std::move(activation), std::move(name)});
        }
        CoTaskMemFree(raw);
        const auto preferred = [this](const Candidate& candidate) {
            std::wstring name = candidate.name;
            std::transform(name.begin(), name.end(), name.begin(), towupper);
            if (vendor_id_ == 0x10DEU) return name.find(L"NVIDIA") != std::wstring::npos;
            if (vendor_id_ == 0x1002U) return name.find(L"AMD") != std::wstring::npos;
            if (vendor_id_ == 0x8086U) return name.find(L"INTEL") != std::wstring::npos;
            return false;
        };
        std::stable_sort(
            candidates.begin(),
            candidates.end(),
            [&](const Candidate& left, const Candidate& right) {
                return preferred(left) && !preferred(right);
            });
        HRESULT last_error = E_FAIL;
        for (const auto& candidate_entry : candidates) {
            const auto& candidate_activation = candidate_entry.activation;
            ComPtr<IMFTransform> candidate;
            last_error = candidate_activation->ActivateObject(IID_PPV_ARGS(&candidate));
            if (FAILED(last_error)) {
                continue;
            }
            ComPtr<IMFAttributes> attributes;
            last_error = candidate->GetAttributes(&attributes);
            if (SUCCEEDED(last_error)) {
                UINT32 asynchronous = FALSE;
                if (SUCCEEDED(attributes->GetUINT32(MF_TRANSFORM_ASYNC, &asynchronous))
                    && asynchronous != FALSE) {
                    last_error = attributes->SetUINT32(MF_TRANSFORM_ASYNC_UNLOCK, TRUE);
                }
            }
            if (SUCCEEDED(last_error)) {
                encoder_ = std::move(candidate);
                activation_ = candidate_activation;
                break;
            }
            candidate_activation->ShutdownObject();
        }
        if (!encoder_) {
            Check(last_error, "No H.264 encoder accepted the selected D3D device");
        }
        Check(encoder_.As(&event_generator_), "IMFMediaEventGenerator");
        encoder_.As(&codec_);
    }

    void Configure() {
        const UINT32 bitrate = (std::max)(2'000'000U, width_ * height_ * 6U);
        const auto output = CreateVideoType(MFVideoFormat_H264, width_, height_, frame_rate_, bitrate);
        const auto input = CreateVideoType(MFVideoFormat_NV12, width_, height_, frame_rate_);
        Check(encoder_->SetOutputType(0, output.Get(), 0), "SetOutputType(H264)");
        Check(encoder_->SetInputType(0, input.Get(), 0), "SetInputType(NV12)");
        SetCodecBoolean(CODECAPI_AVLowLatencyMode, true);
        SetCodecUnsigned(CODECAPI_AVEncMPVDefaultBPictureCount, 0);
        SetCodecUnsigned(CODECAPI_AVEncMPVGOPSize, frame_rate_);
        Check(encoder_->ProcessMessage(MFT_MESSAGE_NOTIFY_BEGIN_STREAMING, 0), "BEGIN_STREAMING");
        Check(encoder_->ProcessMessage(MFT_MESSAGE_NOTIFY_START_OF_STREAM, 0), "START_OF_STREAM");
        streaming_started_ = true;
    }

    void SetCodecBoolean(const GUID& key, bool enabled) noexcept {
        if (!codec_) return;
        VARIANT value;
        VariantInit(&value);
        value.vt = VT_BOOL;
        value.boolVal = enabled ? VARIANT_TRUE : VARIANT_FALSE;
        codec_->SetValue(&key, &value);
        VariantClear(&value);
    }

    void SetCodecUnsigned(const GUID& key, ULONG value_number) noexcept {
        if (!codec_) return;
        VARIANT value;
        VariantInit(&value);
        value.vt = VT_UI4;
        value.ulVal = value_number;
        codec_->SetValue(&key, &value);
        VariantClear(&value);
    }

    void DrainEvents(std::uint64_t fallback_frame_id) {
        while (true) {
            ComPtr<IMFMediaEvent> event;
            const HRESULT event_result = event_generator_->GetEvent(MF_EVENT_FLAG_NO_WAIT, &event);
            if (event_result == MF_E_NO_EVENTS_AVAILABLE) {
                return;
            }
            Check(event_result, "IMFMediaEventGenerator::GetEvent");
            HRESULT status = S_OK;
            Check(event->GetStatus(&status), "IMFMediaEvent::GetStatus");
            Check(status, "asynchronous encoder event");
            MediaEventType type{};
            Check(event->GetType(&type), "IMFMediaEvent::GetType");
            if (type == METransformNeedInput) {
                ++input_requests_;
                if (!need_input_reported_) {
                    writer_.SendText(MessageType::debug, "encoder_need_input");
                    need_input_reported_ = true;
                }
            } else if (type == METransformHaveOutput) {
                EncoderOutput output = ReadEncoderOutput(encoder_.Get());
                if (output.bytes.empty()) {
                    continue;
                }
                const std::uint64_t frame_id = pending_frame_ids_.empty()
                    ? fallback_frame_id
                    : pending_frame_ids_.front();
                if (!pending_frame_ids_.empty()) {
                    pending_frame_ids_.erase(pending_frame_ids_.begin());
                }
                VideoHeader header{
                    frame_id,
                    output.timestamp_100ns,
                    width_,
                    height_,
                    output.key_frame ? static_cast<std::uint8_t>(1) : static_cast<std::uint8_t>(0),
                    {},
                };
                std::vector<std::byte> payload(sizeof(header) + output.bytes.size());
                std::memcpy(payload.data(), &header, sizeof(header));
                std::memcpy(payload.data() + sizeof(header), output.bytes.data(), output.bytes.size());
                writer_.Send(MessageType::video, payload.data(), payload.size());
                if (!output_reported_) {
                    writer_.SendText(MessageType::debug, "encoder_output");
                    output_reported_ = true;
                }
            }
        }
    }

    ComPtr<ID3D11Device> capture_device_;
    ComPtr<ID3D11DeviceContext> capture_context_;
    UINT32 width_;
    UINT32 height_;
    UINT32 frame_rate_;
    PipeWriter& writer_;
    UINT vendor_id_ = 0;
    ComPtr<IMFActivate> activation_;
    ComPtr<IMFTransform> encoder_;
    ComPtr<IMFMediaEventGenerator> event_generator_;
    ComPtr<ICodecAPI> codec_;
    std::atomic_bool force_key_frame_{false};
    std::mutex queue_mutex_;
    std::optional<PendingInput> pending_input_;
    unsigned input_requests_ = 0;
    std::uint64_t last_submitted_frame_id_ = 0;
    bool need_input_reported_ = false;
    bool submitted_reported_ = false;
    bool submit_called_reported_ = false;
    bool output_reported_ = false;
    bool streaming_started_ = false;
    std::vector<std::uint64_t> pending_frame_ids_;
};

class EncoderWorkerClient final {
public:
    EncoderWorkerClient(
        ID3D11Device* capture_device,
        ID3D11DeviceContext* capture_context,
        unsigned adapter,
        UINT32 width,
        UINT32 height,
        UINT32 frame_rate,
        PipeWriter& writer,
        HANDLE stop_event)
        : capture_device_(capture_device),
          capture_context_(capture_context),
          width_(width),
          height_(height),
          frame_rate_(frame_rate),
          writer_(writer),
          stop_event_(stop_event) {
        StartProcess(adapter);
        writer_.SendText(MessageType::debug, "worker_started");
        thread_ = std::thread([this] { Run(); });
    }

    ~EncoderWorkerClient() {
        stopping_.store(true);
        condition_.notify_all();
        if (process_) TerminateProcess(process_.get(), 0);
        input_.reset();
        output_.reset();
        if (thread_.joinable()) thread_.join();
    }

    EncoderWorkerClient(const EncoderWorkerClient&) = delete;
    EncoderWorkerClient& operator=(const EncoderWorkerClient&) = delete;

    void Submit(ID3D11Texture2D* texture, std::uint64_t frame_id, std::int64_t timestamp_100ns) {
        D3D11_TEXTURE2D_DESC description{};
        texture->GetDesc(&description);
        if (description.Width != width_ || description.Height != height_
            || description.Format != DXGI_FORMAT_NV12) {
            throw std::runtime_error("encoder worker received an unexpected texture");
        }
        D3D11_TEXTURE2D_DESC staging_description = description;
        staging_description.BindFlags = 0;
        staging_description.MiscFlags = 0;
        staging_description.Usage = D3D11_USAGE_STAGING;
        staging_description.CPUAccessFlags = D3D11_CPU_ACCESS_READ;
        ComPtr<ID3D11Texture2D> staging;
        Check(capture_device_->CreateTexture2D(
            &staging_description,
            nullptr,
            &staging), "CreateTexture2D(worker staging)");
        capture_context_->CopyResource(staging.Get(), texture);
        D3D11_MAPPED_SUBRESOURCE mapped{};
        Check(capture_context_->Map(
            staging.Get(),
            0,
            D3D11_MAP_READ,
            0,
            &mapped), "Map(worker staging)");
        std::vector<std::byte> bytes(
            static_cast<std::size_t>(width_) * height_ * 3U / 2U);
        const auto* source = static_cast<const std::byte*>(mapped.pData);
        for (UINT32 row = 0; row < height_ * 3U / 2U; ++row) {
            std::memcpy(
                bytes.data() + static_cast<std::size_t>(row) * width_,
                source + static_cast<std::size_t>(row) * mapped.RowPitch,
                width_);
        }
        capture_context_->Unmap(staging.Get(), 0);
        {
            std::scoped_lock lock(mutex_);
            pending_ = Pending{std::move(bytes), frame_id, timestamp_100ns};
        }
        if (!submit_reported_.exchange(true)) {
            writer_.SendText(MessageType::debug, "worker_submit");
        }
        condition_.notify_one();
    }

    void RequestKeyFrame() noexcept { force_key_frame_.store(true); }

private:
    struct Pending {
        std::vector<std::byte> bytes;
        std::uint64_t frame_id;
        std::int64_t timestamp_100ns;
    };

    void StartProcess(unsigned adapter) {
        SECURITY_ATTRIBUTES security{};
        security.nLength = sizeof(security);
        security.bInheritHandle = TRUE;
        HANDLE child_input = nullptr;
        HANDLE parent_input = nullptr;
        HANDLE parent_output = nullptr;
        HANDLE child_output = nullptr;
        if (!CreatePipe(&child_input, &parent_input, &security, 0)
            || !CreatePipe(&parent_output, &child_output, &security, 0)) {
            throw std::runtime_error("failed to create encoder worker pipes");
        }
        UniqueHandle child_input_owner(child_input);
        UniqueHandle parent_input_owner(parent_input);
        UniqueHandle parent_output_owner(parent_output);
        UniqueHandle child_output_owner(child_output);
        if (!SetHandleInformation(parent_input_owner.get(), HANDLE_FLAG_INHERIT, 0)
            || !SetHandleInformation(parent_output_owner.get(), HANDLE_FLAG_INHERIT, 0)) {
            throw std::runtime_error("failed to protect encoder worker pipe handles");
        }
        wchar_t executable[MAX_PATH]{};
        const DWORD length = GetModuleFileNameW(nullptr, executable, MAX_PATH);
        if (length == 0 || length == MAX_PATH) {
            throw std::runtime_error("failed to resolve preview helper executable path");
        }
        std::wstring command = L"\"" + std::wstring(executable) + L"\" --encode-worker"
            + L" --adapter " + std::to_wstring(adapter)
            + L" --width " + std::to_wstring(width_)
            + L" --height " + std::to_wstring(height_)
            + L" --frame-rate " + std::to_wstring(frame_rate_);
        STARTUPINFOW startup{};
        startup.cb = sizeof(startup);
        startup.dwFlags = STARTF_USESTDHANDLES;
        startup.hStdInput = child_input_owner.get();
        startup.hStdOutput = child_output_owner.get();
        startup.hStdError = GetStdHandle(STD_ERROR_HANDLE);
        PROCESS_INFORMATION process{};
        if (!CreateProcessW(
                executable,
                command.data(),
                nullptr,
                nullptr,
                TRUE,
                CREATE_NO_WINDOW,
                nullptr,
                nullptr,
                &startup,
                &process)) {
            throw std::runtime_error("failed to start encoder worker process");
        }
        process_.reset(process.hProcess);
        CloseHandle(process.hThread);
        input_ = std::move(parent_input_owner);
        output_ = std::move(parent_output_owner);
    }

    void Run() noexcept {
        try {
            while (!stopping_.load()) {
                std::optional<Pending> pending;
                {
                    std::unique_lock lock(mutex_);
                    condition_.wait(lock, [this] {
                        return stopping_.load() || pending_.has_value();
                    });
                    if (stopping_.load()) return;
                    pending = std::move(pending_);
                    pending_.reset();
                }
                EncoderWorkerInputHeader header{
                    kEncoderWorkerMagic,
                    static_cast<std::uint32_t>(pending->bytes.size()),
                    pending->frame_id,
                    pending->timestamp_100ns,
                    force_key_frame_.exchange(false) ? static_cast<std::uint8_t>(1)
                                                     : static_cast<std::uint8_t>(0),
                    {},
                };
                WriteExact(input_.get(), &header, sizeof(header));
                WriteExact(input_.get(), pending->bytes.data(), pending->bytes.size());
                if (!write_reported_.exchange(true)) {
                    writer_.SendText(MessageType::debug, "worker_written");
                }
                EncoderWorkerOutputHeader response{};
                ReadExact(output_.get(), &response, sizeof(response));
                if (!read_reported_.exchange(true)) {
                    writer_.SendText(MessageType::debug, "worker_read");
                }
                if (response.magic != kEncoderWorkerMagic
                    || response.payload_size == 0
                    || response.payload_size > kMaximumPipePayload) {
                    throw std::runtime_error("encoder worker output protocol mismatch");
                }
                std::vector<std::byte> encoded(response.payload_size);
                ReadExact(output_.get(), encoded.data(), encoded.size());
                VideoHeader video{
                    response.frame_id,
                    response.timestamp_100ns,
                    width_,
                    height_,
                    response.key_frame,
                    {},
                };
                std::vector<std::byte> payload(sizeof(video) + encoded.size());
                std::memcpy(payload.data(), &video, sizeof(video));
                std::memcpy(payload.data() + sizeof(video), encoded.data(), encoded.size());
                writer_.Send(MessageType::video, payload.data(), payload.size());
            }
        } catch (const std::exception& error) {
            if (!stopping_.load()) {
                try {
                    writer_.SendText(MessageType::error, error.what());
                } catch (...) {
                }
                SetEvent(stop_event_);
            }
        }
    }

    ComPtr<ID3D11Device> capture_device_;
    ComPtr<ID3D11DeviceContext> capture_context_;
    UINT32 width_;
    UINT32 height_;
    UINT32 frame_rate_;
    PipeWriter& writer_;
    HANDLE stop_event_;
    UniqueHandle input_;
    UniqueHandle output_;
    UniqueHandle process_;
    std::thread thread_;
    std::mutex mutex_;
    std::condition_variable condition_;
    std::optional<Pending> pending_;
    std::atomic_bool stopping_{false};
    std::atomic_bool force_key_frame_{true};
    std::atomic_bool submit_reported_{false};
    std::atomic_bool write_reported_{false};
    std::atomic_bool read_reported_{false};
};

class RawPreviewPublisher final {
public:
    RawPreviewPublisher(
        UINT32 width,
        UINT32 height,
        PipeWriter& writer,
        HANDLE stop_event)
        : width_(width),
          height_(height),
          writer_(writer),
          stop_event_(stop_event) {
        thread_ = std::thread([this] { Run(); });
    }

    ~RawPreviewPublisher() {
        stopping_.store(true);
        condition_.notify_all();
        if (thread_.joinable()) thread_.join();
    }

    void Submit(
        std::vector<std::byte> bytes,
        std::uint64_t frame_id,
        std::int64_t timestamp_100ns) {
        std::scoped_lock lock(mutex_);
        // Overwrite queued-but-unsent data so transport congestion never
        // accumulates latency; the newest preview frame always wins.
        pending_ = Pending{std::move(bytes), frame_id, timestamp_100ns};
        condition_.notify_one();
    }

    void RequestKeyFrame() noexcept {}

private:
    struct Pending {
        std::vector<std::byte> bytes;
        std::uint64_t frame_id;
        std::int64_t timestamp_100ns;
    };

    void Run() noexcept {
        try {
            while (!stopping_.load()) {
                std::optional<Pending> pending;
                {
                    std::unique_lock lock(mutex_);
                    condition_.wait(lock, [this] {
                        return stopping_.load() || pending_.has_value();
                    });
                    if (stopping_.load()) return;
                    pending = std::move(pending_);
                    pending_.reset();
                }
                VideoHeader header{
                    pending->frame_id,
                    pending->timestamp_100ns,
                    width_,
                    height_,
                    1,
                    {},
                };
                std::vector<std::byte> payload(sizeof(header) + pending->bytes.size());
                std::memcpy(payload.data(), &header, sizeof(header));
                std::memcpy(
                    payload.data() + sizeof(header),
                    pending->bytes.data(),
                    pending->bytes.size());
                writer_.Send(MessageType::video, payload.data(), payload.size());
                if (!sent_reported_.exchange(true)) {
                    writer_.SendText(MessageType::debug, "publisher_sent");
                }
            }
        } catch (const std::exception& error) {
            if (!stopping_.load()) {
                try {
                    writer_.SendText(MessageType::error, error.what());
                } catch (...) {
                }
                SetEvent(stop_event_);
            }
        }
    }

    UINT32 width_;
    UINT32 height_;
    PipeWriter& writer_;
    HANDLE stop_event_;
    std::thread thread_;
    std::mutex mutex_;
    std::condition_variable condition_;
    std::optional<Pending> pending_;
    std::atomic_bool stopping_{false};
    std::atomic_bool sent_reported_{false};
};

struct PreviewSize {
    UINT32 width;
    UINT32 height;
};

PreviewSize CalculatePreviewSize(UINT32 width, UINT32 height, UINT32 maximum_width) {
    if (width == 0 || height == 0 || maximum_width < 2) {
        throw std::invalid_argument("invalid preview dimensions");
    }
    UINT32 output_width = (std::min)(width, maximum_width);
    output_width &= ~1U;
    UINT32 output_height = static_cast<UINT32>(
        static_cast<std::uint64_t>(height) * output_width / width);
    output_height = (std::max)(2U, output_height & ~1U);
    return {output_width, output_height};
}

class VideoProcessor final {
public:
    VideoProcessor(
        ID3D11Device* device,
        ID3D11DeviceContext* context,
        UINT32 input_width,
        UINT32 input_height,
        PreviewSize output_size,
        UINT32 frame_rate)
        : output_size_(output_size) {
        Check(device->QueryInterface(IID_PPV_ARGS(&video_device_)), "ID3D11VideoDevice");
        Check(context->QueryInterface(IID_PPV_ARGS(&video_context_)), "ID3D11VideoContext");
        D3D11_VIDEO_PROCESSOR_CONTENT_DESC content{};
        content.InputFrameFormat = D3D11_VIDEO_FRAME_FORMAT_PROGRESSIVE;
        content.InputFrameRate = {frame_rate, 1};
        content.InputWidth = input_width;
        content.InputHeight = input_height;
        content.OutputFrameRate = {frame_rate, 1};
        content.OutputWidth = output_size.width;
        content.OutputHeight = output_size.height;
        content.Usage = D3D11_VIDEO_USAGE_PLAYBACK_NORMAL;
        Check(video_device_->CreateVideoProcessorEnumerator(&content, &enumerator_), "CreateVideoProcessorEnumerator");
        Check(video_device_->CreateVideoProcessor(enumerator_.Get(), 0, &processor_), "CreateVideoProcessor");

        D3D11_TEXTURE2D_DESC output_description{};
        output_description.Width = output_size.width;
        output_description.Height = output_size.height;
        output_description.MipLevels = 1;
        output_description.ArraySize = 1;
        output_description.Format = DXGI_FORMAT_NV12;
        output_description.SampleDesc.Count = 1;
        output_description.Usage = D3D11_USAGE_DEFAULT;
        output_description.BindFlags = D3D11_BIND_RENDER_TARGET;
        Check(device->CreateTexture2D(&output_description, nullptr, &output_), "CreateTexture2D(NV12)");

        D3D11_VIDEO_PROCESSOR_OUTPUT_VIEW_DESC output_view_description{};
        output_view_description.ViewDimension = D3D11_VPOV_DIMENSION_TEXTURE2D;
        output_view_description.Texture2D.MipSlice = 0;
        Check(video_device_->CreateVideoProcessorOutputView(
            output_.Get(), enumerator_.Get(), &output_view_description, &output_view_),
            "CreateVideoProcessorOutputView");

        const RECT destination{
            0,
            0,
            static_cast<LONG>(output_size.width),
            static_cast<LONG>(output_size.height),
        };
        video_context_->VideoProcessorSetOutputTargetRect(processor_.Get(), TRUE, &destination);
    }

    ID3D11Texture2D* Convert(ID3D11Texture2D* input, UINT32 width, UINT32 height) {
        D3D11_VIDEO_PROCESSOR_INPUT_VIEW_DESC input_view_description{};
        input_view_description.FourCC = 0;
        input_view_description.ViewDimension = D3D11_VPIV_DIMENSION_TEXTURE2D;
        input_view_description.Texture2D.MipSlice = 0;
        input_view_description.Texture2D.ArraySlice = 0;
        ComPtr<ID3D11VideoProcessorInputView> input_view;
        Check(video_device_->CreateVideoProcessorInputView(
            input, enumerator_.Get(), &input_view_description, &input_view),
            "CreateVideoProcessorInputView");
        const RECT source{0, 0, static_cast<LONG>(width), static_cast<LONG>(height)};
        video_context_->VideoProcessorSetStreamSourceRect(processor_.Get(), 0, TRUE, &source);
        D3D11_VIDEO_PROCESSOR_STREAM stream{};
        stream.Enable = TRUE;
        stream.pInputSurface = input_view.Get();
        const HRESULT converted = video_context_->VideoProcessorBlt(
            processor_.Get(), output_view_.Get(), 0, 1, &stream);
        Check(converted, "VideoProcessorBlt");
        return output_.Get();
    }

    [[nodiscard]] PreviewSize output_size() const noexcept { return output_size_; }
    [[nodiscard]] ID3D11Texture2D* output_texture() const noexcept {
        return output_.Get();
    }

private:
    PreviewSize output_size_;
    ComPtr<ID3D11VideoDevice> video_device_;
    ComPtr<ID3D11VideoContext> video_context_;
    ComPtr<ID3D11VideoProcessorEnumerator> enumerator_;
    ComPtr<ID3D11VideoProcessor> processor_;
    ComPtr<ID3D11Texture2D> output_;
    ComPtr<ID3D11VideoProcessorOutputView> output_view_;
};

class NativePreviewWindow final {
public:
    NativePreviewWindow(
        ID3D11Device* device,
        ID3D11DeviceContext* context,
        PipeWriter& writer,
        unsigned frame_rate)
        : device_(device),
          context_(context),
          writer_(writer),
          frame_rate_((std::max)(1U, frame_rate)) {
        Check(context_->QueryInterface(IID_PPV_ARGS(&context1_)), "ID3D11DeviceContext1");
        window_thread_ = std::thread([this] { WindowLoop(); });
        std::unique_lock lock(window_mutex_);
        window_condition_.wait(lock, [this] { return window_ready_; });
        if (window_error_) {
            std::rethrow_exception(window_error_);
        }
        CreateShaders();
        render_thread_ = std::thread([this] { RenderLoop(); });
    }

    ~NativePreviewWindow() { Stop(); }

    NativePreviewWindow(const NativePreviewWindow&) = delete;
    NativePreviewWindow& operator=(const NativePreviewWindow&) = delete;

    void Stop() noexcept {
        if (stopping_.exchange(true)) return;
        render_condition_.notify_all();
        if (render_thread_.joinable()) render_thread_.join();
        const HWND window = window_.load();
        if (window != nullptr) {
            PostMessageW(window, kStopMessage, 0, 0);
        }
        if (window_thread_.joinable()) window_thread_.join();
        std::scoped_lock render_lock(render_mutex_);
        ResetSwapChain();
        sampler_.Reset();
        pixel_shader_.Reset();
        vertex_shader_.Reset();
    }

    void UpdateLayout(const PreviewLayoutMessage& layout) noexcept {
        layout_owner_.store(layout.owner_hwnd);
        layout_x_.store(layout.x);
        layout_y_.store(layout.y);
        layout_width_.store(layout.width);
        layout_height_.store(layout.height);
        layout_scale_.store(layout.scale);
        layout_visible_.store(layout.visible);
        render_condition_.notify_all();
        const HWND window = window_.load();
        if (window != nullptr) PostMessageW(window, kLayoutMessage, 0, 0);
    }

    void UpdateOverlay(const PreviewOverlayMessage& overlay) noexcept {
        std::scoped_lock lock(overlay_mutex_);
        overlay_ = overlay;
    }

    void Submit(ID3D11Texture2D* texture, UINT32 width, UINT32 height) noexcept {
        {
            std::scoped_lock lock(source_mutex_);
            source_texture_ = texture;
            source_width_ = width;
            source_height_ = height;
            source_changed_ = true;
        }
        render_condition_.notify_one();
    }

private:
    bool Render(ID3D11Texture2D* input, UINT32 input_width, UINT32 input_height) {
        stage_.store(1U);
        const HWND window = window_.load();
        PreviewLayoutMessage layout{};
        layout = SnapshotLayout();
        const HWND owner = reinterpret_cast<HWND>(
            static_cast<std::uintptr_t>(layout.owner_hwnd));
        if (window == nullptr) {
            return false;
        }
        if (layout.visible == 0 || layout.width <= 0 || layout.height <= 0) {
            stage_.store(101U);
            return false;
        }
        if (owner != nullptr && (!IsWindowVisible(owner) || IsIconic(owner))) {
            return false;
        }

        RECT client{};
        if (!GetClientRect(window, &client)) {
            return false;
        }
        const UINT32 width = static_cast<UINT32>((std::max)(0L, client.right - client.left));
        const UINT32 height = static_cast<UINT32>((std::max)(0L, client.bottom - client.top));
        if (width == 0 || height == 0) {
            stage_.store(102U);
            return false;
        }

        stage_.store(2U);
        std::scoped_lock render_lock(render_mutex_);
        stage_.store(3U);
        EnsureSwapChain(window, width, height, input_width, input_height);
        stage_.store(4U);

        D3D11_SHADER_RESOURCE_VIEW_DESC input_description{};
        input_description.Format = DXGI_FORMAT_B8G8R8A8_UNORM;
        input_description.ViewDimension = D3D11_SRV_DIMENSION_TEXTURE2D;
        input_description.Texture2D.MipLevels = 1;
        ComPtr<ID3D11ShaderResourceView> input_view;
        Check(device_->CreateShaderResourceView(
            input,
            &input_description,
            &input_view), "CreateShaderResourceView(native preview)");
        stage_.store(5U);

        const RECT destination = ContainedRect(width, height, input_width, input_height);
        constexpr float black[4] = {0.0F, 0.0F, 0.0F, 1.0F};
        context_->OMSetRenderTargets(1, render_target_.GetAddressOf(), nullptr);
        context_->ClearRenderTargetView(render_target_.Get(), black);
        const D3D11_VIEWPORT viewport{
            static_cast<float>(destination.left),
            static_cast<float>(destination.top),
            static_cast<float>(destination.right - destination.left),
            static_cast<float>(destination.bottom - destination.top),
            0.0F,
            1.0F,
        };
        context_->RSSetViewports(1, &viewport);
        context_->IASetInputLayout(nullptr);
        context_->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
        context_->VSSetShader(vertex_shader_.Get(), nullptr, 0);
        context_->PSSetShader(pixel_shader_.Get(), nullptr, 0);
        context_->PSSetSamplers(0, 1, sampler_.GetAddressOf());
        context_->PSSetShaderResources(0, 1, input_view.GetAddressOf());
        stage_.store(6U);
        context_->Draw(3, 0);
        ID3D11ShaderResourceView* null_view = nullptr;
        context_->PSSetShaderResources(0, 1, &null_view);
        stage_.store(7U);

        DrawOverlay(destination);
        stage_.store(8U);
        const HRESULT presented = swap_chain_->Present(0, 0);
        if (presented == DXGI_STATUS_OCCLUDED) return false;
        Check(presented, "IDXGISwapChain1::Present(native preview)");
        stage_.store(9U);
        first_present_reported_.store(true);
        ReportFps();
        return true;
    }

public:
    [[nodiscard]] unsigned Stage() const noexcept { return stage_.load(); }

    [[nodiscard]] bool HasFirstPresent() const noexcept {
        return first_present_reported_.load();
    }

    bool LatestFps(std::uint64_t& generation, double& fps) const noexcept {
        const std::uint64_t current = fps_generation_.load();
        if (current == 0 || current == generation) return false;
        generation = current;
        fps = latest_fps_.load();
        return true;
    }

    std::string TakeError() {
        std::scoped_lock lock(error_mutex_);
        return std::exchange(render_error_, {});
    }

private:

    static constexpr UINT kLayoutMessage = WM_APP + 41U;
    static constexpr UINT kStopMessage = WM_APP + 42U;
    static constexpr UINT_PTR kPositionTimer = 1U;

    static LRESULT CALLBACK WindowProc(
        HWND window,
        UINT message,
        WPARAM wparam,
        LPARAM lparam) noexcept {
        NativePreviewWindow* self = reinterpret_cast<NativePreviewWindow*>(
            GetWindowLongPtrW(window, GWLP_USERDATA));
        if (message == WM_NCCREATE) {
            const auto* create = reinterpret_cast<const CREATESTRUCTW*>(lparam);
            self = static_cast<NativePreviewWindow*>(create->lpCreateParams);
            SetWindowLongPtrW(window, GWLP_USERDATA, reinterpret_cast<LONG_PTR>(self));
        }
        if (self != nullptr) {
            if (message == kLayoutMessage || message == WM_TIMER) {
                self->ApplyLayout(window);
                return 0;
            }
            if (message == kStopMessage) {
                DestroyWindow(window);
                return 0;
            }
        }
        if (message == WM_NCHITTEST) return HTTRANSPARENT;
        if (message == WM_ERASEBKGND) return 1;
        if (message == WM_DESTROY) {
            PostQuitMessage(0);
            return 0;
        }
        return DefWindowProcW(window, message, wparam, lparam);
    }

    void WindowLoop() noexcept {
        try {
            const HINSTANCE instance = GetModuleHandleW(nullptr);
            const wchar_t* class_name = L"XYQQuizNativePreviewWindow";
            WNDCLASSEXW window_class{};
            window_class.cbSize = sizeof(window_class);
            window_class.hInstance = instance;
            window_class.lpfnWndProc = WindowProc;
            window_class.lpszClassName = class_name;
            window_class.hCursor = LoadCursorW(nullptr, IDC_ARROW);
            window_class.hbrBackground = static_cast<HBRUSH>(GetStockObject(BLACK_BRUSH));
            if (RegisterClassExW(&window_class) == 0 && GetLastError() != ERROR_CLASS_ALREADY_EXISTS) {
                throw std::runtime_error("RegisterClassExW(native preview) failed");
            }
            const HWND window = CreateWindowExW(
                WS_EX_NOACTIVATE | WS_EX_TOOLWINDOW | WS_EX_TRANSPARENT,
                class_name,
                L"XYQQuiz Native Preview",
                WS_POPUP | WS_CLIPCHILDREN | WS_CLIPSIBLINGS,
                0,
                0,
                1,
                1,
                nullptr,
                nullptr,
                instance,
                this);
            if (window == nullptr) {
                throw std::runtime_error("CreateWindowExW(native preview) failed");
            }
            {
                std::scoped_lock lock(window_mutex_);
                window_.store(window);
                window_ready_ = true;
            }
            window_condition_.notify_all();
            SetTimer(window, kPositionTimer, 100U, nullptr);
            MSG message{};
            while (GetMessageW(&message, nullptr, 0, 0) > 0) {
                TranslateMessage(&message);
                DispatchMessageW(&message);
            }
            KillTimer(window, kPositionTimer);
            window_.store(nullptr);
        } catch (...) {
            {
                std::scoped_lock lock(window_mutex_);
                window_error_ = std::current_exception();
                window_ready_ = true;
            }
            window_condition_.notify_all();
        }
    }

    void RenderLoop() noexcept {
        try {
            ComPtr<ID3D11Texture2D> current;
            UINT32 current_width = 0;
            UINT32 current_height = 0;
            auto next_frame = std::chrono::steady_clock::now();
            const auto interval = std::chrono::microseconds(1'000'000 / frame_rate_);
            while (!stopping_.load()) {
                if (!current) {
                    std::unique_lock lock(source_mutex_);
                    render_condition_.wait_for(lock, std::chrono::milliseconds(100), [this] {
                        return stopping_.load() || source_changed_;
                    });
                    if (stopping_.load()) return;
                    if (source_changed_) {
                        current = source_texture_;
                        current_width = source_width_;
                        current_height = source_height_;
                        source_changed_ = false;
                    }
                } else {
                    std::scoped_lock lock(source_mutex_);
                    if (source_changed_) {
                        current = source_texture_;
                        current_width = source_width_;
                        current_height = source_height_;
                        source_changed_ = false;
                    }
                }
                if (!current) continue;
                const auto now = std::chrono::steady_clock::now();
                if (now < next_frame) {
                    std::this_thread::sleep_until(next_frame);
                    if (stopping_.load()) return;
                }
                if (Render(current.Get(), current_width, current_height)) {
                    next_frame = std::chrono::steady_clock::now() + interval;
                    current.Reset();
                } else {
                    // A hidden/minimized owner or a layout that has not arrived
                    // yet must not turn the render thread into a busy loop.
                    std::this_thread::sleep_for(std::chrono::milliseconds(10));
                }
            }
        } catch (const std::exception& error) {
            std::scoped_lock lock(error_mutex_);
            render_error_ = error.what();
        }
    }

    void ApplyLayout(HWND window) noexcept {
        const PreviewLayoutMessage layout = SnapshotLayout();
        const HWND owner = reinterpret_cast<HWND>(
            static_cast<std::uintptr_t>(layout.owner_hwnd));
        if (layout.visible == 0 || layout.width <= 0 || layout.height <= 0
            || (owner != nullptr && (!IsWindow(owner) || !IsWindowVisible(owner) || IsIconic(owner)))) {
            ShowWindow(window, SW_HIDE);
            return;
        }
        SetWindowLongPtrW(window, GWLP_HWNDPARENT, reinterpret_cast<LONG_PTR>(owner));
        POINT origin{};
        if (owner != nullptr && !ClientToScreen(owner, &origin)) {
            ShowWindow(window, SW_HIDE);
            return;
        }
        const float scale = layout.scale > 0.0F ? layout.scale : 1.0F;
        const int x = origin.x + static_cast<int>(layout.x * scale + 0.5F);
        const int y = origin.y + static_cast<int>(layout.y * scale + 0.5F);
        const int width = (std::max)(1, static_cast<int>(layout.width * scale + 0.5F));
        const int height = (std::max)(1, static_cast<int>(layout.height * scale + 0.5F));
        SetWindowPos(
            window,
            HWND_TOP,
            x,
            y,
            width,
            height,
            SWP_NOACTIVATE | SWP_SHOWWINDOW);
    }

    static RECT ContainedRect(
        UINT32 output_width,
        UINT32 output_height,
        UINT32 input_width,
        UINT32 input_height) noexcept {
        const std::uint64_t width_limited =
            static_cast<std::uint64_t>(output_width) * input_height;
        const std::uint64_t height_limited =
            static_cast<std::uint64_t>(output_height) * input_width;
        UINT32 width = output_width;
        UINT32 height = output_height;
        if (width_limited <= height_limited) {
            height = static_cast<UINT32>(
                static_cast<std::uint64_t>(input_height) * output_width / input_width);
        } else {
            width = static_cast<UINT32>(
                static_cast<std::uint64_t>(input_width) * output_height / input_height);
        }
        const LONG left = static_cast<LONG>((output_width - width) / 2U);
        const LONG top = static_cast<LONG>((output_height - height) / 2U);
        return {
            left,
            top,
            left + static_cast<LONG>(width),
            top + static_cast<LONG>(height),
        };
    }

    void EnsureSwapChain(
        HWND window,
        UINT32 width,
        UINT32 height,
        UINT32 input_width,
        UINT32 input_height) {
        if (swap_chain_ && output_width_ == width && output_height_ == height
            && input_width_ == input_width && input_height_ == input_height) {
            return;
        }
        ResetSwapChain();
        stage_.store(31U);
        ComPtr<IDXGIDevice> dxgi_device;
        Check(device_.As(&dxgi_device), "IDXGIDevice(native preview)");
        stage_.store(32U);
        ComPtr<IDXGIAdapter> adapter;
        Check(dxgi_device->GetAdapter(&adapter), "IDXGIDevice::GetAdapter(native preview)");
        stage_.store(33U);
        ComPtr<IDXGIFactory2> factory;
        Check(adapter->GetParent(IID_PPV_ARGS(&factory)), "IDXGIFactory2(native preview)");
        DXGI_SWAP_CHAIN_DESC1 description{};
        description.Width = width;
        description.Height = height;
        description.Format = DXGI_FORMAT_B8G8R8A8_UNORM;
        description.SampleDesc.Count = 1;
        description.BufferUsage = DXGI_USAGE_RENDER_TARGET_OUTPUT;
        description.BufferCount = 1;
        description.Scaling = DXGI_SCALING_STRETCH;
        description.SwapEffect = DXGI_SWAP_EFFECT_DISCARD;
        description.AlphaMode = DXGI_ALPHA_MODE_IGNORE;
        stage_.store(34U);
        Check(factory->CreateSwapChainForHwnd(
            device_.Get(),
            window,
            &description,
            nullptr,
            nullptr,
            &swap_chain_), "CreateSwapChainForHwnd(native preview)");
        stage_.store(35U);
        factory->MakeWindowAssociation(window, DXGI_MWA_NO_ALT_ENTER);

        Check(swap_chain_->GetBuffer(0, IID_PPV_ARGS(&back_buffer_)),
              "IDXGISwapChain1::GetBuffer(native preview)");
        stage_.store(36U);
        Check(device_->CreateRenderTargetView(back_buffer_.Get(), nullptr, &render_target_),
              "CreateRenderTargetView(native preview)");
        stage_.store(37U);
        stage_.store(38U);
        stage_.store(41U);
        output_width_ = width;
        output_height_ = height;
        input_width_ = input_width;
        input_height_ = input_height;
    }

    void ResetSwapChain() noexcept {
        render_target_.Reset();
        back_buffer_.Reset();
        swap_chain_.Reset();
        output_width_ = 0;
        output_height_ = 0;
        input_width_ = 0;
        input_height_ = 0;
    }

    void CreateShaders() {
        stage_.store(381U);
        stage_.store(382U);
        stage_.store(383U);
        Check(device_->CreateVertexShader(
            g_preview_vs,
            sizeof(g_preview_vs),
            nullptr,
            &vertex_shader_), "CreateVertexShader(native preview)");
        stage_.store(384U);
        Check(device_->CreatePixelShader(
            g_preview_ps,
            sizeof(g_preview_ps),
            nullptr,
            &pixel_shader_), "CreatePixelShader(native preview)");
        stage_.store(385U);
        D3D11_SAMPLER_DESC sampler_description{};
        sampler_description.Filter = D3D11_FILTER_MIN_MAG_MIP_LINEAR;
        sampler_description.AddressU = D3D11_TEXTURE_ADDRESS_CLAMP;
        sampler_description.AddressV = D3D11_TEXTURE_ADDRESS_CLAMP;
        sampler_description.AddressW = D3D11_TEXTURE_ADDRESS_CLAMP;
        sampler_description.MaxLOD = D3D11_FLOAT32_MAX;
        Check(device_->CreateSamplerState(&sampler_description, &sampler_),
              "CreateSamplerState(native preview)");
        stage_.store(386U);
    }

    void DrawOverlay(const RECT& destination) {
        PreviewOverlayMessage overlay{};
        {
            std::scoped_lock lock(overlay_mutex_);
            overlay = overlay_;
        }
        if (overlay.level == 0 || overlay.width <= 0.0F || overlay.height <= 0.0F) return;
        const float normalized_score = (std::clamp)(overlay.score, 0.0F, 100.0F) / 100.0F;
        const float color[4] = {
            0.13F + normalized_score * 0.81F,
            0.86F - normalized_score * 0.59F,
            0.35F - normalized_score * 0.08F,
            1.0F,
        };
        const LONG content_width = destination.right - destination.left;
        const LONG content_height = destination.bottom - destination.top;
        const LONG left = destination.left + static_cast<LONG>(overlay.x * content_width);
        const LONG top = destination.top + static_cast<LONG>(overlay.y * content_height);
        const LONG right = destination.left
            + static_cast<LONG>((overlay.x + overlay.width) * content_width);
        const LONG bottom = destination.top
            + static_cast<LONG>((overlay.y + overlay.height) * content_height);
        const LONG thickness = (std::max)(3L, static_cast<LONG>(output_width_ / 400U));
        const D3D11_RECT rectangles[] = {
            {left, top, right, (std::min)(bottom, top + thickness)},
            {left, (std::max)(top, bottom - thickness), right, bottom},
            {left, top, (std::min)(right, left + thickness), bottom},
            {(std::max)(left, right - thickness), top, right, bottom},
        };
        context1_->ClearView(
            render_target_.Get(),
            color,
            rectangles,
            static_cast<UINT>(std::size(rectangles)));
    }

    void ReportFps() {
        LARGE_INTEGER now{};
        LARGE_INTEGER frequency{};
        QueryPerformanceCounter(&now);
        QueryPerformanceFrequency(&frequency);
        ++presented_frames_;
        if (fps_window_started_ == 0) fps_window_started_ = now.QuadPart;
        const std::int64_t elapsed = now.QuadPart - fps_window_started_;
        if (elapsed < frequency.QuadPart) return;
        const double fps = static_cast<double>(presented_frames_)
            * static_cast<double>(frequency.QuadPart) / static_cast<double>(elapsed);
        latest_fps_.store(fps);
        fps_generation_.fetch_add(1U);
        presented_frames_ = 0;
        fps_window_started_ = now.QuadPart;
    }

    PreviewLayoutMessage SnapshotLayout() const noexcept {
        return {
            layout_owner_.load(),
            layout_x_.load(),
            layout_y_.load(),
            layout_width_.load(),
            layout_height_.load(),
            layout_scale_.load(),
            layout_visible_.load(),
        };
    }

    ComPtr<ID3D11Device> device_;
    ComPtr<ID3D11DeviceContext> context_;
    ComPtr<ID3D11DeviceContext1> context1_;
    PipeWriter& writer_;
    std::thread window_thread_;
    std::thread render_thread_;
    std::mutex window_mutex_;
    std::condition_variable window_condition_;
    std::atomic<HWND> window_{nullptr};
    bool window_ready_ = false;
    std::exception_ptr window_error_;
    std::atomic_bool stopping_{false};
    std::atomic_uint64_t layout_owner_{0};
    std::atomic_int32_t layout_x_{0};
    std::atomic_int32_t layout_y_{0};
    std::atomic_int32_t layout_width_{0};
    std::atomic_int32_t layout_height_{0};
    std::atomic<float> layout_scale_{1.0F};
    std::atomic_uint32_t layout_visible_{0};
    std::mutex overlay_mutex_;
    PreviewOverlayMessage overlay_{};
    std::mutex source_mutex_;
    std::condition_variable render_condition_;
    ComPtr<ID3D11Texture2D> source_texture_;
    UINT32 source_width_ = 0;
    UINT32 source_height_ = 0;
    bool source_changed_ = false;
    unsigned frame_rate_ = 30;
    std::mutex render_mutex_;
    ComPtr<IDXGISwapChain1> swap_chain_;
    ComPtr<ID3D11Texture2D> back_buffer_;
    ComPtr<ID3D11RenderTargetView> render_target_;
    ComPtr<ID3D11VertexShader> vertex_shader_;
    ComPtr<ID3D11PixelShader> pixel_shader_;
    ComPtr<ID3D11SamplerState> sampler_;
    UINT32 output_width_ = 0;
    UINT32 output_height_ = 0;
    UINT32 input_width_ = 0;
    UINT32 input_height_ = 0;
    std::uint64_t presented_frames_ = 0;
    std::int64_t fps_window_started_ = 0;
    std::atomic_bool first_present_reported_{false};
    std::atomic<double> latest_fps_{0.0};
    std::atomic_uint64_t fps_generation_{0};
    mutable std::mutex error_mutex_;
    std::string render_error_;
    std::atomic_uint stage_{0};
};

class CaptureServer final {
public:
    CaptureServer(
        const ServerArguments& arguments,
        PipeWriter& writer,
        SharedFrameWriter& shared_frames,
        HANDLE stop_event)
        : arguments_(arguments),
          writer_(writer),
          shared_frames_(shared_frames),
          stop_event_(stop_event) {
        CreateDevice();
        if (arguments_.native_window) {
            native_preview_ = std::make_unique<NativePreviewWindow>(
                device_.Get(), context_.Get(), writer_, arguments_.preview_fps);
        }
        CreateCaptureItem();
        StartCapture();
    }

    ~CaptureServer() { Stop(); }

    void Stop() noexcept {
        if (stopped_.exchange(true)) return;
        stopping_.store(true);
        try {
            std::scoped_lock lock(frame_mutex_);
            if (session_) session_.Close();
            if (frame_pool_) frame_pool_.Close();
        } catch (...) {
        }
        session_ = nullptr;
        frame_pool_ = nullptr;
        processing_condition_.notify_all();
        if (processing_thread_.joinable()) processing_thread_.join();
        try {
            publisher_.reset();
            native_preview_.reset();
        } catch (...) {
        }
    }

    void RequestKeyFrame() noexcept {
        if (publisher_) publisher_->RequestKeyFrame();
    }

    void UpdatePreviewLayout(const PreviewLayoutMessage& layout) noexcept {
        if (native_preview_) native_preview_->UpdateLayout(layout);
    }

    void UpdatePreviewOverlay(const PreviewOverlayMessage& overlay) noexcept {
        if (native_preview_) native_preview_->UpdateOverlay(overlay);
    }

    void PumpEncoder() {
        if (!native_preview_) return;
        const unsigned stage = native_preview_->Stage();
        if (stage != 0U && stage != native_stage_reported_) {
            native_stage_reported_ = stage;
            writer_.SendText(
                MessageType::debug,
                "native_preview_stage=" + std::to_string(stage));
        }
        if (native_preview_->HasFirstPresent() && !native_present_reported_.exchange(true)) {
            writer_.SendText(MessageType::debug, "native_preview_first_present");
        }
        double fps = 0.0;
        if (native_preview_->LatestFps(native_fps_generation_, fps)) {
            char buffer[64]{};
            const int length = _snprintf_s(
                buffer, sizeof(buffer), _TRUNCATE, "native_preview_fps=%.1f", fps);
            if (length > 0) {
                writer_.SendText(MessageType::debug, std::string(buffer, length));
            }
        }
        const std::string render_error = native_preview_->TakeError();
        if (!render_error.empty()) {
            throw std::runtime_error(render_error);
        }
    }

    [[nodiscard]] bool HasFirstFrame() const noexcept {
        return first_frame_seen_.load();
    }

    void InitializePreviewPublisher() {
        publisher_ = std::make_unique<RawPreviewPublisher>(
            preview_size_.width,
            preview_size_.height,
            writer_,
            stop_event_);
        publisher_ready_.store(true);
    }

private:
    ComPtr<IDXGIAdapter1> SelectWindowAdapter(IDXGIFactory1* factory) {
        const HWND window = reinterpret_cast<HWND>(
            static_cast<std::uintptr_t>(arguments_.hwnd));
        const HMONITOR monitor = MonitorFromWindow(window, MONITOR_DEFAULTTONEAREST);
        ComPtr<IDXGIAdapter1> fallback;
        for (UINT adapter_index = 0;; ++adapter_index) {
            ComPtr<IDXGIAdapter1> candidate;
            const HRESULT enumerated = factory->EnumAdapters1(adapter_index, &candidate);
            if (enumerated == DXGI_ERROR_NOT_FOUND) break;
            Check(enumerated, "EnumAdapters1(auto)");
            if (!fallback) fallback = candidate;
            for (UINT output_index = 0;; ++output_index) {
                ComPtr<IDXGIOutput> output;
                const HRESULT output_result = candidate->EnumOutputs(output_index, &output);
                if (output_result == DXGI_ERROR_NOT_FOUND) break;
                Check(output_result, "EnumOutputs(auto)");
                DXGI_OUTPUT_DESC output_description{};
                Check(output->GetDesc(&output_description), "IDXGIOutput::GetDesc");
                if (output_description.Monitor == monitor) {
                    return candidate;
                }
            }
        }
        if (fallback) return fallback;
        throw std::runtime_error("no hardware DXGI adapter is available");
    }

    void CreateDevice() {
        ComPtr<IDXGIFactory1> factory;
        Check(CreateDXGIFactory1(IID_PPV_ARGS(&factory)), "CreateDXGIFactory1");
        ComPtr<IDXGIAdapter1> adapter;
        if (arguments_.auto_adapter) {
            adapter = SelectWindowAdapter(factory.Get());
        } else {
            const HRESULT enumerated = factory->EnumAdapters1(arguments_.adapter, &adapter);
            if (enumerated == DXGI_ERROR_NOT_FOUND) {
                throw std::invalid_argument("selected DXGI adapter does not exist");
            }
            Check(enumerated, "EnumAdapters1");
        }
        constexpr D3D_FEATURE_LEVEL levels[] = {
            D3D_FEATURE_LEVEL_12_1,
            D3D_FEATURE_LEVEL_12_0,
            D3D_FEATURE_LEVEL_11_1,
            D3D_FEATURE_LEVEL_11_0,
        };
        D3D_FEATURE_LEVEL selected{};
        Check(D3D11CreateDevice(
            adapter.Get(),
            D3D_DRIVER_TYPE_UNKNOWN,
            nullptr,
            D3D11_CREATE_DEVICE_BGRA_SUPPORT | D3D11_CREATE_DEVICE_VIDEO_SUPPORT,
            levels,
            static_cast<UINT>(std::size(levels)),
            D3D11_SDK_VERSION,
            &device_,
            &selected,
            &context_), "D3D11CreateDevice");
        ComPtr<ID3D10Multithread> multithread;
        if (SUCCEEDED(context_.As(&multithread))) {
            multithread->SetMultithreadProtected(TRUE);
        }
        ComPtr<IDXGIDevice> dxgi_device;
        Check(device_.As(&dxgi_device), "IDXGIDevice");
        winrt::com_ptr<IInspectable> inspectable;
        Check(CreateDirect3D11DeviceFromDXGIDevice(dxgi_device.Get(), inspectable.put()),
              "CreateDirect3D11DeviceFromDXGIDevice");
        capture_device_ = inspectable.as<IDirect3DDevice>();
    }

    void CreateCaptureItem() {
        const auto interop = winrt::get_activation_factory<
            GraphicsCaptureItem,
            IGraphicsCaptureItemInterop>();
        Check(interop->CreateForWindow(
            reinterpret_cast<HWND>(static_cast<std::uintptr_t>(arguments_.hwnd)),
            winrt::guid_of<GraphicsCaptureItem>(),
            winrt::put_abi(item_)), "IGraphicsCaptureItemInterop::CreateForWindow");
    }

    void StartCapture() {
        capture_size_ = item_.Size();
        if (capture_size_.Width <= 0 || capture_size_.Height <= 0) {
            throw std::runtime_error("capture window has an empty client area");
        }
        RebuildPipeline(
            static_cast<UINT32>(capture_size_.Width),
            static_cast<UINT32>(capture_size_.Height));
        // Build the publisher before WGC can deliver its first frame.  Starting
        // it from RunServer leaves a race where a static window's only frame is
        // converted while publisher_ready_ is still false and is then lost.
        if (!arguments_.native_window) {
            InitializePreviewPublisher();
            writer_.SendText(MessageType::debug, "publisher_initialized");
        } else {
            writer_.SendText(MessageType::debug, "native_preview_initialized");
        }
        frame_pool_ = Direct3D11CaptureFramePool::CreateFreeThreaded(
            capture_device_,
            DirectXPixelFormat::B8G8R8A8UIntNormalized,
            2,
            capture_size_);
        frame_arrived_token_ = frame_pool_.FrameArrived(
            [this](
                Direct3D11CaptureFramePool const& sender,
                winrt::Windows::Foundation::IInspectable const&) {
                OnFrame(sender);
            });
        session_ = frame_pool_.CreateCaptureSession(item_);
        session_.IsCursorCaptureEnabled(false);
        pipeline_ready_.store(true);
        processing_thread_ = std::thread([this] { ProcessingLoop(); });
        session_.StartCapture();
    }

    void RebuildPipeline(UINT32 width, UINT32 height) {
        preview_size_ = CalculatePreviewSize(width, height, arguments_.preview_width);
        preview_x_map_.resize(preview_size_.width);
        preview_y_map_.resize(preview_size_.height);
        for (UINT32 x = 0; x < preview_size_.width; ++x) {
            preview_x_map_[x] = static_cast<UINT32>(
                static_cast<std::uint64_t>(x) * width / preview_size_.width);
        }
        for (UINT32 y = 0; y < preview_size_.height; ++y) {
            preview_y_map_[y] = static_cast<UINT32>(
                static_cast<std::uint64_t>(y) * height / preview_size_.height);
        }
        staging_.Reset();
        capture_width_ = width;
        capture_height_ = height;
    }

    void OnFrame(Direct3D11CaptureFramePool const& sender) noexcept {
        std::scoped_lock frame_lock(frame_mutex_);
        if (stopping_.load() || !pipeline_ready_.load()) {
            return;
        }
        try {
            const Direct3D11CaptureFrame frame = sender.TryGetNextFrame();
            if (!frame) return;
            const bool trace_first = !first_frame_seen_.exchange(true);
            if (trace_first) writer_.SendText(MessageType::debug, "frame_received");
            const SizeInt32 content_size = frame.ContentSize();
            if (content_size.Width <= 0 || content_size.Height <= 0) return;
            if (content_size.Width != capture_size_.Width || content_size.Height != capture_size_.Height) {
                throw std::runtime_error(
                    "capture window size changed during hardware preview");
            }
            if (trace_first) writer_.SendText(MessageType::debug, "frame_size_valid");
            ComPtr<ID3D11Texture2D> texture;
            const auto access = frame.Surface().as<IDirect3DDxgiInterfaceAccess>();
            Check(access->GetInterface(IID_PPV_ARGS(&texture)), "IDirect3DDxgiInterfaceAccess");
            if (trace_first) writer_.SendText(MessageType::debug, "frame_texture_acquired");
            if (native_preview_) {
                native_preview_->Submit(
                    texture.Get(),
                    static_cast<UINT32>(content_size.Width),
                    static_cast<UINT32>(content_size.Height));
            }
            const std::int64_t timestamp_100ns = FrameTimestamp100ns(
                frame.SystemRelativeTime());
            frame.Close();
            LARGE_INTEGER counter{};
            QueryPerformanceCounter(&counter);
            PendingCapture pending{
                std::move(texture),
                frame_id_.fetch_add(1) + 1,
                timestamp_100ns,
                counter.QuadPart,
            };
            {
                std::scoped_lock processing_lock(processing_mutex_);
                pending_capture_ = std::move(pending);
            }
            processing_condition_.notify_one();
            if (trace_first) writer_.SendText(MessageType::debug, "frame_queued");
        } catch (const std::exception& error) {
            stopping_.store(true);
            try {
                writer_.SendText(MessageType::error, error.what());
            } catch (...) {
            }
            SetEvent(stop_event_);
        }
    }

    struct PendingCapture {
        ComPtr<ID3D11Texture2D> texture;
        std::uint64_t frame_id;
        std::int64_t timestamp_100ns;
        std::int64_t captured_qpc;
    };

    void ProcessingLoop() noexcept {
        try {
            while (!stopping_.load()) {
                std::optional<PendingCapture> pending;
                {
                    std::unique_lock lock(processing_mutex_);
                    processing_condition_.wait(lock, [this] {
                        return stopping_.load() || pending_capture_.has_value();
                    });
                    if (stopping_.load()) return;
                    pending = std::move(pending_capture_);
                    pending_capture_.reset();
                }
                ProcessFrame(*pending);
            }
        } catch (const std::exception& error) {
            stopping_.store(true);
            try {
                writer_.SendText(MessageType::error, error.what());
            } catch (...) {
            }
            SetEvent(stop_event_);
        }
    }

    void ProcessFrame(const PendingCapture& frame) {
        const bool recognition_due = Due(
            frame.captured_qpc, last_recognition_qpc_, arguments_.recognition_fps);
        const bool preview_due = Due(
            frame.captured_qpc, last_preview_qpc_, arguments_.preview_fps);
        if (!recognition_due && !preview_due) return;

        std::vector<std::byte> bgra;
        if (recognition_due) {
            if (!readback_started_.exchange(true)) {
                writer_.SendText(MessageType::debug, "bgra_readback_started");
            }
            bgra = ReadbackFrame(frame.texture.Get());
            if (!readback_completed_.exchange(true)) {
                writer_.SendText(MessageType::debug, "bgra_readback_completed");
            }
            const std::uint32_t packed_stride = capture_width_ * 4U;
            shared_frames_.Publish(
                frame.frame_id,
                frame.captured_qpc,
                capture_width_,
                capture_height_,
                packed_stride,
                bgra.data(),
                static_cast<std::uint32_t>(bgra.size()));
            last_recognition_qpc_ = frame.captured_qpc;
        }
        if (preview_due && !native_preview_ && publisher_ready_.load()) {
            if (bgra.empty()) {
                bgra = ReadbackFrame(frame.texture.Get());
            }
            if (!conversion_started_.exchange(true)) {
                writer_.SendText(MessageType::debug, "cpu_nv12_conversion_started");
            }
            publisher_->Submit(
                ConvertBGRAtoNV12(bgra),
                frame.frame_id,
                frame.timestamp_100ns);
            if (!conversion_completed_.exchange(true)) {
                writer_.SendText(MessageType::debug, "cpu_nv12_conversion_completed");
            }
            last_preview_qpc_ = frame.captured_qpc;
        }
    }

    static std::int64_t FrameTimestamp100ns(winrt::Windows::Foundation::TimeSpan value) noexcept {
        return value.count();
    }

    static bool Due(std::int64_t now, std::int64_t previous, unsigned fps) {
        if (fps == 0) return false;
        LARGE_INTEGER frequency{};
        QueryPerformanceFrequency(&frequency);
        return previous == 0 || now - previous >= frequency.QuadPart / fps;
    }

    std::vector<std::byte> ReadbackFrame(ID3D11Texture2D* source) {
        if (!staging_) {
            D3D11_TEXTURE2D_DESC description{};
            source->GetDesc(&description);
            description.BindFlags = 0;
            description.MiscFlags = 0;
            description.Usage = D3D11_USAGE_STAGING;
            description.CPUAccessFlags = D3D11_CPU_ACCESS_READ;
            Check(device_->CreateTexture2D(&description, nullptr, &staging_),
                  "CreateTexture2D(staging)");
        }
        context_->CopyResource(staging_.Get(), source);
        D3D11_MAPPED_SUBRESOURCE mapped{};
        Check(context_->Map(staging_.Get(), 0, D3D11_MAP_READ, 0, &mapped), "Map(staging)");
        const std::uint32_t packed_stride = capture_width_ * 4U;
        const std::uint64_t byte_count_64 = static_cast<std::uint64_t>(packed_stride) * capture_height_;
        if (byte_count_64 > (std::numeric_limits<std::uint32_t>::max)()) {
            context_->Unmap(staging_.Get(), 0);
            throw std::runtime_error("recognition frame is too large");
        }
        std::vector<std::byte> packed(static_cast<std::size_t>(byte_count_64));
        const auto* source_bytes = static_cast<const std::byte*>(mapped.pData);
        for (UINT32 row = 0; row < capture_height_; ++row) {
            std::memcpy(
                packed.data() + static_cast<std::size_t>(row) * packed_stride,
                source_bytes + static_cast<std::size_t>(row) * mapped.RowPitch,
                packed_stride);
        }
        context_->Unmap(staging_.Get(), 0);
        return packed;
    }

    std::vector<std::byte> ConvertBGRAtoNV12(
        const std::vector<std::byte>& source) const {
        const UINT32 output_width = preview_size_.width;
        const UINT32 output_height = preview_size_.height;
        std::vector<std::byte> output(
            static_cast<std::size_t>(output_width) * output_height * 3U / 2U);
        const auto* pixels = reinterpret_cast<const std::uint8_t*>(source.data());
        auto* destination = reinterpret_cast<std::uint8_t*>(output.data());
        auto clamp_byte = [](int value) noexcept -> std::uint8_t {
            return static_cast<std::uint8_t>((std::min)(255, (std::max)(0, value)));
        };
        auto sample = [&](UINT32 x, UINT32 y, int& blue, int& green, int& red) noexcept {
            const UINT32 source_x = preview_x_map_[x];
            const UINT32 source_y = preview_y_map_[y];
            const auto offset = (
                static_cast<std::size_t>(source_y) * capture_width_ + source_x) * 4U;
            blue = pixels[offset];
            green = pixels[offset + 1U];
            red = pixels[offset + 2U];
        };

        for (UINT32 y = 0; y < output_height; ++y) {
            auto* output_row = destination + static_cast<std::size_t>(y) * output_width;
            for (UINT32 x = 0; x < output_width; ++x) {
                int blue = 0;
                int green = 0;
                int red = 0;
                sample(x, y, blue, green, red);
                output_row[x] = clamp_byte(
                    ((66 * red + 129 * green + 25 * blue + 128) >> 8) + 16);
            }
        }

        auto* uv_plane = destination
            + static_cast<std::size_t>(output_width) * output_height;
        for (UINT32 y = 0; y < output_height; y += 2U) {
            auto* uv_row = uv_plane + static_cast<std::size_t>(y / 2U) * output_width;
            for (UINT32 x = 0; x < output_width; x += 2U) {
                int blue_sum = 0;
                int green_sum = 0;
                int red_sum = 0;
                for (UINT32 dy = 0; dy < 2U; ++dy) {
                    for (UINT32 dx = 0; dx < 2U; ++dx) {
                        int blue = 0;
                        int green = 0;
                        int red = 0;
                        sample(x + dx, y + dy, blue, green, red);
                        blue_sum += blue;
                        green_sum += green;
                        red_sum += red;
                    }
                }
                const int blue = (blue_sum + 2) / 4;
                const int green = (green_sum + 2) / 4;
                const int red = (red_sum + 2) / 4;
                uv_row[x] = clamp_byte(
                    ((-38 * red - 74 * green + 112 * blue + 128) >> 8) + 128);
                uv_row[x + 1U] = clamp_byte(
                    ((112 * red - 94 * green - 18 * blue + 128) >> 8) + 128);
            }
        }
        return output;
    }

    const ServerArguments& arguments_;
    PipeWriter& writer_;
    SharedFrameWriter& shared_frames_;
    HANDLE stop_event_;
    ComPtr<ID3D11Device> device_;
    ComPtr<ID3D11DeviceContext> context_;
    IDirect3DDevice capture_device_{nullptr};
    GraphicsCaptureItem item_{nullptr};
    Direct3D11CaptureFramePool frame_pool_{nullptr};
    GraphicsCaptureSession session_{nullptr};
    winrt::event_token frame_arrived_token_{};
    SizeInt32 capture_size_{};
    UINT32 capture_width_ = 0;
    UINT32 capture_height_ = 0;
    ComPtr<ID3D11Texture2D> staging_;
    std::unique_ptr<RawPreviewPublisher> publisher_;
    std::unique_ptr<NativePreviewWindow> native_preview_;
    PreviewSize preview_size_{0, 0};
    std::vector<UINT32> preview_x_map_;
    std::vector<UINT32> preview_y_map_;
    std::mutex frame_mutex_;
    std::mutex processing_mutex_;
    std::condition_variable processing_condition_;
    std::optional<PendingCapture> pending_capture_;
    std::thread processing_thread_;
    std::atomic_bool stopping_{false};
    std::atomic_bool stopped_{false};
    std::atomic_bool pipeline_ready_{false};
    std::atomic_uint64_t frame_id_{0};
    std::atomic_bool first_frame_seen_{false};
    std::atomic_bool publisher_ready_{false};
    std::atomic_bool readback_started_{false};
    std::atomic_bool readback_completed_{false};
    std::atomic_bool conversion_started_{false};
    std::atomic_bool conversion_completed_{false};
    std::int64_t last_preview_qpc_ = 0;
    std::int64_t last_recognition_qpc_ = 0;
    std::atomic_bool native_present_reported_{false};
    std::uint64_t native_fps_generation_ = 0;
    unsigned native_stage_reported_ = 0;
};

std::string ReadyJson(const ServerArguments& arguments) {
    return std::string("{\"ok\":true,\"protocol\":1,\"adapter_id\":")
        + std::to_string(arguments.adapter)
        + std::string(
            arguments.native_window
                ? ",\"preview_codec\":\"native\",\"shared_format\":\"BGRA\"}"
                : ",\"preview_codec\":\"NV12\",\"shared_format\":\"BGRA\"}");
}

} // namespace

int RunServer(const ServerArguments& arguments) {
    winrt::init_apartment(winrt::apartment_type::multi_threaded);
    Check(MFStartup(MF_VERSION, MFSTARTUP_FULL), "MFStartup");
    try {
        UniqueHandle pipe(CreateNamedPipeW(
            arguments.pipe_name.c_str(),
            PIPE_ACCESS_DUPLEX,
            PIPE_TYPE_BYTE | PIPE_READMODE_BYTE | PIPE_WAIT | PIPE_REJECT_REMOTE_CLIENTS,
            1,
            4U * 1024U * 1024U,
            64U * 1024U,
            0,
            nullptr));
        if (!pipe) {
            Check(HRESULT_FROM_WIN32(GetLastError()), "CreateNamedPipeW");
        }
        if (!ConnectNamedPipe(pipe.get(), nullptr)) {
            const DWORD error = GetLastError();
            if (error != ERROR_PIPE_CONNECTED) {
                Check(HRESULT_FROM_WIN32(error), "ConnectNamedPipe");
            }
        }
        const IncomingMessage hello = ReadMessage(pipe.get());
        const std::string supplied_token(
            reinterpret_cast<const char*>(hello.payload.data()),
            hello.payload.size());
        if (hello.type != MessageType::hello || supplied_token != arguments.token) {
            throw std::runtime_error("preview helper authentication failed");
        }

        PipeWriter writer(pipe.get());
        SharedFrameWriter shared_frames(arguments.mapping_name, arguments.mapping_capacity);
        UniqueHandle stop_event(CreateEventW(nullptr, TRUE, FALSE, nullptr));
        if (!stop_event) {
            Check(HRESULT_FROM_WIN32(GetLastError()), "CreateEventW");
        }
        // READY means the authenticated IPC and shared-memory contract are
        // available.  WGC/MFT startup follows and reports any failure through
        // the already-live error channel, so the Python launcher can enforce
        // its own bounded first-frame deadline instead of hanging in a pipe
        // handshake inside a vendor driver.
        writer.SendText(MessageType::ready, ReadyJson(arguments));
        std::unique_ptr<CaptureServer> capture;
        try {
            capture = std::make_unique<CaptureServer>(
                arguments,
                writer,
                shared_frames,
                stop_event.get());
        } catch (const std::exception& error) {
            writer.SendText(MessageType::error, error.what());
            throw;
        }
        writer.SendText(MessageType::debug, "capture_created");
        const ULONGLONG first_frame_deadline = GetTickCount64() + 5'000;
        while (!capture->HasFirstFrame() && GetTickCount64() < first_frame_deadline) {
            Sleep(1);
        }
        if (!capture->HasFirstFrame()) {
            throw std::runtime_error("WGC did not produce a first frame");
        }
        writer.SendText(MessageType::debug, "wgc_first_frame");
        std::thread control([&] {
            try {
                writer.SendText(MessageType::debug, "control_started");
                while (WaitForSingleObject(stop_event.get(), 0) != WAIT_OBJECT_0) {
                    DWORD available = 0;
                    if (!PeekNamedPipe(
                            pipe.get(), nullptr, 0, nullptr, &available, nullptr)) {
                        throw std::runtime_error("preview helper pipe disconnected");
                    }
                    if (available < sizeof(PipeHeader)) {
                        if (WaitForSingleObject(stop_event.get(), 5) == WAIT_OBJECT_0) {
                            return;
                        }
                        continue;
                    }
                    const IncomingMessage message = ReadMessage(pipe.get());
                    if (message.type == MessageType::force_key_frame) {
                        capture->RequestKeyFrame();
                    } else if (message.type == MessageType::preview_layout) {
                        if (message.payload.size() != sizeof(PreviewLayoutMessage)) {
                            throw std::runtime_error("invalid native preview layout payload");
                        }
                        PreviewLayoutMessage layout{};
                        std::memcpy(&layout, message.payload.data(), sizeof(layout));
                        writer.SendText(MessageType::debug, "preview_layout_received");
                        capture->UpdatePreviewLayout(layout);
                    } else if (message.type == MessageType::preview_overlay) {
                        if (message.payload.size() != sizeof(PreviewOverlayMessage)) {
                            throw std::runtime_error("invalid native preview overlay payload");
                        }
                        PreviewOverlayMessage overlay{};
                        std::memcpy(&overlay, message.payload.data(), sizeof(overlay));
                        capture->UpdatePreviewOverlay(overlay);
                    } else if (message.type == MessageType::stop) {
                        SetEvent(stop_event.get());
                        return;
                    }
                }
            } catch (...) {
                SetEvent(stop_event.get());
            }
        });

        while (WaitForSingleObject(stop_event.get(), 2) == WAIT_TIMEOUT) {
            try {
                capture->PumpEncoder();
            } catch (const std::exception& error) {
                writer.SendText(MessageType::error, error.what());
                SetEvent(stop_event.get());
            }
        }
        capture->Stop();
        CancelIoEx(pipe.get(), nullptr);
        DisconnectNamedPipe(pipe.get());
        if (control.joinable()) control.join();
        MFShutdown();
        winrt::uninit_apartment();
        return 0;
    } catch (...) {
        MFShutdown();
        winrt::uninit_apartment();
        throw;
    }
}
