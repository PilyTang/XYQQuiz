#include "encoder_worker.h"

#include <windows.h>

#include <codecapi.h>
#include <dxgi1_2.h>
#include <mfapi.h>
#include <mferror.h>
#include <mfidl.h>
#include <mftransform.h>
#include <icodecapi.h>
#include <wrl/client.h>

#include <algorithm>
#include <cstddef>
#include <cwctype>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

using Microsoft::WRL::ComPtr;

namespace {

void Check(HRESULT result, std::string_view operation) {
    if (FAILED(result)) {
        char message[192]{};
        _snprintf_s(
            message,
            sizeof(message),
            _TRUNCATE,
            "%.*s failed with HRESULT 0x%08lX",
            static_cast<int>(operation.size()),
            operation.data(),
            static_cast<unsigned long>(result));
        throw std::runtime_error(message);
    }
}

void ReadExact(HANDLE input, void* destination, std::size_t size) {
    auto* cursor = static_cast<std::byte*>(destination);
    while (size != 0) {
        DWORD read = 0;
        const DWORD chunk = static_cast<DWORD>((std::min)(
            size,
            static_cast<std::size_t>((std::numeric_limits<DWORD>::max)())));
        if (!ReadFile(input, cursor, chunk, &read, nullptr) || read == 0) {
            throw std::runtime_error("encoder worker input closed");
        }
        cursor += read;
        size -= read;
    }
}

void WriteExact(HANDLE output, const void* source, std::size_t size) {
    const auto* cursor = static_cast<const std::byte*>(source);
    while (size != 0) {
        DWORD written = 0;
        const DWORD chunk = static_cast<DWORD>((std::min)(
            size,
            static_cast<std::size_t>((std::numeric_limits<DWORD>::max)())));
        if (!WriteFile(output, cursor, chunk, &written, nullptr) || written == 0) {
            throw std::runtime_error("encoder worker output closed");
        }
        cursor += written;
        size -= written;
    }
}

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

void SetCodecBoolean(ICodecAPI* codec, const GUID& key, bool enabled) noexcept {
    if (codec == nullptr) return;
    VARIANT value;
    VariantInit(&value);
    value.vt = VT_BOOL;
    value.boolVal = enabled ? VARIANT_TRUE : VARIANT_FALSE;
    codec->SetValue(&key, &value);
    VariantClear(&value);
}

void SetCodecUnsigned(ICodecAPI* codec, const GUID& key, ULONG number) noexcept {
    if (codec == nullptr) return;
    VARIANT value;
    VariantInit(&value);
    value.vt = VT_UI4;
    value.ulVal = number;
    codec->SetValue(&key, &value);
    VariantClear(&value);
}

std::wstring Upper(std::wstring value) {
    std::transform(value.begin(), value.end(), value.begin(), towupper);
    return value;
}

struct EncoderState {
    ComPtr<IMFActivate> activation;
    ComPtr<IMFTransform> encoder;
    ComPtr<IMFMediaEventGenerator> events;
    ComPtr<ICodecAPI> codec;
};

EncoderState CreateEncoder(const EncoderWorkerArguments& arguments) {
    ComPtr<IDXGIFactory1> factory;
    Check(CreateDXGIFactory1(IID_PPV_ARGS(&factory)), "CreateDXGIFactory1");
    ComPtr<IDXGIAdapter1> adapter;
    Check(factory->EnumAdapters1(arguments.adapter, &adapter), "EnumAdapters1");
    DXGI_ADAPTER_DESC1 adapter_description{};
    Check(adapter->GetDesc1(&adapter_description), "GetDesc1");

    MFT_REGISTER_TYPE_INFO output_type{MFMediaType_Video, MFVideoFormat_H264};
    IMFActivate** raw = nullptr;
    UINT32 count = 0;
    Check(MFTEnumEx(
        MFT_CATEGORY_VIDEO_ENCODER,
        MFT_ENUM_FLAG_HARDWARE | MFT_ENUM_FLAG_SORTANDFILTER,
        nullptr,
        &output_type,
        &raw,
        &count), "MFTEnumEx(H264)");
    std::vector<ComPtr<IMFActivate>> candidates;
    for (UINT32 index = 0; index < count; ++index) {
        candidates.emplace_back(raw[index]);
    }
    CoTaskMemFree(raw);
    const auto matches_vendor = [&](IMFActivate* candidate) {
        WCHAR* raw_name = nullptr;
        UINT32 length = 0;
        if (FAILED(candidate->GetAllocatedString(
                MFT_FRIENDLY_NAME_Attribute,
                &raw_name,
                &length))) return false;
        const std::wstring name = Upper(std::wstring(raw_name, length));
        CoTaskMemFree(raw_name);
        if (adapter_description.VendorId == 0x10DEU) return name.find(L"NVIDIA") != std::wstring::npos;
        if (adapter_description.VendorId == 0x1002U) return name.find(L"AMD") != std::wstring::npos;
        if (adapter_description.VendorId == 0x8086U) return name.find(L"INTEL") != std::wstring::npos;
        return false;
    };
    std::stable_sort(candidates.begin(), candidates.end(), [&](const auto& left, const auto& right) {
        return matches_vendor(left.Get()) && !matches_vendor(right.Get());
    });

    EncoderState state;
    HRESULT last_error = E_FAIL;
    for (const auto& activation : candidates) {
        ComPtr<IMFTransform> candidate;
        last_error = activation->ActivateObject(IID_PPV_ARGS(&candidate));
        if (FAILED(last_error)) continue;
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
            state.activation = activation;
            state.encoder = std::move(candidate);
            break;
        }
        activation->ShutdownObject();
    }
    if (!state.encoder) Check(last_error, "Activate H264 encoder");
    Check(state.encoder.As(&state.events), "IMFMediaEventGenerator");
    state.encoder.As(&state.codec);
    const UINT32 bitrate = (std::max)(2'000'000U, arguments.width * arguments.height * 6U);
    const auto output = CreateVideoType(
        MFVideoFormat_H264,
        arguments.width,
        arguments.height,
        arguments.frame_rate,
        bitrate);
    const auto input = CreateVideoType(
        MFVideoFormat_NV12,
        arguments.width,
        arguments.height,
        arguments.frame_rate);
    Check(state.encoder->SetOutputType(0, output.Get(), 0), "SetOutputType(H264)");
    Check(state.encoder->SetInputType(0, input.Get(), 0), "SetInputType(NV12)");
    SetCodecBoolean(state.codec.Get(), CODECAPI_AVLowLatencyMode, true);
    SetCodecUnsigned(state.codec.Get(), CODECAPI_AVEncMPVDefaultBPictureCount, 0);
    SetCodecUnsigned(state.codec.Get(), CODECAPI_AVEncMPVGOPSize, arguments.frame_rate);
    Check(state.encoder->ProcessMessage(MFT_MESSAGE_NOTIFY_BEGIN_STREAMING, 0), "BEGIN_STREAMING");
    Check(state.encoder->ProcessMessage(MFT_MESSAGE_NOTIFY_START_OF_STREAM, 0), "START_OF_STREAM");
    return state;
}

ComPtr<IMFSample> CreateInputSample(
    const std::vector<std::byte>& bytes,
    std::int64_t timestamp,
    std::int64_t duration) {
    ComPtr<IMFMediaBuffer> buffer;
    Check(MFCreateMemoryBuffer(static_cast<DWORD>(bytes.size()), &buffer), "MFCreateMemoryBuffer(input)");
    BYTE* destination = nullptr;
    Check(buffer->Lock(&destination, nullptr, nullptr), "Lock(input)");
    std::memcpy(destination, bytes.data(), bytes.size());
    Check(buffer->Unlock(), "Unlock(input)");
    Check(buffer->SetCurrentLength(static_cast<DWORD>(bytes.size())), "SetCurrentLength(input)");
    ComPtr<IMFSample> sample;
    Check(MFCreateSample(&sample), "MFCreateSample(input)");
    Check(sample->AddBuffer(buffer.Get()), "AddBuffer(input)");
    Check(sample->SetSampleTime(timestamp), "SetSampleTime(input)");
    Check(sample->SetSampleDuration(duration), "SetSampleDuration(input)");
    return sample;
}

struct EncodedBytes {
    std::vector<std::byte> bytes;
    std::int64_t timestamp = 0;
    bool key_frame = false;
};

EncodedBytes ReadOutput(IMFTransform* encoder) {
    MFT_OUTPUT_STREAM_INFO info{};
    Check(encoder->GetOutputStreamInfo(0, &info), "GetOutputStreamInfo");
    ComPtr<IMFSample> owned;
    MFT_OUTPUT_DATA_BUFFER output{};
    if ((info.dwFlags & MFT_OUTPUT_STREAM_PROVIDES_SAMPLES) == 0) {
        Check(MFCreateSample(&owned), "MFCreateSample(output)");
        ComPtr<IMFMediaBuffer> buffer;
        Check(MFCreateMemoryBuffer(
            (std::max)(info.cbSize, static_cast<DWORD>(2U * 1024U * 1024U)),
            &buffer), "MFCreateMemoryBuffer(output)");
        Check(owned->AddBuffer(buffer.Get()), "AddBuffer(output)");
        output.pSample = owned.Get();
    }
    DWORD status = 0;
    Check(encoder->ProcessOutput(0, 1, &output, &status), "ProcessOutput");
    if (output.pEvents != nullptr) output.pEvents->Release();
    ComPtr<IMFSample> provided;
    IMFSample* sample = output.pSample;
    if (!owned && sample != nullptr) provided.Attach(sample);
    if (sample == nullptr) throw std::runtime_error("encoder returned no output sample");
    EncodedBytes result;
    LONGLONG timestamp = 0;
    if (SUCCEEDED(sample->GetSampleTime(&timestamp))) result.timestamp = timestamp;
    UINT32 clean = FALSE;
    result.key_frame = SUCCEEDED(sample->GetUINT32(MFSampleExtension_CleanPoint, &clean)) && clean != FALSE;
    ComPtr<IMFMediaBuffer> contiguous;
    Check(sample->ConvertToContiguousBuffer(&contiguous), "ConvertToContiguousBuffer");
    BYTE* data = nullptr;
    DWORD length = 0;
    Check(contiguous->Lock(&data, nullptr, &length), "Lock(output)");
    result.bytes.resize(length);
    if (length != 0) std::memcpy(result.bytes.data(), data, length);
    Check(contiguous->Unlock(), "Unlock(output)");
    return result;
}

void WaitForNeedInput(IMFMediaEventGenerator* events) {
    while (true) {
        ComPtr<IMFMediaEvent> event;
        Check(events->GetEvent(0, &event), "GetEvent(need input)");
        HRESULT status = S_OK;
        Check(event->GetStatus(&status), "GetStatus(need input)");
        Check(status, "encoder event");
        MediaEventType type{};
        Check(event->GetType(&type), "GetType(need input)");
        if (type == METransformNeedInput) return;
    }
}

EncodedBytes EncodeOne(
    EncoderState& state,
    const EncoderWorkerInputHeader& header,
    const std::vector<std::byte>& bytes,
    unsigned frame_rate) {
    if (header.force_key_frame != 0) {
        SetCodecBoolean(state.codec.Get(), CODECAPI_AVEncVideoForceKeyFrame, true);
    }
    const auto sample = CreateInputSample(bytes, header.timestamp_100ns, 10'000'000LL / frame_rate);
    Check(state.encoder->ProcessInput(0, sample.Get(), 0), "ProcessInput");
    bool need_input = false;
    while (true) {
        ComPtr<IMFMediaEvent> event;
        Check(state.events->GetEvent(0, &event), "GetEvent(output)");
        HRESULT status = S_OK;
        Check(event->GetStatus(&status), "GetStatus(output)");
        Check(status, "encoder event");
        MediaEventType type{};
        Check(event->GetType(&type), "GetType(output)");
        if (type == METransformNeedInput) {
            need_input = true;
        } else if (type == METransformHaveOutput) {
            EncodedBytes output = ReadOutput(state.encoder.Get());
            if (!need_input) WaitForNeedInput(state.events.Get());
            return output;
        }
    }
}

} // namespace

int RunEncoderWorker(const EncoderWorkerArguments& arguments) {
    if (arguments.width == 0 || arguments.height == 0 || arguments.frame_rate == 0) {
        throw std::invalid_argument("encoder worker dimensions and frame rate must be positive");
    }
    const std::size_t expected = static_cast<std::size_t>(arguments.width)
        * arguments.height * 3U / 2U;
    EncoderState encoder = CreateEncoder(arguments);
    WaitForNeedInput(encoder.events.Get());
    const HANDLE input = GetStdHandle(STD_INPUT_HANDLE);
    const HANDLE output = GetStdHandle(STD_OUTPUT_HANDLE);
    while (true) {
        EncoderWorkerInputHeader header{};
        ReadExact(input, &header, sizeof(header));
        if (header.magic != kEncoderWorkerMagic || header.payload_size != expected) {
            throw std::runtime_error("encoder worker input protocol mismatch");
        }
        std::vector<std::byte> bytes(header.payload_size);
        ReadExact(input, bytes.data(), bytes.size());
        EncodedBytes encoded = EncodeOne(encoder, header, bytes, arguments.frame_rate);
        EncoderWorkerOutputHeader response{
            kEncoderWorkerMagic,
            static_cast<std::uint32_t>(encoded.bytes.size()),
            header.frame_id,
            encoded.timestamp,
            encoded.key_frame ? static_cast<std::uint8_t>(1) : static_cast<std::uint8_t>(0),
            {},
        };
        WriteExact(output, &response, sizeof(response));
        WriteExact(output, encoded.bytes.data(), encoded.bytes.size());
    }
}
