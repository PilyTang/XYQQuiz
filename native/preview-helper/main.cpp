#include <windows.h>

#include <d3d11.h>
#include <dxgi1_2.h>
#include <mfapi.h>
#include <mferror.h>
#include <mfidl.h>
#include <mftransform.h>
#include <codecapi.h>
#include <icodecapi.h>
#include <wrl/client.h>

#include <charconv>
#include <algorithm>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include "server.h"
#include "encoder_worker.h"

using Microsoft::WRL::ComPtr;

namespace {

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
        std::ostringstream message;
        message << operation << " failed with HRESULT 0x" << std::hex
                << static_cast<unsigned long>(result);
        throw HResultError(message.str(), result);
    }
}

std::string Utf8(std::wstring_view value) {
    if (value.empty()) {
        return {};
    }
    const int size = WideCharToMultiByte(
        CP_UTF8,
        WC_ERR_INVALID_CHARS,
        value.data(),
        static_cast<int>(value.size()),
        nullptr,
        0,
        nullptr,
        nullptr);
    if (size <= 0) {
        throw std::runtime_error("WideCharToMultiByte failed");
    }
    std::string result(static_cast<size_t>(size), '\0');
    if (WideCharToMultiByte(
            CP_UTF8,
            WC_ERR_INVALID_CHARS,
            value.data(),
            static_cast<int>(value.size()),
            result.data(),
            size,
            nullptr,
            nullptr) != size) {
        throw std::runtime_error("WideCharToMultiByte produced an incomplete result");
    }
    return result;
}

std::string JsonEscape(std::string_view value) {
    std::ostringstream output;
    for (const unsigned char character : value) {
        switch (character) {
        case '\\': output << "\\\\"; break;
        case '"': output << "\\\""; break;
        case '\b': output << "\\b"; break;
        case '\f': output << "\\f"; break;
        case '\n': output << "\\n"; break;
        case '\r': output << "\\r"; break;
        case '\t': output << "\\t"; break;
        default:
            if (character < 0x20) {
                constexpr char hex[] = "0123456789ABCDEF";
                output << "\\u00" << hex[character >> 4] << hex[character & 0x0F];
            } else {
                output << static_cast<char>(character);
            }
        }
    }
    return output.str();
}

unsigned ParseUnsigned(std::wstring_view value, std::wstring_view name) {
    if (value.empty()) {
        throw std::invalid_argument(Utf8(name) + " is empty");
    }
    unsigned result = 0;
    const std::string utf8 = Utf8(value);
    const auto parsed = std::from_chars(utf8.data(), utf8.data() + utf8.size(), result);
    if (parsed.ec != std::errc{} || parsed.ptr != utf8.data() + utf8.size()) {
        throw std::invalid_argument(Utf8(name) + " must be a non-negative integer");
    }
    return result;
}

std::uint64_t ParseUnsigned64(std::wstring_view value, std::wstring_view name) {
    if (value.empty()) {
        throw std::invalid_argument(Utf8(name) + " is empty");
    }
    std::uint64_t result = 0;
    const std::string utf8 = Utf8(value);
    const auto parsed = std::from_chars(utf8.data(), utf8.data() + utf8.size(), result);
    if (parsed.ec != std::errc{} || parsed.ptr != utf8.data() + utf8.size()) {
        throw std::invalid_argument(Utf8(name) + " must be a non-negative integer");
    }
    return result;
}

struct Arguments {
    bool self_test = false;
    bool serve = false;
    bool encode_worker = false;
    unsigned adapter = 0;
    ServerArguments server;
    EncoderWorkerArguments encoder;
};

Arguments ParseArguments(int argc, wchar_t** argv) {
    Arguments result;
    for (int index = 1; index < argc; ++index) {
        const std::wstring_view argument(argv[index]);
        if (argument == L"--self-test") {
            result.self_test = true;
        } else if (argument == L"--serve") {
            result.serve = true;
        } else if (argument == L"--encode-worker") {
            result.encode_worker = true;
        } else if (argument == L"--adapter") {
            if (++index >= argc) {
                throw std::invalid_argument("--adapter requires a value");
            }
            const std::wstring_view adapter_value(argv[index]);
            if (adapter_value == L"auto") {
                result.server.auto_adapter = true;
            } else {
                result.adapter = ParseUnsigned(adapter_value, L"--adapter");
                result.server.adapter = result.adapter;
                result.encoder.adapter = result.adapter;
            }
        } else if (argument == L"--width") {
            if (++index >= argc) throw std::invalid_argument("--width requires a value");
            result.encoder.width = ParseUnsigned(argv[index], L"--width");
        } else if (argument == L"--height") {
            if (++index >= argc) throw std::invalid_argument("--height requires a value");
            result.encoder.height = ParseUnsigned(argv[index], L"--height");
        } else if (argument == L"--frame-rate") {
            if (++index >= argc) throw std::invalid_argument("--frame-rate requires a value");
            result.encoder.frame_rate = ParseUnsigned(argv[index], L"--frame-rate");
        } else if (argument == L"--hwnd") {
            if (++index >= argc) throw std::invalid_argument("--hwnd requires a value");
            result.server.hwnd = ParseUnsigned64(argv[index], L"--hwnd");
        } else if (argument == L"--pipe") {
            if (++index >= argc) throw std::invalid_argument("--pipe requires a value");
            result.server.pipe_name = argv[index];
        } else if (argument == L"--token") {
            if (++index >= argc) throw std::invalid_argument("--token requires a value");
            result.server.token = Utf8(argv[index]);
        } else if (argument == L"--mapping") {
            if (++index >= argc) throw std::invalid_argument("--mapping requires a value");
            result.server.mapping_name = argv[index];
        } else if (argument == L"--preview-width") {
            if (++index >= argc) throw std::invalid_argument("--preview-width requires a value");
            result.server.preview_width = ParseUnsigned(argv[index], L"--preview-width");
        } else if (argument == L"--preview-fps") {
            if (++index >= argc) throw std::invalid_argument("--preview-fps requires a value");
            result.server.preview_fps = ParseUnsigned(argv[index], L"--preview-fps");
        } else if (argument == L"--recognition-fps") {
            if (++index >= argc) throw std::invalid_argument("--recognition-fps requires a value");
            result.server.recognition_fps = ParseUnsigned(argv[index], L"--recognition-fps");
        } else if (argument == L"--mapping-capacity") {
            if (++index >= argc) throw std::invalid_argument("--mapping-capacity requires a value");
            result.server.mapping_capacity = ParseUnsigned(argv[index], L"--mapping-capacity");
        } else if (argument == L"--native-window") {
            result.server.native_window = true;
        } else {
            throw std::invalid_argument("unknown argument: " + Utf8(argument));
        }
    }
    const int mode_count = static_cast<int>(result.self_test)
        + static_cast<int>(result.serve)
        + static_cast<int>(result.encode_worker);
    if (mode_count != 1) {
        throw std::invalid_argument("exactly one helper mode is required");
    }
    if (result.serve && (
            result.server.hwnd == 0
            || result.server.pipe_name.empty()
            || result.server.token.empty()
            || result.server.mapping_name.empty())) {
        throw std::invalid_argument("--serve requires --hwnd, --pipe, --token and --mapping");
    }
    if (result.server.auto_adapter && !result.serve) {
        throw std::invalid_argument("--adapter auto is only valid with --serve");
    }
    if (result.encode_worker && (result.encoder.width == 0 || result.encoder.height == 0)) {
        throw std::invalid_argument("--encode-worker requires --width and --height");
    }
    return result;
}

struct ComRuntime {
    ComRuntime() {
        const HRESULT initialized = CoInitializeEx(nullptr, COINIT_MULTITHREADED);
        if (initialized != RPC_E_CHANGED_MODE) {
            Check(initialized, "CoInitializeEx");
            uninitialize_com = true;
        }
        Check(MFStartup(MF_VERSION, MFSTARTUP_FULL), "MFStartup");
        shutdown_mf = true;
    }

    ~ComRuntime() {
        if (shutdown_mf) {
            MFShutdown();
        }
        if (uninitialize_com) {
            CoUninitialize();
        }
    }

    bool uninitialize_com = false;
    bool shutdown_mf = false;
};

struct SelfTestResult {
    std::string adapter_name;
    std::string encoder_name;
    D3D_FEATURE_LEVEL feature_level{};
    DWORD encoded_bytes = 0;
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
    Check(
        MFSetAttributeRatio(type.Get(), MF_MT_FRAME_RATE, frame_rate, 1),
        "MF_MT_FRAME_RATE");
    Check(
        MFSetAttributeRatio(type.Get(), MF_MT_PIXEL_ASPECT_RATIO, 1, 1),
        "MF_MT_PIXEL_ASPECT_RATIO");
    Check(
        type->SetUINT32(MF_MT_INTERLACE_MODE, MFVideoInterlace_Progressive),
        "MF_MT_INTERLACE_MODE");
    if (bitrate != 0) {
        Check(type->SetUINT32(MF_MT_AVG_BITRATE, bitrate), "MF_MT_AVG_BITRATE");
        Check(
            type->SetUINT32(MF_MT_MPEG2_PROFILE, eAVEncH264VProfile_Base),
            "MF_MT_MPEG2_PROFILE");
    }
    return type;
}

void ConfigureEncoder(
    IMFTransform* encoder,
    UINT32 width,
    UINT32 height,
    UINT32 frame_rate,
    UINT32 bitrate) {
    const auto output = CreateVideoType(
        MFVideoFormat_H264,
        width,
        height,
        frame_rate,
        bitrate);
    const auto input = CreateVideoType(
        MFVideoFormat_NV12,
        width,
        height,
        frame_rate);
    Check(encoder->SetOutputType(0, output.Get(), 0), "SetOutputType(H264)");
    Check(encoder->SetInputType(0, input.Get(), 0), "SetInputType(NV12)");

    ComPtr<ICodecAPI> codec;
    if (SUCCEEDED(encoder->QueryInterface(IID_PPV_ARGS(&codec)))) {
        VARIANT value;
        VariantInit(&value);
        value.vt = VT_BOOL;
        value.boolVal = VARIANT_TRUE;
        codec->SetValue(&CODECAPI_AVLowLatencyMode, &value);
        VariantClear(&value);

        VariantInit(&value);
        value.vt = VT_UI4;
        value.ulVal = 0;
        codec->SetValue(&CODECAPI_AVEncMPVDefaultBPictureCount, &value);
        VariantClear(&value);

        VariantInit(&value);
        value.vt = VT_UI4;
        value.ulVal = frame_rate;
        codec->SetValue(&CODECAPI_AVEncMPVGOPSize, &value);
        VariantClear(&value);
    }
}

ComPtr<ID3D11Texture2D> CreateSyntheticNv12Texture(
    ID3D11Device* device,
    UINT32 width,
    UINT32 height) {
    std::vector<BYTE> pixels(static_cast<size_t>(width) * height * 3 / 2, 128);
    for (UINT32 row = 0; row < height; ++row) {
        for (UINT32 column = 0; column < width; ++column) {
            pixels[static_cast<size_t>(row) * width + column] =
                static_cast<BYTE>(32 + (column * 180 / width));
        }
    }
    D3D11_TEXTURE2D_DESC texture_description{};
    texture_description.Width = width;
    texture_description.Height = height;
    texture_description.MipLevels = 1;
    texture_description.ArraySize = 1;
    texture_description.Format = DXGI_FORMAT_NV12;
    texture_description.SampleDesc.Count = 1;
    texture_description.Usage = D3D11_USAGE_DEFAULT;
    texture_description.BindFlags = D3D11_BIND_RENDER_TARGET;
    D3D11_SUBRESOURCE_DATA initial{};
    initial.pSysMem = pixels.data();
    initial.SysMemPitch = width;
    initial.SysMemSlicePitch = static_cast<UINT>(pixels.size());
    ComPtr<ID3D11Texture2D> texture;
    Check(
        device->CreateTexture2D(&texture_description, &initial, &texture),
        "CreateTexture2D(NV12)");
    return texture;
}

ComPtr<IMFSample> CreateSurfaceSample(
    ID3D11Texture2D* texture,
    LONGLONG timestamp,
    LONGLONG duration) {
    ComPtr<IMFMediaBuffer> buffer;
    Check(
        MFCreateDXGISurfaceBuffer(
            __uuidof(ID3D11Texture2D),
            texture,
            0,
            FALSE,
            &buffer),
        "MFCreateDXGISurfaceBuffer");
    ComPtr<IMFSample> sample;
    Check(MFCreateSample(&sample), "MFCreateSample");
    Check(sample->AddBuffer(buffer.Get()), "IMFSample::AddBuffer");
    Check(sample->SetSampleTime(timestamp), "SetSampleTime");
    Check(sample->SetSampleDuration(duration), "SetSampleDuration");
    return sample;
}

DWORD ReadOutputSample(IMFTransform* encoder) {
    MFT_OUTPUT_STREAM_INFO stream_info{};
    Check(encoder->GetOutputStreamInfo(0, &stream_info), "GetOutputStreamInfo");
    ComPtr<IMFSample> owned_sample;
    MFT_OUTPUT_DATA_BUFFER output{};
    output.dwStreamID = 0;
    if ((stream_info.dwFlags & MFT_OUTPUT_STREAM_PROVIDES_SAMPLES) == 0) {
        Check(MFCreateSample(&owned_sample), "MFCreateSample(output)");
        ComPtr<IMFMediaBuffer> buffer;
        Check(
            MFCreateMemoryBuffer(
                (std::max)(stream_info.cbSize, static_cast<DWORD>(1024U * 1024U)),
                &buffer),
            "MFCreateMemoryBuffer(output)");
        Check(owned_sample->AddBuffer(buffer.Get()), "AddBuffer(output)");
        output.pSample = owned_sample.Get();
    }
    DWORD status = 0;
    const HRESULT processed = encoder->ProcessOutput(0, 1, &output, &status);
    if (output.pEvents != nullptr) {
        output.pEvents->Release();
    }
    Check(processed, "ProcessOutput");
    ComPtr<IMFSample> provided_sample;
    IMFSample* sample = output.pSample;
    if (!owned_sample && sample != nullptr) {
        provided_sample.Attach(sample);
    }
    if (sample == nullptr) {
        throw std::runtime_error("H.264 encoder returned no output sample");
    }
    ComPtr<IMFMediaBuffer> contiguous;
    Check(sample->ConvertToContiguousBuffer(&contiguous), "ConvertToContiguousBuffer");
    DWORD length = 0;
    Check(contiguous->GetCurrentLength(&length), "GetCurrentLength");
    if (length == 0) {
        throw std::runtime_error("H.264 encoder returned an empty sample");
    }
    return length;
}

DWORD EncodeSyntheticFrame(
    IMFTransform* encoder,
    ID3D11Device* device,
    UINT32 width,
    UINT32 height,
    UINT32 frame_rate) {
    ConfigureEncoder(encoder, width, height, frame_rate, 4'000'000);
    Check(
        encoder->ProcessMessage(MFT_MESSAGE_NOTIFY_BEGIN_STREAMING, 0),
        "MFT_MESSAGE_NOTIFY_BEGIN_STREAMING");
    Check(
        encoder->ProcessMessage(MFT_MESSAGE_NOTIFY_START_OF_STREAM, 0),
        "MFT_MESSAGE_NOTIFY_START_OF_STREAM");
    ComPtr<IMFMediaEventGenerator> event_generator;
    Check(encoder->QueryInterface(IID_PPV_ARGS(&event_generator)), "IMFMediaEventGenerator");
    bool submitted = false;
    const ULONGLONG deadline = GetTickCount64() + 10'000;
    while (GetTickCount64() < deadline) {
        ComPtr<IMFMediaEvent> event;
        const HRESULT result = event_generator->GetEvent(MF_EVENT_FLAG_NO_WAIT, &event);
        if (result == MF_E_NO_EVENTS_AVAILABLE) {
            Sleep(1);
            continue;
        }
        Check(result, "IMFMediaEventGenerator::GetEvent");
        MediaEventType event_type{};
        Check(event->GetType(&event_type), "IMFMediaEvent::GetType");
        HRESULT event_status = S_OK;
        Check(event->GetStatus(&event_status), "IMFMediaEvent::GetStatus");
        Check(event_status, "asynchronous encoder event");
        if (event_type == METransformNeedInput && !submitted) {
            const auto texture = CreateSyntheticNv12Texture(device, width, height);
            const auto sample = CreateSurfaceSample(
                texture.Get(),
                0,
                10'000'000 / frame_rate);
            Check(encoder->ProcessInput(0, sample.Get(), 0), "ProcessInput");
            submitted = true;
        } else if (event_type == METransformHaveOutput) {
            return ReadOutputSample(encoder);
        }
    }
    throw std::runtime_error("hardware H.264 encoder self-test timed out");
}

SelfTestResult RunSelfTest(unsigned adapter_index) {
    ComPtr<IDXGIFactory1> factory;
    Check(CreateDXGIFactory1(IID_PPV_ARGS(&factory)), "CreateDXGIFactory1");

    ComPtr<IDXGIAdapter1> adapter;
    const HRESULT enumerate = factory->EnumAdapters1(adapter_index, &adapter);
    if (enumerate == DXGI_ERROR_NOT_FOUND) {
        throw std::invalid_argument("DXGI adapter does not exist");
    }
    Check(enumerate, "IDXGIFactory1::EnumAdapters1");

    DXGI_ADAPTER_DESC1 description{};
    Check(adapter->GetDesc1(&description), "IDXGIAdapter1::GetDesc1");
    if ((description.Flags & DXGI_ADAPTER_FLAG_SOFTWARE) != 0) {
        throw std::invalid_argument("software DXGI adapters are not supported");
    }

    constexpr D3D_FEATURE_LEVEL requested_levels[] = {
        D3D_FEATURE_LEVEL_12_1,
        D3D_FEATURE_LEVEL_12_0,
        D3D_FEATURE_LEVEL_11_1,
        D3D_FEATURE_LEVEL_11_0,
    };
    ComPtr<ID3D11Device> device;
    ComPtr<ID3D11DeviceContext> context;
    D3D_FEATURE_LEVEL selected_level{};
    Check(
        D3D11CreateDevice(
            adapter.Get(),
            D3D_DRIVER_TYPE_UNKNOWN,
            nullptr,
            D3D11_CREATE_DEVICE_BGRA_SUPPORT | D3D11_CREATE_DEVICE_VIDEO_SUPPORT,
            requested_levels,
            static_cast<UINT>(std::size(requested_levels)),
            D3D11_SDK_VERSION,
            &device,
            &selected_level,
            &context),
        "D3D11CreateDevice");

    MFT_REGISTER_TYPE_INFO output_type{MFMediaType_Video, MFVideoFormat_H264};
    IMFActivate** raw_activations = nullptr;
    UINT32 activation_count = 0;
    Check(
        MFTEnumEx(
            MFT_CATEGORY_VIDEO_ENCODER,
            MFT_ENUM_FLAG_HARDWARE | MFT_ENUM_FLAG_SORTANDFILTER,
            nullptr,
            &output_type,
            &raw_activations,
            &activation_count),
        "MFTEnumEx(H.264 hardware encoder)");
    if (activation_count == 0 || raw_activations == nullptr) {
        CoTaskMemFree(raw_activations);
        throw std::runtime_error("no hardware H.264 Media Foundation encoder is available");
    }

    std::vector<ComPtr<IMFActivate>> activations;
    activations.reserve(activation_count);
    for (UINT32 index = 0; index < activation_count; ++index) {
        activations.emplace_back(raw_activations[index]);
    }
    CoTaskMemFree(raw_activations);

    UINT reset_token = 0;
    ComPtr<IMFDXGIDeviceManager> device_manager;
    Check(
        MFCreateDXGIDeviceManager(&reset_token, &device_manager),
        "MFCreateDXGIDeviceManager");
    Check(device_manager->ResetDevice(device.Get(), reset_token), "ResetDevice");
    std::wstring encoder_wide;
    ComPtr<IMFTransform> encoder;
    HRESULT last_encoder_error = E_FAIL;
    for (const auto& activation : activations) {
        WCHAR* friendly_name = nullptr;
        UINT32 friendly_name_length = 0;
        if (SUCCEEDED(activation->GetAllocatedString(
                MFT_FRIENDLY_NAME_Attribute,
                &friendly_name,
                &friendly_name_length))) {
            encoder_wide.assign(friendly_name, friendly_name_length);
            CoTaskMemFree(friendly_name);
        } else {
            encoder_wide = L"unnamed hardware H.264 encoder";
        }

        ComPtr<IMFTransform> candidate;
        last_encoder_error = activation->ActivateObject(IID_PPV_ARGS(&candidate));
        if (FAILED(last_encoder_error)) {
            continue;
        }
        ComPtr<IMFAttributes> encoder_attributes;
        last_encoder_error = candidate->GetAttributes(&encoder_attributes);
        if (SUCCEEDED(last_encoder_error)) {
            UINT32 asynchronous = FALSE;
            if (SUCCEEDED(
                    encoder_attributes->GetUINT32(MF_TRANSFORM_ASYNC, &asynchronous))
                && asynchronous != FALSE) {
                last_encoder_error = encoder_attributes->SetUINT32(
                    MF_TRANSFORM_ASYNC_UNLOCK,
                    TRUE);
            }
        }
        if (SUCCEEDED(last_encoder_error)) {
            last_encoder_error = candidate->ProcessMessage(
                MFT_MESSAGE_SET_D3D_MANAGER,
                reinterpret_cast<ULONG_PTR>(device_manager.Get()));
        }
        if (SUCCEEDED(last_encoder_error)) {
            encoder = std::move(candidate);
            break;
        }
        activation->ShutdownObject();
    }
    if (!encoder) {
        Check(last_encoder_error, "No H.264 encoder accepted the selected D3D device");
    }

    const DWORD encoded_bytes = EncodeSyntheticFrame(
        encoder.Get(),
        device.Get(),
        640,
        360,
        30);

    return {
        Utf8(description.Description),
        Utf8(encoder_wide),
        selected_level,
        encoded_bytes,
    };
}

} // namespace

int wmain(int argc, wchar_t** argv) {
    try {
        // Match WebView2/WinForms coordinates on scaled or mixed-DPI displays.
        // ERROR_ACCESS_DENIED only means a manifest or host already selected a
        // process DPI context before this entry point, which is also safe.
        if (!SetProcessDpiAwarenessContext(
                DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2)
            && GetLastError() != ERROR_ACCESS_DENIED) {
            throw std::runtime_error("SetProcessDpiAwarenessContext failed");
        }
        const Arguments arguments = ParseArguments(argc, argv);
        if (arguments.serve) {
            return RunServer(arguments.server);
        }
        if (arguments.encode_worker) {
            ComRuntime runtime;
            return RunEncoderWorker(arguments.encoder);
        }
        ComRuntime runtime;
        const SelfTestResult result = RunSelfTest(arguments.adapter);
        std::cout << "{\"ok\":true,\"adapter_id\":" << arguments.adapter
                  << ",\"adapter_name\":\"" << JsonEscape(result.adapter_name)
                  << "\",\"encoder_name\":\"" << JsonEscape(result.encoder_name)
                  << "\",\"feature_level\":"
                  << static_cast<unsigned>(result.feature_level)
                  << ",\"encoded_bytes\":" << result.encoded_bytes << "}\n";
        return 0;
    } catch (const HResultError& error) {
        std::cout << "{\"ok\":false,\"error\":\"" << JsonEscape(error.what())
                  << "\",\"hresult\":" << static_cast<long>(error.result()) << "}\n";
        return 2;
    } catch (const std::exception& error) {
        std::cout << "{\"ok\":false,\"error\":\"" << JsonEscape(error.what())
                  << "\"}\n";
        return 2;
    }
}
