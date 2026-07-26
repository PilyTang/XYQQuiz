#pragma once

#include <cstdint>
#include <string>

struct ServerArguments {
    unsigned adapter = 0;
    bool auto_adapter = false;
    std::uint64_t hwnd = 0;
    std::wstring pipe_name;
    std::string token;
    std::wstring mapping_name;
    unsigned preview_width = 1024;
    unsigned preview_fps = 30;
    unsigned recognition_fps = 15;
    unsigned mapping_capacity = 3840U * 2160U * 4U;
    bool native_window = false;
};

int RunServer(const ServerArguments& arguments);
