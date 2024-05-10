#pragma once

#include <pybind11/pybind11.h>

#include <cstddef>
#include <string>

namespace py = pybind11;

template <typename T> py::capsule encapsulate_function(T *fn) {
    return pybind11::capsule(
        reinterpret_cast<void *>(fn),
        "xla._CUSTOM_CALL_TARGET"
    );
}

template <typename T>
const T *unpack_descriptor(const char *opaque, std::size_t opaque_len) {
    if (opaque_len != sizeof(T)) {
        throw std::runtime_error("Invalid opaque object size");
    }
    return reinterpret_cast<const T *>(opaque);
}

template <typename T> py::bytes pack_descriptor(const T &descriptor) {
    const std::string str =
        std::string(reinterpret_cast<const char *>(&descriptor), sizeof(T));
    return py::bytes(str);
}
