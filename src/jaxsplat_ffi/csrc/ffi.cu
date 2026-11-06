// XLA FFI bindings for the jaxsplat CUDA kernels.
//
// Replaces the legacy `register_custom_call_target` glue of jaxsplat (removed
// in JAX 0.6) with the typed XLA FFI API. Differences to the original glue:
//   * Batched: every argument may carry leading batch dimensions (JAX vmap
//     with vmap_method="broadcast_all"). All images of a batch are projected,
//     sorted and rasterized in one launch per kernel, with one host sync.
//   * Temporary buffers (sorting, binning, CUB workspaces) come from XLA's
//     allocator instead of cudaMalloc/cudaFree.
//   * Only one host sync per rasterize call (to read the number of
//     tile/Gaussian intersections, which sizes the sort buffers).
//   * Kernels initialise their own outputs (few memsets).
//   * Gaussians with opacity < 1/255 are culled during projection.
//   * No intersections: the image is the background, transmittance is 1.

#include <algorithm>
#include <cstdint>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include <cub/cub.cuh>
#include <cuda_runtime.h>

#include "xla/ffi/api/ffi.h"

#include "common.h"
#include "kernels/kernels.h"

namespace ffi = xla::ffi;

namespace {

constexpr unsigned kThreads1D = 256;
// XLA's scratch allocator rejects explicit alignments ("Unsupported alignment"),
// but its GPU buffers are at least 256-byte aligned; we verify that below.
constexpr size_t kRequestedAlignment = 1;
constexpr size_t kRequiredAlignment = 16;

#define JS_CUDA_CHECK(expr)                                                    \
    do {                                                                       \
        cudaError_t err_ = (expr);                                             \
        if (err_ != cudaSuccess) {                                             \
            return ffi::Error::Internal(                                       \
                std::string("jaxsplat_ffi: ") + #expr + " failed: " +       \
                cudaGetErrorString(err_)                                       \
            );                                                                 \
        }                                                                      \
    } while (false)

#define JS_RETURN_IF_ERROR(expr)                                               \
    do {                                                                       \
        ffi::Error err_ = (expr);                                              \
        if (err_.failure()) {                                                  \
            return err_;                                                       \
        }                                                                      \
    } while (false)

inline unsigned NumBlocks(size_t n) {
    return static_cast<unsigned>((n + kThreads1D - 1) / kThreads1D);
}

struct Geometry {
    dim3 img_size;     // (W, H, 1)
    dim3 tile_bounds;  // number of tiles per image in (x, y)
    dim3 block_2d;     // (block_width, block_width, 1)
    unsigned block_width;
    int64_t batch;     // number of images

    size_t tiles_per_image() const {
        return size_t(tile_bounds.x) * tile_bounds.y;
    }
    size_t pixels_per_image() const { return size_t(img_size.x) * img_size.y; }
    // Rasterizer grid: one block per tile, blockIdx.z = image.
    dim3 raster_grid() const {
        return dim3(tile_bounds.x, tile_bounds.y, static_cast<unsigned>(batch));
    }
};

Geometry MakeGeometry(
    int32_t img_h, int32_t img_w, int32_t block_width, int64_t batch
) {
    const unsigned bw = static_cast<unsigned>(block_width);
    Geometry g;
    g.img_size = dim3(img_w, img_h, 1);
    g.tile_bounds = dim3((img_w + bw - 1) / bw, (img_h + bw - 1) / bw, 1);
    g.block_2d = dim3(bw, bw, 1);
    g.block_width = bw;
    g.batch = batch;
    return g;
}

ffi::Error CheckGeometry(int32_t img_h, int32_t img_w, int32_t block_width) {
    if (img_h <= 0 || img_w <= 0) {
        return ffi::Error::InvalidArgument("image shape must be positive");
    }
    if (block_width < 2 || block_width * block_width > MAX_GRID_DIM) {
        return ffi::Error::InvalidArgument(
            "block_width must satisfy 2 <= block_width <= 16"
        );
    }
    return ffi::Error::Success();
}

// Reads a (*batch, rows, cols) buffer: returns batch (product of leading
// dims) and rows.
template <ffi::DataType dtype>
ffi::Error BatchAndRows(
    const ffi::Buffer<dtype> &buf,
    int64_t cols,
    const char *name,
    int64_t *batch,
    int64_t *rows
) {
    auto dims = buf.dimensions();
    if (dims.size() < 2 || dims[dims.size() - 1] != cols) {
        return ffi::Error::InvalidArgument(
            std::string("jaxsplat_ffi: unexpected shape for ") + name
        );
    }
    *rows = dims[dims.size() - 2];
    *batch = 1;
    for (size_t i = 0; i + 2 < dims.size(); ++i) {
        *batch *= dims[i];
    }
    return ffi::Error::Success();
}

// Checks the element count of a buffer (shapes were broadcast by JAX).
template <ffi::DataType dtype>
ffi::Error CheckCount(const ffi::Buffer<dtype> &buf, size_t count, const char *name) {
    if (buf.element_count() != count) {
        return ffi::Error::InvalidArgument(
            std::string("jaxsplat_ffi: unexpected size for ") + name +
            ": expected " + std::to_string(count) + " elements, got " +
            std::to_string(buf.element_count())
        );
    }
    return ffi::Error::Success();
}

template <ffi::DataType dtype>
ffi::Error CheckCount(
    const ffi::Result<ffi::Buffer<dtype>> &buf, size_t count, const char *name
) {
    if (buf->element_count() != count) {
        return ffi::Error::InvalidArgument(
            std::string("jaxsplat_ffi: unexpected size for output ") + name
        );
    }
    return ffi::Error::Success();
}

}  // namespace

// Device scratch memory from XLA's allocator, freed when the handler returns.
// Same as ffi::ScratchAllocator, but reports XLA's error message on failure
// (ffi::ScratchAllocator only writes it to a diagnostic that is never shown).
class Scratch {
  public:
    Scratch(const XLA_FFI_Api *api, XLA_FFI_InvokeContext *ctx)
        : api_(api), ctx_(ctx) {}
    Scratch(Scratch &&other) noexcept
        : api_(other.api_), ctx_(other.ctx_),
          allocations_(std::move(other.allocations_)) {
        other.allocations_.clear();
    }
    Scratch(const Scratch &) = delete;
    Scratch &operator=(const Scratch &) = delete;

    ~Scratch() {
        for (auto &[size, data] : allocations_) {
            XLA_FFI_DeviceMemory_Free_Args args;
            args.struct_size = XLA_FFI_DeviceMemory_Free_Args_STRUCT_SIZE;
            args.extension_start = nullptr;
            args.ctx = ctx_;
            args.size = size;
            args.data = data;
            if (XLA_FFI_Error *err = api_->XLA_FFI_DeviceMemory_Free(&args)) {
                ffi::internal::DestroyError(api_, err);
            }
        }
    }

    ffi::Error Allocate(size_t size, void **out) {
        XLA_FFI_DeviceMemory_Allocate_Args args;
        args.struct_size = XLA_FFI_DeviceMemory_Allocate_Args_STRUCT_SIZE;
        args.extension_start = nullptr;
        args.ctx = ctx_;
        args.size = size;
        args.alignment = kRequestedAlignment;
        args.data = nullptr;
        if (XLA_FFI_Error *err = api_->XLA_FFI_DeviceMemory_Allocate(&args)) {
            std::string msg = ffi::internal::GetErrorMessage(api_, err);
            ffi::internal::DestroyError(api_, err);
            return ffi::Error::Internal(
                "jaxsplat_ffi: scratch allocation of " +
                std::to_string(size) + " bytes failed: " + msg
            );
        }
        allocations_.emplace_back(size, args.data);
        if (reinterpret_cast<uintptr_t>(args.data) % kRequiredAlignment != 0) {
            return ffi::Error::Internal(
                "jaxsplat_ffi: scratch memory is not 16-byte aligned"
            );
        }
        *out = args.data;
        return ffi::Error::Success();
    }

  private:
    const XLA_FFI_Api *api_;
    XLA_FFI_InvokeContext *ctx_;
    std::vector<std::pair<size_t, void *>> allocations_;
};

template <>
struct xla::ffi::CtxDecoding<Scratch> {
    using Type = Scratch;
    static std::optional<Type> Decode(
        const XLA_FFI_Api *api,
        XLA_FFI_InvokeContext *ctx,
        DiagnosticEngine &
    ) {
        return Scratch(api, ctx);
    }
};

namespace {

template <typename T>
ffi::Error Allocate(Scratch &scratch, size_t count, T **out) {
    *out = nullptr;
    if (count == 0) {
        return ffi::Error::Success();
    }
    void *mem = nullptr;
    JS_RETURN_IF_ERROR(scratch.Allocate(count * sizeof(T), &mem));
    *out = static_cast<T *>(mem);
    return ffi::Error::Success();
}

template <typename T>
ffi::Error Zero(cudaStream_t stream, T *ptr, size_t count) {
    if (count > 0) {
        JS_CUDA_CHECK(cudaMemsetAsync(ptr, 0, count * sizeof(T), stream));
    }
    return ffi::Error::Success();
}

// Copies one int32 from the device to the host and waits for it.
// Goes through a pinned (page-locked) buffer: a copy into ordinary pageable
// memory is staged by the driver and kept the stream blocked for 0.5-2 ms
// afterwards, which was most of the GPU idle time in a training step.
ffi::Error ReadDeviceInt(cudaStream_t stream, const int32_t *src, int32_t *dst) {
    thread_local int32_t *pinned = nullptr;
    if (pinned == nullptr) {
        JS_CUDA_CHECK(cudaMallocHost(&pinned, sizeof(int32_t)));
    }
    JS_CUDA_CHECK(cudaMemcpyAsync(
        pinned, src, sizeof(int32_t), cudaMemcpyDeviceToHost, stream
    ));
    JS_CUDA_CHECK(cudaStreamSynchronize(stream));
    *dst = *pinned;
    return ffi::Error::Success();
}

// Counts tile/Gaussian intersections of all images: per-Gaussian tile counts,
// their running sum (offsets into the intersection list) and the total. The
// total decides the size of the sort buffers, so it needs one host sync.
ffi::Error CountIntersections(
    cudaStream_t stream,
    Scratch &scratch,
    const Geometry &g,
    int64_t num_points,
    const float2 *xys,
    const int32_t *radii,
    int32_t **cum_tiles_hit,
    int32_t *num_intersects
) {
    *num_intersects = 0;
    *cum_tiles_hit = nullptr;
    if (num_points == 0) {
        return ffi::Error::Success();
    }
    int32_t *num_tiles_hit;
    JS_RETURN_IF_ERROR(Allocate(scratch, num_points, &num_tiles_hit));
    JS_RETURN_IF_ERROR(Allocate(scratch, num_points, cum_tiles_hit));
    kernels::count_tiles_hit<<<NumBlocks(num_points), kThreads1D, 0, stream>>>(
        static_cast<int>(num_points), xys, radii, g.tile_bounds, g.block_width,
        num_tiles_hit
    );
    JS_CUDA_CHECK(cudaGetLastError());

    size_t ws_bytes = 0;
    JS_CUDA_CHECK(cub::DeviceScan::InclusiveSum(
        nullptr, ws_bytes, num_tiles_hit, *cum_tiles_hit,
        static_cast<int>(num_points), stream
    ));
    uint8_t *ws;
    JS_RETURN_IF_ERROR(Allocate(scratch, ws_bytes, &ws));
    JS_CUDA_CHECK(cub::DeviceScan::InclusiveSum(
        ws, ws_bytes, num_tiles_hit, *cum_tiles_hit,
        static_cast<int>(num_points), stream
    ));

    JS_RETURN_IF_ERROR(
        ReadDeviceInt(stream, *cum_tiles_hit + num_points - 1, num_intersects)
    );
    if (*num_intersects < 0) {
        return ffi::Error::Internal(
            "jaxsplat_ffi: negative intersection count (int32 overflow?)"
        );
    }
    return ffi::Error::Success();
}

// Running sum of tiles hit per Gaussian (offsets into the intersection list)
// and the total. For one image the projection's cum_tiles_hit is used as is.
// For a batch it is recounted: if JAX broadcast one projection to several
// images, the projection's sums restart in every image and would be wrong.
ffi::Error IntersectionOffsets(
    cudaStream_t stream,
    Scratch &scratch,
    const Geometry &g,
    int64_t num_points,
    const float2 *xys,
    const int32_t *radii,
    const int32_t *projected_cum_tiles_hit,
    const int32_t **cum_tiles_hit,
    int32_t *num_intersects
) {
    if (g.batch == 1) {
        *cum_tiles_hit = projected_cum_tiles_hit;
        *num_intersects = 0;
        if (num_points == 0) {
            return ffi::Error::Success();
        }
        JS_RETURN_IF_ERROR(ReadDeviceInt(
            stream, projected_cum_tiles_hit + num_points - 1, num_intersects
        ));
        return ffi::Error::Success();
    }
    int32_t *recounted;
    JS_RETURN_IF_ERROR(CountIntersections(
        stream, scratch, g, num_points, xys, radii, &recounted, num_intersects
    ));
    *cum_tiles_hit = recounted;
    return ffi::Error::Success();
}

// Builds (tile_id | depth) keys for every tile/Gaussian intersection of all
// images, sorts them and finds the [start, end) range of every tile.
ffi::Error SortAndBin(
    cudaStream_t stream,
    Scratch &scratch,
    const Geometry &g,
    int64_t num_points,        // all images
    int64_t points_per_image,
    int32_t num_intersects,
    const float2 *xys,
    const float *depths,
    const int32_t *radii,
    const int32_t *cum_tiles_hit,
    int32_t **gaussian_ids_sorted,
    int2 **tile_bins
) {
    const size_t num_tiles = g.tiles_per_image() * g.batch;

    int32_t *gaussian_ids_unsorted;
    int64_t *isect_ids_unsorted;
    int64_t *isect_ids_sorted;
    JS_RETURN_IF_ERROR(Allocate(scratch, num_intersects, &gaussian_ids_unsorted));
    JS_RETURN_IF_ERROR(Allocate(scratch, num_intersects, &isect_ids_unsorted));
    JS_RETURN_IF_ERROR(Allocate(scratch, num_intersects, &isect_ids_sorted));
    JS_RETURN_IF_ERROR(Allocate(scratch, num_intersects, gaussian_ids_sorted));
    JS_RETURN_IF_ERROR(Allocate(scratch, num_tiles, tile_bins));
    JS_RETURN_IF_ERROR(Zero(stream, *tile_bins, num_tiles));  // empty tiles = (0, 0)

    kernels::map_gaussian_to_intersects<<<
        NumBlocks(num_points), kThreads1D, 0, stream>>>(
        static_cast<int>(num_points),
        static_cast<int>(points_per_image),
        xys,
        depths,
        radii,
        cum_tiles_hit,
        g.tile_bounds,
        g.block_width,
        isect_ids_unsorted,
        gaussian_ids_unsorted
    );
    JS_CUDA_CHECK(cudaGetLastError());

    // Keys are (tile_id << 32 | depth bits); only sort the bits in use.
    const int32_t max_tile_id = static_cast<int32_t>(num_tiles);
    const int msb = 32 - __builtin_clz(max_tile_id) + 1;
    const int end_bit = std::min(64, 32 + msb);

    size_t sort_ws_bytes = 0;
    JS_CUDA_CHECK(cub::DeviceRadixSort::SortPairs(
        nullptr, sort_ws_bytes,
        isect_ids_unsorted, isect_ids_sorted,
        gaussian_ids_unsorted, *gaussian_ids_sorted,
        num_intersects, 0, end_bit, stream
    ));
    uint8_t *sort_ws;
    JS_RETURN_IF_ERROR(Allocate(scratch, sort_ws_bytes, &sort_ws));
    JS_CUDA_CHECK(cub::DeviceRadixSort::SortPairs(
        sort_ws, sort_ws_bytes,
        isect_ids_unsorted, isect_ids_sorted,
        gaussian_ids_unsorted, *gaussian_ids_sorted,
        num_intersects, 0, end_bit, stream
    ));

    kernels::get_tile_bin_edges<<<
        NumBlocks(num_intersects), kThreads1D, 0, stream>>>(
        num_intersects, isect_ids_sorted, *tile_bins
    );
    JS_CUDA_CHECK(cudaGetLastError());
    return ffi::Error::Success();
}

}  // namespace

// ---------------------------------------------------------------------------
// project forward
// ---------------------------------------------------------------------------

ffi::Error ProjectFwdImpl(
    cudaStream_t stream,
    Scratch scratch,
    float glob_scale,
    float fx, float fy, float cx, float cy,
    int32_t img_h, int32_t img_w, int32_t block_width,
    float clip_thresh,
    ffi::Buffer<ffi::F32> means3d,
    ffi::Buffer<ffi::F32> scales,
    ffi::Buffer<ffi::F32> quats,
    ffi::Buffer<ffi::F32> viewmat,
    ffi::Buffer<ffi::F32> opacities,
    ffi::ResultBuffer<ffi::F32> cov3ds,
    ffi::ResultBuffer<ffi::F32> xys,
    ffi::ResultBuffer<ffi::F32> depths,
    ffi::ResultBuffer<ffi::S32> radii,
    ffi::ResultBuffer<ffi::F32> conics,
    ffi::ResultBuffer<ffi::F32> compensation,
    ffi::ResultBuffer<ffi::S32> num_tiles_hit,
    ffi::ResultBuffer<ffi::S32> cum_tiles_hit
) {
    JS_RETURN_IF_ERROR(CheckGeometry(img_h, img_w, block_width));
    int64_t batch, n;
    JS_RETURN_IF_ERROR(BatchAndRows(means3d, 3, "means3d", &batch, &n));
    const size_t total = size_t(batch) * n;
    JS_RETURN_IF_ERROR(CheckCount(scales, total * 3, "scales"));
    JS_RETURN_IF_ERROR(CheckCount(quats, total * 4, "quats"));
    JS_RETURN_IF_ERROR(CheckCount(viewmat, size_t(batch) * 16, "viewmat"));
    JS_RETURN_IF_ERROR(CheckCount(opacities, total, "opacities"));
    JS_RETURN_IF_ERROR(CheckCount(cov3ds, total * 6, "cov3ds"));
    if (total == 0) {
        return ffi::Error::Success();
    }

    const Geometry g = MakeGeometry(img_h, img_w, block_width, batch);
    kernels::project_gaussians_fwd<<<NumBlocks(total), kThreads1D, 0, stream>>>(
        static_cast<int>(total),
        static_cast<int>(n),
        reinterpret_cast<const float3 *>(means3d.typed_data()),
        reinterpret_cast<const float3 *>(scales.typed_data()),
        glob_scale,
        reinterpret_cast<const float4 *>(quats.typed_data()),
        viewmat.typed_data(),
        make_float4(fx, fy, cx, cy),
        g.img_size,
        g.tile_bounds,
        g.block_width,
        clip_thresh,
        opacities.typed_data(),
        cov3ds->typed_data(),
        reinterpret_cast<float2 *>(xys->typed_data()),
        depths->typed_data(),
        radii->typed_data(),
        reinterpret_cast<float3 *>(conics->typed_data()),
        compensation->typed_data(),
        num_tiles_hit->typed_data()
    );
    JS_CUDA_CHECK(cudaGetLastError());

    // Cumulative sum over all images: global offsets into the intersection list.
    size_t ws_bytes = 0;
    JS_CUDA_CHECK(cub::DeviceScan::InclusiveSum(
        nullptr, ws_bytes, num_tiles_hit->typed_data(),
        cum_tiles_hit->typed_data(), static_cast<int>(total), stream
    ));
    uint8_t *ws;
    JS_RETURN_IF_ERROR(Allocate(scratch, ws_bytes, &ws));
    JS_CUDA_CHECK(cub::DeviceScan::InclusiveSum(
        ws, ws_bytes, num_tiles_hit->typed_data(),
        cum_tiles_hit->typed_data(), static_cast<int>(total), stream
    ));
    return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    JaxsplatProjectFwd, ProjectFwdImpl,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Ctx<Scratch>()
        .Attr<float>("glob_scale")
        .Attr<float>("fx")
        .Attr<float>("fy")
        .Attr<float>("cx")
        .Attr<float>("cy")
        .Attr<int32_t>("img_h")
        .Attr<int32_t>("img_w")
        .Attr<int32_t>("block_width")
        .Attr<float>("clip_thresh")
        .Arg<ffi::Buffer<ffi::F32>>()  // means3d (*B, N, 3)
        .Arg<ffi::Buffer<ffi::F32>>()  // scales (*B, N, 3)
        .Arg<ffi::Buffer<ffi::F32>>()  // quats (*B, N, 4)
        .Arg<ffi::Buffer<ffi::F32>>()  // viewmat (*B, 4, 4)
        .Arg<ffi::Buffer<ffi::F32>>()  // opacities (*B, N, 1), only for culling
        .Ret<ffi::Buffer<ffi::F32>>()  // cov3ds (*B, N, 6)
        .Ret<ffi::Buffer<ffi::F32>>()  // xys (*B, N, 2)
        .Ret<ffi::Buffer<ffi::F32>>()  // depths (*B, N, 1)
        .Ret<ffi::Buffer<ffi::S32>>()  // radii (*B, N, 1)
        .Ret<ffi::Buffer<ffi::F32>>()  // conics (*B, N, 3)
        .Ret<ffi::Buffer<ffi::F32>>()  // compensation (*B, N, 1)
        .Ret<ffi::Buffer<ffi::S32>>()  // num_tiles_hit (*B, N, 1)
        .Ret<ffi::Buffer<ffi::S32>>()  // cum_tiles_hit (*B, N, 1), over all images
);

// ---------------------------------------------------------------------------
// project backward
// ---------------------------------------------------------------------------

ffi::Error ProjectBwdImpl(
    cudaStream_t stream,
    float glob_scale,
    float fx, float fy, float cx, float cy,
    int32_t img_h, int32_t img_w,
    ffi::Buffer<ffi::F32> means3d,
    ffi::Buffer<ffi::F32> scales,
    ffi::Buffer<ffi::F32> quats,
    ffi::Buffer<ffi::F32> viewmat,
    ffi::Buffer<ffi::F32> cov3ds,
    ffi::Buffer<ffi::S32> radii,
    ffi::Buffer<ffi::F32> conics,
    ffi::Buffer<ffi::F32> compensation,
    ffi::Buffer<ffi::F32> v_xy,
    ffi::Buffer<ffi::F32> v_depth,
    ffi::Buffer<ffi::F32> v_conic,
    ffi::Buffer<ffi::F32> v_compensation,
    ffi::ResultBuffer<ffi::F32> v_cov2d,
    ffi::ResultBuffer<ffi::F32> v_cov3d,
    ffi::ResultBuffer<ffi::F32> v_mean3d,
    ffi::ResultBuffer<ffi::F32> v_scale,
    ffi::ResultBuffer<ffi::F32> v_quat
) {
    int64_t batch, n;
    JS_RETURN_IF_ERROR(BatchAndRows(means3d, 3, "means3d", &batch, &n));
    const size_t total = size_t(batch) * n;
    JS_RETURN_IF_ERROR(CheckCount(viewmat, size_t(batch) * 16, "viewmat"));
    JS_RETURN_IF_ERROR(CheckCount(v_xy, total * 2, "v_xy"));
    JS_RETURN_IF_ERROR(CheckCount(v_conic, total * 3, "v_conic"));
    if (total == 0) {
        return ffi::Error::Success();
    }

    kernels::project_gaussians_bwd<<<NumBlocks(total), kThreads1D, 0, stream>>>(
        static_cast<int>(total),
        static_cast<int>(n),
        reinterpret_cast<const float3 *>(means3d.typed_data()),
        reinterpret_cast<const float3 *>(scales.typed_data()),
        glob_scale,
        reinterpret_cast<const float4 *>(quats.typed_data()),
        viewmat.typed_data(),
        make_float4(fx, fy, cx, cy),
        dim3(img_w, img_h, 1),
        cov3ds.typed_data(),
        radii.typed_data(),
        reinterpret_cast<const float3 *>(conics.typed_data()),
        compensation.typed_data(),
        reinterpret_cast<const float2 *>(v_xy.typed_data()),
        v_depth.typed_data(),
        reinterpret_cast<const float3 *>(v_conic.typed_data()),
        v_compensation.typed_data(),
        reinterpret_cast<float3 *>(v_cov2d->typed_data()),
        v_cov3d->typed_data(),
        reinterpret_cast<float3 *>(v_mean3d->typed_data()),
        reinterpret_cast<float3 *>(v_scale->typed_data()),
        reinterpret_cast<float4 *>(v_quat->typed_data())
    );
    JS_CUDA_CHECK(cudaGetLastError());
    return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    JaxsplatProjectBwd, ProjectBwdImpl,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Attr<float>("glob_scale")
        .Attr<float>("fx")
        .Attr<float>("fy")
        .Attr<float>("cx")
        .Attr<float>("cy")
        .Attr<int32_t>("img_h")
        .Attr<int32_t>("img_w")
        .Arg<ffi::Buffer<ffi::F32>>()  // means3d
        .Arg<ffi::Buffer<ffi::F32>>()  // scales
        .Arg<ffi::Buffer<ffi::F32>>()  // quats
        .Arg<ffi::Buffer<ffi::F32>>()  // viewmat
        .Arg<ffi::Buffer<ffi::F32>>()  // cov3ds
        .Arg<ffi::Buffer<ffi::S32>>()  // radii
        .Arg<ffi::Buffer<ffi::F32>>()  // conics
        .Arg<ffi::Buffer<ffi::F32>>()  // compensation
        .Arg<ffi::Buffer<ffi::F32>>()  // v_xy
        .Arg<ffi::Buffer<ffi::F32>>()  // v_depth
        .Arg<ffi::Buffer<ffi::F32>>()  // v_conic
        .Arg<ffi::Buffer<ffi::F32>>()  // v_compensation
        .Ret<ffi::Buffer<ffi::F32>>()  // v_cov2d (*B, N, 3)
        .Ret<ffi::Buffer<ffi::F32>>()  // v_cov3d (*B, N, 6)
        .Ret<ffi::Buffer<ffi::F32>>()  // v_mean3d (*B, N, 3)
        .Ret<ffi::Buffer<ffi::F32>>()  // v_scale (*B, N, 3)
        .Ret<ffi::Buffer<ffi::F32>>()  // v_quat (*B, N, 4)
);

// ---------------------------------------------------------------------------
// rasterize forward
// ---------------------------------------------------------------------------

ffi::Error RasterizeFwdImpl(
    cudaStream_t stream,
    Scratch scratch,
    int32_t img_h, int32_t img_w, int32_t block_width,
    ffi::Buffer<ffi::F32> colors,
    ffi::Buffer<ffi::F32> opacities,
    ffi::Buffer<ffi::F32> background,
    ffi::Buffer<ffi::F32> xys,
    ffi::Buffer<ffi::F32> depths,
    ffi::Buffer<ffi::S32> radii,
    ffi::Buffer<ffi::F32> conics,
    ffi::Buffer<ffi::S32> cum_tiles_hit,
    ffi::ResultBuffer<ffi::F32> final_Ts,
    ffi::ResultBuffer<ffi::S32> final_idx,
    ffi::ResultBuffer<ffi::F32> out_img
) {
    JS_RETURN_IF_ERROR(CheckGeometry(img_h, img_w, block_width));
    int64_t batch, n;
    JS_RETURN_IF_ERROR(BatchAndRows(colors, 3, "colors", &batch, &n));
    const size_t total = size_t(batch) * n;
    JS_RETURN_IF_ERROR(CheckCount(opacities, total, "opacities"));
    JS_RETURN_IF_ERROR(CheckCount(xys, total * 2, "xys"));
    JS_RETURN_IF_ERROR(CheckCount(background, size_t(batch) * 3, "background"));
    const Geometry g = MakeGeometry(img_h, img_w, block_width, batch);
    const size_t num_pixels = g.pixels_per_image() * batch;
    JS_RETURN_IF_ERROR(CheckCount(out_img, num_pixels * 3, "img"));
    const float3 *bg = reinterpret_cast<const float3 *>(background.typed_data());

    int32_t num_intersects = 0;
    const int32_t *cum_hits;
    JS_RETURN_IF_ERROR(IntersectionOffsets(
        stream, scratch, g, total,
        reinterpret_cast<const float2 *>(xys.typed_data()), radii.typed_data(),
        cum_tiles_hit.typed_data(), &cum_hits, &num_intersects
    ));

    if (num_intersects == 0) {
        // Nothing visible: image = background, transmittance = 1.
        JS_RETURN_IF_ERROR(Zero(stream, final_idx->typed_data(), num_pixels));
        kernels::fill_background<<<NumBlocks(num_pixels), kThreads1D, 0, stream>>>(
            reinterpret_cast<float3 *>(out_img->typed_data()),
            final_Ts->typed_data(),
            g.pixels_per_image(),
            batch,
            bg
        );
        JS_CUDA_CHECK(cudaGetLastError());
        return ffi::Error::Success();
    }

    int32_t *gaussian_ids_sorted;
    int2 *tile_bins;
    JS_RETURN_IF_ERROR(SortAndBin(
        stream, scratch, g, total, n, num_intersects,
        reinterpret_cast<const float2 *>(xys.typed_data()),
        depths.typed_data(), radii.typed_data(), cum_hits,
        &gaussian_ids_sorted, &tile_bins
    ));

    // Writes every pixel of every image (final_Ts, final_idx, out_img).
    kernels::rasterize_fwd<<<g.raster_grid(), g.block_2d, 0, stream>>>(
        g.tile_bounds,
        g.img_size,
        gaussian_ids_sorted,
        tile_bins,
        reinterpret_cast<const float2 *>(xys.typed_data()),
        reinterpret_cast<const float3 *>(conics.typed_data()),
        reinterpret_cast<const float3 *>(colors.typed_data()),
        opacities.typed_data(),
        final_Ts->typed_data(),
        final_idx->typed_data(),
        reinterpret_cast<float3 *>(out_img->typed_data()),
        bg
    );
    JS_CUDA_CHECK(cudaGetLastError());
    return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    JaxsplatRasterizeFwd, RasterizeFwdImpl,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Ctx<Scratch>()
        .Attr<int32_t>("img_h")
        .Attr<int32_t>("img_w")
        .Attr<int32_t>("block_width")
        .Arg<ffi::Buffer<ffi::F32>>()  // colors (*B, N, 3)
        .Arg<ffi::Buffer<ffi::F32>>()  // opacities (*B, N, 1)
        .Arg<ffi::Buffer<ffi::F32>>()  // background (*B, 3)
        .Arg<ffi::Buffer<ffi::F32>>()  // xys (*B, N, 2)
        .Arg<ffi::Buffer<ffi::F32>>()  // depths (*B, N, 1)
        .Arg<ffi::Buffer<ffi::S32>>()  // radii (*B, N, 1)
        .Arg<ffi::Buffer<ffi::F32>>()  // conics (*B, N, 3)
        .Arg<ffi::Buffer<ffi::S32>>()  // cum_tiles_hit (unused, kept for the API)
        .Ret<ffi::Buffer<ffi::F32>>()  // final_Ts (*B, H, W, 1)
        .Ret<ffi::Buffer<ffi::S32>>()  // final_idx (*B, H, W, 1)
        .Ret<ffi::Buffer<ffi::F32>>()  // out_img (*B, H, W, 3)
);

// ---------------------------------------------------------------------------
// rasterize backward
// ---------------------------------------------------------------------------

ffi::Error RasterizeBwdImpl(
    cudaStream_t stream,
    Scratch scratch,
    int32_t img_h, int32_t img_w, int32_t block_width,
    ffi::Buffer<ffi::F32> colors,
    ffi::Buffer<ffi::F32> opacities,
    ffi::Buffer<ffi::F32> background,
    ffi::Buffer<ffi::F32> xys,
    ffi::Buffer<ffi::F32> depths,
    ffi::Buffer<ffi::S32> radii,
    ffi::Buffer<ffi::F32> conics,
    ffi::Buffer<ffi::S32> cum_tiles_hit,
    ffi::Buffer<ffi::F32> final_Ts,
    ffi::Buffer<ffi::S32> final_idx,
    ffi::Buffer<ffi::F32> v_img,
    ffi::Buffer<ffi::F32> v_img_alpha,
    ffi::ResultBuffer<ffi::F32> v_colors,
    ffi::ResultBuffer<ffi::F32> v_opacity,
    ffi::ResultBuffer<ffi::F32> v_xy,
    ffi::ResultBuffer<ffi::F32> v_conic
) {
    JS_RETURN_IF_ERROR(CheckGeometry(img_h, img_w, block_width));
    int64_t batch, n;
    JS_RETURN_IF_ERROR(BatchAndRows(colors, 3, "colors", &batch, &n));
    const size_t total = size_t(batch) * n;
    JS_RETURN_IF_ERROR(CheckCount(background, size_t(batch) * 3, "background"));
    const Geometry g = MakeGeometry(img_h, img_w, block_width, batch);
    const size_t num_pixels = g.pixels_per_image() * batch;
    JS_RETURN_IF_ERROR(CheckCount(v_img, num_pixels * 3, "v_img"));
    JS_RETURN_IF_ERROR(CheckCount(v_img_alpha, num_pixels, "v_img_alpha"));

    // The kernel accumulates with atomicAdd, so all outputs start at zero.
    JS_RETURN_IF_ERROR(Zero(stream, v_colors->typed_data(), total * 3));
    JS_RETURN_IF_ERROR(Zero(stream, v_opacity->typed_data(), total));
    JS_RETURN_IF_ERROR(Zero(stream, v_xy->typed_data(), total * 2));
    JS_RETURN_IF_ERROR(Zero(stream, v_conic->typed_data(), total * 3));

    int32_t num_intersects = 0;
    const int32_t *cum_hits;
    JS_RETURN_IF_ERROR(IntersectionOffsets(
        stream, scratch, g, total,
        reinterpret_cast<const float2 *>(xys.typed_data()), radii.typed_data(),
        cum_tiles_hit.typed_data(), &cum_hits, &num_intersects
    ));
    if (num_intersects == 0) {
        return ffi::Error::Success();
    }

    int32_t *gaussian_ids_sorted;
    int2 *tile_bins;
    JS_RETURN_IF_ERROR(SortAndBin(
        stream, scratch, g, total, n, num_intersects,
        reinterpret_cast<const float2 *>(xys.typed_data()),
        depths.typed_data(), radii.typed_data(), cum_hits,
        &gaussian_ids_sorted, &tile_bins
    ));

    kernels::rasterize_bwd<<<g.raster_grid(), g.block_2d, 0, stream>>>(
        g.tile_bounds,
        g.img_size,
        gaussian_ids_sorted,
        tile_bins,
        reinterpret_cast<const float2 *>(xys.typed_data()),
        reinterpret_cast<const float3 *>(conics.typed_data()),
        reinterpret_cast<const float3 *>(colors.typed_data()),
        opacities.typed_data(),
        reinterpret_cast<const float3 *>(background.typed_data()),
        final_Ts.typed_data(),
        final_idx.typed_data(),
        reinterpret_cast<const float3 *>(v_img.typed_data()),
        v_img_alpha.typed_data(),
        reinterpret_cast<float2 *>(v_xy->typed_data()),
        reinterpret_cast<float3 *>(v_conic->typed_data()),
        reinterpret_cast<float3 *>(v_colors->typed_data()),
        v_opacity->typed_data()
    );
    JS_CUDA_CHECK(cudaGetLastError());
    return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    JaxsplatRasterizeBwd, RasterizeBwdImpl,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Ctx<Scratch>()
        .Attr<int32_t>("img_h")
        .Attr<int32_t>("img_w")
        .Attr<int32_t>("block_width")
        .Arg<ffi::Buffer<ffi::F32>>()  // colors
        .Arg<ffi::Buffer<ffi::F32>>()  // opacities
        .Arg<ffi::Buffer<ffi::F32>>()  // background
        .Arg<ffi::Buffer<ffi::F32>>()  // xys
        .Arg<ffi::Buffer<ffi::F32>>()  // depths
        .Arg<ffi::Buffer<ffi::S32>>()  // radii
        .Arg<ffi::Buffer<ffi::F32>>()  // conics
        .Arg<ffi::Buffer<ffi::S32>>()  // cum_tiles_hit (unused, kept for the API)
        .Arg<ffi::Buffer<ffi::F32>>()  // final_Ts
        .Arg<ffi::Buffer<ffi::S32>>()  // final_idx
        .Arg<ffi::Buffer<ffi::F32>>()  // v_img (*B, H, W, 3)
        .Arg<ffi::Buffer<ffi::F32>>()  // v_img_alpha (*B, H, W, 1)
        .Ret<ffi::Buffer<ffi::F32>>()  // v_colors (*B, N, 3)
        .Ret<ffi::Buffer<ffi::F32>>()  // v_opacity (*B, N, 1)
        .Ret<ffi::Buffer<ffi::F32>>()  // v_xy (*B, N, 2)
        .Ret<ffi::Buffer<ffi::F32>>()  // v_conic (*B, N, 3)
);
