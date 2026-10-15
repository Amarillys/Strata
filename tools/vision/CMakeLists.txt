# tools/vision - strata-vision, the image encoder for Strata's multimodal path (optional; setup.py builds it when
# the vision encoder is chosen).  Uses llama.cpp's mtmd library at the same pinned commit as the engine's ggml.
#
#   cmake -S tools/vision -B build-vision -G Ninja -DCMAKE_BUILD_TYPE=Release -DLLAMA_DIR=third_party/llama.cpp
#         [-DSTRATA_VISION_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=native] [-DSTRATA_PORTABLE=OFF]
#         [-DSTRATA_VISION_VULKAN=ON]  # Intel iGPU; can be built without CUDA
#         [-DSTRATA_VISION_WEIGHTS_HOST=ON]  # MTMD_WEIGHTS_HOST: the projector's weights in RAM (see the end)
#   cmake --build build-vision --target strata-vision
#
# Portable (AVX2 baseline) is the default, so an encoder built without extra flags runs on any AVX2 PC: the 0.1.32
# release's encoder was built native on an AVX-512 machine and stopped with an illegal instruction elsewhere
# (#411 #412 #419).  setup.py, which builds for the PC it runs on, passes -DSTRATA_PORTABLE=OFF.  The option is
# kept in CMakeCache.txt, so a build folder configured before this default still needs the flag (or a fresh folder).
cmake_minimum_required(VERSION 3.21)
project(strata_vision C CXX)

set(LLAMA_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../third_party/llama.cpp" CACHE PATH "llama.cpp checkout")
option(STRATA_VISION_CUDA "run the vision encoder on the GPU (builds ggml-cuda)" OFF)
option(STRATA_VISION_VULKAN "build the Vulkan image encoder backend (including Intel integrated GPUs)" OFF)
option(STRATA_PORTABLE "build for other PCs: AVX2 baseline, static MSVC runtime" ON)
option(STRATA_VISION_WEIGHTS_HOST "allow MTMD_WEIGHTS_HOST=cpu|pinned: the projector's weights stay in RAM and only the compute runs on the GPU (builds a patched copy of mtmd's clip.cpp)" OFF)
if(STRATA_PORTABLE AND MSVC)
  set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded")
endif()

set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_COMMON OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_TOOLS OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_SERVER OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_APP OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_MTMD ON CACHE BOOL "" FORCE)
set(LLAMA_OPENSSL OFF CACHE BOOL "" FORCE)
set(LLAMA_CURL OFF CACHE BOOL "" FORCE)
if(STRATA_PORTABLE)
  set(GGML_NATIVE OFF CACHE BOOL "" FORCE)
  set(GGML_OPENMP OFF CACHE BOOL "" FORCE)        # no vcomp140.dll / libgomp needed on the user's PC
  foreach(_f GGML_AVX GGML_AVX2 GGML_FMA GGML_F16C GGML_BMI2)
    set(${_f} ON CACHE BOOL "" FORCE)
  endforeach()
  # nothing above AVX2 either, even when a -D asks for it (a 9900K or a Zen 3 has neither AVX-512 nor AVX-VNNI)
  foreach(_f GGML_AVX512 GGML_AVX512_VBMI GGML_AVX512_VNNI GGML_AVX512_BF16 GGML_AVX_VNNI
             GGML_AMX_TILE GGML_AMX_INT8 GGML_AMX_BF16)
    set(${_f} OFF CACHE BOOL "" FORCE)
  endforeach()
  message(STATUS "strata-vision: portable build (AVX2 baseline, runs on other PCs)")
else()
  # this PC's instruction set; a build folder that was portable before gets a fresh native build's settings back
  set(GGML_NATIVE ON CACHE BOOL "" FORCE)
  set(GGML_OPENMP ON CACHE BOOL "" FORCE)
  foreach(_f GGML_AVX GGML_AVX2 GGML_FMA GGML_F16C GGML_BMI2)
    set(${_f} OFF CACHE BOOL "" FORCE)
  endforeach()
  message(STATUS "strata-vision: native build for this CPU (STRATA_PORTABLE=OFF) - not for other PCs")
endif()
set(GGML_BACKEND_DL OFF CACHE BOOL "" FORCE)
set(GGML_CUDA ${STRATA_VISION_CUDA} CACHE BOOL "" FORCE)
set(GGML_VULKAN ${STRATA_VISION_VULKAN} CACHE BOOL "" FORCE)

add_subdirectory(${LLAMA_DIR} llama EXCLUDE_FROM_ALL)

add_executable(strata-vision strata_vision.cpp)
target_link_libraries(strata-vision PRIVATE mtmd llama)
target_compile_features(strata-vision PRIVATE cxx_std_17)
set_target_properties(strata-vision PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# Opt-in (-DSTRATA_VISION_WEIGHTS_HOST=ON): mtmd is built from a copy of clip.cpp in this build folder with one change.
# With MTMD_WEIGHTS_HOST set in the encoder's environment ("cpu"; "pinned" is ggml-vulkan's pinned buffer, which belongs
# to Vulkan device 0) the projector's weights are allocated in RAM instead of on the GPU.  mtmd's scheduler already
# offloads the large ops (op_offload), so each op's weights are copied to the GPU while it runs and the GPU keeps only
# the compute buffers: 113 MiB instead of 963 MiB on an RTX 2080 Ti for the F16 projector at 625 image tokens, 0.44 s
# instead of 0.22 s per picture over PCIe 3.0 x8, the same embeddings bit for bit (docs/DETAILS.md).  Without the
# variable the encoder runs as before; the checkout in LLAMA_DIR is not modified.
if(STRATA_VISION_WEIGHTS_HOST)
  set(_clip_in "${LLAMA_DIR}/tools/mtmd/clip.cpp")
  set(_clip_out "${CMAKE_BINARY_DIR}/mtmd-weights-host/clip.cpp")
  file(READ "${_clip_in}" _clip)
  set(_anchor "ctx_clip.buf.reset(ggml_backend_alloc_ctx_tensors_from_buft(ctx_clip.ctx_data.get(), buft));")
  set(_insert [=[// Strata (tools/vision, STRATA_VISION_WEIGHTS_HOST): MTMD_WEIGHTS_HOST keeps the weights in host memory
            if (const char * wh = std::getenv("MTMD_WEIGHTS_HOST"); wh != nullptr && ctx_clip.backend != ctx_clip.backend_cpu) {
                ggml_backend_buffer_type_t host = nullptr;
                if (std::string(wh) == "pinned") {
                    host = ggml_backend_dev_host_buffer_type(ggml_backend_get_device(ctx_clip.backend));
                }
                buft = host ? host : ggml_backend_get_default_buffer_type(ctx_clip.backend_cpu);
                LOG_WRN("%s: MTMD_WEIGHTS_HOST: weights in %s, computed on %s\n", __func__,
                        ggml_backend_buft_name(buft), ggml_backend_name(ctx_clip.backend));
            }
            ]=])
  string(REPLACE "${_anchor}" "${_insert}${_anchor}" _patched "${_clip}")
  string(LENGTH "${_clip}" _len_in)
  string(LENGTH "${_patched}" _len_out)
  string(LENGTH "${_insert}" _len_insert)
  math(EXPR _grown "${_len_out} - ${_len_in}")
  if(NOT _grown EQUAL _len_insert)
    message(FATAL_ERROR "STRATA_VISION_WEIGHTS_HOST: ${_clip_in} does not have mtmd's weight allocation line exactly "
                        "once; this llama.cpp is not one the change was written for")
  endif()
  file(WRITE "${_clip_out}.in" "${_patched}")
  configure_file("${_clip_out}.in" "${_clip_out}" COPYONLY)     # rewritten only when it changes (no needless rebuild)
  set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${_clip_in}")
  get_target_property(_mtmd_sources mtmd SOURCES)
  list(FIND _mtmd_sources "clip.cpp" _at)
  if(_at LESS 0)
    message(FATAL_ERROR "STRATA_VISION_WEIGHTS_HOST: mtmd's sources do not list clip.cpp")
  endif()
  list(REMOVE_AT _mtmd_sources ${_at})
  list(INSERT _mtmd_sources ${_at} "${_clip_out}")
  set_property(TARGET mtmd PROPERTY SOURCES "${_mtmd_sources}")
  target_include_directories(mtmd PRIVATE "${LLAMA_DIR}/tools/mtmd")   # its headers, beside the original clip.cpp
  target_compile_definitions(strata-vision PRIVATE STRATA_VISION_WEIGHTS_HOST=1)
  message(STATUS "strata-vision: MTMD_WEIGHTS_HOST support (patched copy: ${_clip_out})")
endif()
