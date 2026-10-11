// Read-only validation with Strata's real C++ GGUF loader; no GPU initialization.
#include "strata/artifact/gguf_reader.hpp"
#include <cstdio>
#include <string>

int main(int argc, char** argv) {
    if (argc != 2) return 2;
    try {
        auto model = strata::GgufModel::open(argv[1]);
        const auto error = strata::check_architecture(model.meta());
        if (!error.empty()) throw std::runtime_error(error);
        if (model.size() != 2 || model.shard(0).tensors().size() != 1223 ||
            model.shard(1).tensors().size() != 1)
            throw std::runtime_error("Expected 1223 main tensors and one PLE tensor");
        size_t count = 0, experts = 0;
        for (size_t shard = 0; shard < model.size(); ++shard)
            for (const auto& t : model.shard(shard).tensors()) {
                if (!model.in_bounds(t, shard)) throw std::runtime_error("Out-of-bounds tensor: " + t.name);
                ++count;
                const std::string suffix = "_exps.weight";
                if (t.name.size() >= suffix.size() &&
                    t.name.compare(t.name.size() - suffix.size(), suffix.size(), suffix) == 0) {
                    if (shard != 0 || t.type != strata::kt::IQ3)
                        throw std::runtime_error("Expert is not in main shard as IQ3_KT: " + t.name);
                    ++experts;
                }
            }
        size_t shard = 0;
        const auto* ple = model.find("per_layer_token_embd.weight", &shard);
        if (!ple || shard != 1 || ple->type != strata::kt::IQ4 || ple->offset != 0 ||
            ple->shape != std::vector<uint64_t>{160, 320001536} ||
            strata::tensor_payload_bytes(*ple) != model.shard(1).file_size() - model.shard(1).data_start())
            throw std::runtime_error("Wrong standalone IQ4_KT PLE");
        if (experts != 144 || count != 1224) throw std::runtime_error("Wrong tensor counts");
        std::printf("{\"cpp_loader_passed\":true,\"architecture_passed\":true,\"shards\":2,"
                    "\"tensors_in_bounds\":%zu,\"iq3kt_expert_tensors\":%zu,\"separate_iq4kt_ple\":true}\n",
                    count, experts);
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "GGUF verification failed: %s\n", e.what());
        return 1;
    }
}
