// Bounded logical contexts for the opt-in ID-routed conversation pool.
// One anonymous channel is reserved; the other slots belong to stable conversation IDs.
#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace strata::program {

class ConversationPool {
public:
    struct Entry {
        std::string key; // empty = the shared anonymous channel
        int64_t tokens = 0;
        uint64_t used = 0;
    };
    struct Change {
        bool admitted = true;
        std::vector<std::string> evicted;
    };

    ConversationPool(int64_t tokens, size_t slots) : budget_(tokens), named_slots_(slots > 0 ? slots - 1 : 0) {}
    bool enabled() const { return budget_ > 0; }
    int64_t tokens() const { return tokens_; }
    size_t evictions() const { return evictions_; }
    const std::vector<Entry>& entries() const { return entries_; }
    size_t named() const {
        return size_t(std::count_if(entries_.begin(), entries_.end(), [](const Entry& e) { return !e.key.empty(); }));
    }
    bool contains(const std::string& key) const {
        return std::any_of(entries_.begin(), entries_.end(), [&](const Entry& e) { return e.key == key; });
    }
    void forget(const std::string& key) {
        const auto it = std::find_if(entries_.begin(), entries_.end(), [&](const Entry& e) { return e.key == key; });
        if (it != entries_.end()) { tokens_ -= it->tokens; entries_.erase(it); }
    }

    // Begin a request with its real prompt length, then grow with committed tokens.
    // Do not reserve max_output_tokens: an unspecified output budget can be an entire context.
    // A shrinking request replaces its channel's old branch rather than keeping another copy.
    Change update(const std::string& key, int64_t tokens, bool touch = true) {
        Change out;
        if (!enabled()) return out;
        if (tokens < 0 || tokens > budget_ || (!key.empty() && named_slots_ == 0)) {
            out.admitted = false;
            return out; // invalid admissions leave every old entry intact
        }
        const bool existed = contains(key);
        auto evict = [&](bool named_only) {
            size_t victim = entries_.size();
            for (size_t i = 0; i < entries_.size(); ++i) {
                const auto& e = entries_[i];
                if (e.key == key || (named_only && e.key.empty())) continue;
                if (victim == entries_.size() || e.used < entries_[victim].used) victim = i;
            }
            if (victim == entries_.size()) return false;
            out.evicted.push_back(entries_[victim].key);
            tokens_ -= entries_[victim].tokens;
            entries_.erase(entries_.begin() + (std::ptrdiff_t) victim);
            ++evictions_;
            return true;
        };
        if (!existed && !key.empty() && named() >= named_slots_) evict(true);
        auto old = [&]() -> int64_t {
            for (const auto& e : entries_) if (e.key == key) return e.tokens;
            return 0;
        };
        while (tokens_ - old() > budget_ - tokens)
            if (!evict(false)) { out.admitted = false; return out; }
        for (auto& e : entries_) if (e.key == key) {
            tokens_ += tokens - e.tokens;
            e.tokens = tokens;
            if (touch) e.used = ++clock_;
            return out;
        }
        entries_.push_back({key, tokens, ++clock_});
        tokens_ += tokens;
        return out;
    }

private:
    int64_t budget_ = 0, tokens_ = 0;
    size_t named_slots_ = 0, evictions_ = 0;
    uint64_t clock_ = 0;
    std::vector<Entry> entries_;
};

} // namespace strata::program
