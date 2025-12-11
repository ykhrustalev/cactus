#include "../cactus/cactus.h"
#include <iostream>
#include <string>
#include <vector>
#include <chrono>
#include <iomanip>
#include <numeric>
#include <cmath>

using namespace cactus::engine;

// Create exact token sequence
std::vector<uint32_t> create_token_sequence(size_t count) {
    std::vector<uint32_t> tokens;
    tokens.reserve(count);
    for (size_t i = 0; i < count; i++) {
        tokens.push_back(10 + (i % 10));  // Pattern: 10-19 repeating
    }
    return tokens;
}

// Measure prefill time
double measure_prefill(Model* model, const std::vector<uint32_t>& tokens) {
    model->reset_cache();
    auto start = std::chrono::high_resolution_clock::now();
    model->generate(tokens, -1.0f, -1.0f, 0, "", true);
    auto end = std::chrono::high_resolution_clock::now();
    return std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1000.0;
}

// Measure decode time (returns prefill_ms, decode_ms)
std::pair<double, double> measure_decode(Model* model, uint32_t eos_token,
                                         const std::vector<uint32_t>& tokens, size_t decode_count) {
    model->reset_cache();

    // Prefill
    auto t1 = std::chrono::high_resolution_clock::now();
    model->generate(tokens, -1.0f, -1.0f, 0, "", true);
    auto t2 = std::chrono::high_resolution_clock::now();
    double prefill_ms = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count() / 1000.0;

    // Decode
    auto t3 = std::chrono::high_resolution_clock::now();
    uint32_t last_token = 0;
    for (size_t i = 0; i < decode_count; i++) {
        last_token = (i == 0) ? model->generate({}, -1.0f, -1.0f, 0, "", false)
                              : model->generate({last_token}, -1.0f, -1.0f, 0, "", false);
    }
    auto t4 = std::chrono::high_resolution_clock::now();
    double decode_ms = std::chrono::duration_cast<std::chrono::microseconds>(t4 - t3).count() / 1000.0;

    return {prefill_ms, decode_ms};
}

double calculate_mean(const std::vector<double>& values) {
    return std::accumulate(values.begin(), values.end(), 0.0) / values.size();
}

double calculate_stddev(const std::vector<double>& values, double mean) {
    double sum = 0.0;
    for (auto v : values) sum += (v - mean) * (v - mean);
    return std::sqrt(sum / values.size());
}

void print_usage(const char* program) {
    std::cerr << "Usage: " << program << " <model_path> --action <prefill|decode> --tokens <N> [--offset <N>] [--measurements <N>]\n\n";
    std::cerr << "Examples:\n";
    std::cerr << "  " << program << " weights/model --action prefill --tokens 512\n";
    std::cerr << "  " << program << " weights/model --action decode --offset 128 --tokens 100\n";
}

int main(int argc, char* argv[]) {
    std::cerr.setf(std::ios::unitbuf);

    if (argc < 2) {
        print_usage(argv[0]);
        return 1;
    }

    // Parse arguments
    const char* model_path = argv[1];
    std::string action;
    int tokens = -1;
    int offset = -1;
    int measurements = 3;

    for (int i = 2; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--action" && i + 1 < argc) action = argv[++i];
        else if (arg == "--tokens" && i + 1 < argc) tokens = std::atoi(argv[++i]);
        else if (arg == "--offset" && i + 1 < argc) offset = std::atoi(argv[++i]);
        else if (arg == "--measurements" && i + 1 < argc) measurements = std::atoi(argv[++i]);
        else {
            std::cerr << "Error: Unknown argument: " << arg << std::endl;
            print_usage(argv[0]);
            return 1;
        }
    }

    // Validate
    if (action != "prefill" && action != "decode") {
        std::cerr << "Error: --action must be 'prefill' or 'decode'" << std::endl;
        print_usage(argv[0]);
        return 1;
    }
    if (tokens <= 0) {
        std::cerr << "Error: --tokens required (must be > 0)" << std::endl;
        return 1;
    }
    if (action == "decode" && offset <= 0) {
        std::cerr << "Error: --offset required for decode" << std::endl;
        return 1;
    }

    // Load model
    std::cerr << "Loading model..." << std::endl;
    auto model = create_model(model_path);
    if (!model || !model->init(model_path, 8192, "", false)) {
        std::cerr << "Error: Failed to load model" << std::endl;
        return 1;
    }

    auto tokenizer = model->get_tokenizer();
    if (!tokenizer) {
        std::cerr << "Error: Failed to get tokenizer" << std::endl;
        return 1;
    }

    std::cerr << "Model: " << model_path << std::endl;
    std::cerr << "Action: " << action << std::endl;

    if (action == "prefill") {
        std::cerr << "Tokens: " << tokens << std::endl;
        std::cerr << "\nRunning prefill benchmark (" << measurements << " measurements)..." << std::endl;

        auto input = create_token_sequence(tokens);
        std::vector<double> times;

        for (int i = 0; i < measurements; i++) {
            std::cerr << "  [" << (i+1) << "/" << measurements << "] ";
            double ms = measure_prefill(model.get(), input);
            times.push_back(ms);
            std::cerr << ms << " ms" << std::endl;
        }

        double avg = calculate_mean(times);
        double stdev = calculate_stddev(times, avg);
        double tps = (tokens * 1000.0) / avg;

        std::cerr << "\nResults:" << std::endl;
        std::cerr << "  Avg: " << std::fixed << std::setprecision(2) << avg << " ms" << std::endl;
        std::cerr << "  StdDev: " << stdev << " ms" << std::endl;
        std::cerr << "  Throughput: " << tps << " tok/s" << std::endl;

        double stdev_tps = (tokens * 1000.0 * stdev) / (avg * avg);
        std::cout << "avg_ts: " << std::fixed << std::setprecision(2) << tps << std::endl;
        std::cout << "stddev_ts: " << stdev_tps << std::endl;

    } else { // decode
        std::cerr << "Offset: " << offset << " tokens" << std::endl;
        std::cerr << "Decode: " << tokens << " tokens" << std::endl;
        std::cerr << "\nRunning decode benchmark (" << measurements << " measurements)..." << std::endl;

        auto input = create_token_sequence(offset);
        uint32_t eos = tokenizer->get_eos_token();
        std::vector<double> prefill_times, decode_times;

        for (int i = 0; i < measurements; i++) {
            std::cerr << "  [" << (i+1) << "/" << measurements << "] ";
            auto [p_ms, d_ms] = measure_decode(model.get(), eos, input, tokens);
            prefill_times.push_back(p_ms);
            decode_times.push_back(d_ms);
            std::cerr << "prefill=" << std::fixed << std::setprecision(2) << p_ms
                      << " ms, decode=" << d_ms << " ms" << std::endl;
        }

        double p_avg = calculate_mean(prefill_times);
        double p_std = calculate_stddev(prefill_times, p_avg);
        double p_tps = (offset * 1000.0) / p_avg;

        double d_avg = calculate_mean(decode_times);
        double d_std = calculate_stddev(decode_times, d_avg);
        double d_tps = (tokens * 1000.0) / d_avg;

        std::cerr << "\nResults:" << std::endl;
        std::cerr << "  Prefill - Avg: " << std::fixed << std::setprecision(2) << p_avg
                  << " ms, StdDev: " << p_std << " ms, Throughput: " << p_tps << " tok/s" << std::endl;
        std::cerr << "  Decode  - Avg: " << d_avg
                  << " ms, StdDev: " << d_std << " ms, Throughput: " << d_tps << " tok/s" << std::endl;

        double d_stdev_tps = (tokens * 1000.0 * d_std) / (d_avg * d_avg);
        std::cout << "avg_ts: " << std::fixed << std::setprecision(2) << d_tps << std::endl;
        std::cout << "stddev_ts: " << d_stdev_tps << std::endl;
    }

    return 0;
}
