/* Copyright 2021 Aristocratos (jakob@qvantnet.com)

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

	   http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.

indent = tab
tab-size = 4
*/
#pragma once

#include <atomic>
#include <cstdint>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

namespace Llm {
	//* Collection and display of LLM inference engine metrics.
	//*
	//* Data source is the Prometheus text exposition format, which is served by all
	//* common engines (llama.cpp, vLLM, SGLang, TGI, Strata). Engine-specific metric
	//* names are mapped onto the normalized fields in Engine through alias tables,
	//* so the collector works against any engine that exposes the conventional names.

	struct Target {
		std::string label;
		std::string url;
		std::string token;		//!< Bearer token, resolved from literal or token_file
		std::string token_file;	//!< path to file containing the token (plain text or JSON with "api_key")
		std::string live_url;	//!< optional engine-specific JSON endpoint for live per-request state
		std::string accept = "text/plain; version=0.0.4";	//!< Accept header for the exposition scrape
		//? Parsed from url
		std::string host;
		std::string port;
		std::string path;
		bool valid = false;
		std::string error;
	};

	struct Series {
		std::string name;
		std::string key;			//!< name + canonicalized labels, stable across scrapes
		std::unordered_map<std::string, std::string> tags;
		double value = 0;
	};

	struct Engine {
		Target target;
		std::string detected;		//!< "strata", "llamacpp", "vllm", "sglang", "tgi", "generic"
		std::string model;
		bool up = false;
		std::string error;
		int status = 0;				//!< last HTTP status
		uint64_t bytes = 0;			//!< size of last response body
		uint64_t last_ok = 0;		//!< ms since epoch of last successful scrape
		uint64_t last_attempt = 0;
		uint32_t fails = 0;			//!< consecutive scrape failures

		//? Normalized fields, -1 when the engine does not expose the metric
		double gen_tok_s = -1;		//!< generated tokens/s (derived from counter deltas)
		double prefill_tok_s = -1;	//!< prompt tokens/s (derived from counter deltas)
		double gen_tok_s_avg = -1;	//!< engine-reported lifetime average, when available
		double prefill_tok_s_avg = -1;
		long long running = -1;		//!< requests currently in the batch
		long long waiting = -1;		//!< requests queued
		long long requests_total = -1;
		double kv_usage = -1;		//!< KV cache / token pool utilization, 0-1
		double hit_rate = -1;		//!< prefix cache hit ratio, 0-1 (derived from counter deltas)
		double accept_rate = -1;	//!< speculative decoding acceptance ratio, 0-1 (derived)
		//!< Same two ratios over the engine's whole lifetime. Interval values need traffic
		//!< in the last scrape, so these keep the panel informative while an engine idles.
		double hit_rate_lifetime = -1;
		double accept_rate_lifetime = -1;
		double rss_mib = -1;		//!< engine process resident memory
		std::unordered_map<std::string, double> gauges;	//!< summed raw values by metric name

		//? Live in-flight state, from the optional engine-specific JSON endpoint.
		//!< Engines that expose cumulative counters only (Strata updates its Prometheus
		//!< counters when a request finishes) need this for instantaneous throughput.
		bool live_available = false;
		std::string state;				//!< "idle", "generating", "prefilling", ...
		std::string phase;
		double live_tok_s = -1;
		double live_tok_s_mean = -1;
		double live_prefill_tok_s_mean = -1;
		double live_elapsed_s = -1;
		long long live_generated = -1;
		long long live_prompt_tokens = -1;
		long long live_prompt_read = -1;
		long long live_prompt_total = -1;
		long long live_queued = -1;

		//? Engine configuration, when the JSON endpoint reports it
		std::string engine_version;
		std::string kv_backend;
		double context_tokens = -1;
		double vram_free_mib = -1;
		double expert_cache_mib = -1;
		long long expert_slots = -1;

		//? Rate calculation state
		std::unordered_map<std::string, double> prev_values;
		std::unordered_map<std::string, uint64_t> prev_times;

		//* Best available instantaneous generation throughput: live JSON when the engine
		//!< provides it, otherwise the rate derived from Prometheus counter deltas.
		double current_tok_s() const {
			if(live_tok_s > 0)
				return live_tok_s;
			return gen_tok_s > 0 ? gen_tok_s : -1;
		}
	};

	struct Options {
		int timeout_ms = 400;		//!< per-connection socket timeout
		int interval_ms = 1000;		//!< scrape cadence, independent of update_ms
		size_t max_bytes = 8U << 20;	//!< response body cap
	};

	//* Guarded by mtx: written by collect() on the collector thread, read by draw on demand
	extern std::vector<Engine> engines;
	extern std::mutex mtx;
	extern Options options;

	//* Parse "<label>|<url>[|<token-or-token-file>[|<live-json-url>]]" specs and (re)initialize the target list
	void set_targets(const std::vector<std::string>& specs);

	bool enabled();

	//* Start/stop the background collector thread (own cadence, never blocks the UI thread)
	void start();
	void stop();

	//* Scrape all targets, refresh normalized fields. Called by the collector thread.
	void collect();

	void clear();

#ifndef BTOP_LLM_STANDALONE
	//? Menu support, see Menu::llmMenu
	std::vector<std::string> menu_content();
	std::string menu_title();
	void menu_key(const std::string& key);
	extern std::atomic<bool> menu_open;
#endif

	//* Pure helpers, exposed for standalone testing
	bool parse_url(const std::string& url, Target& out);
	std::vector<Series> parse_prometheus(const std::string& text);

	//* Fetch <text/plain exposition body> over a plain HTTP GET. Returns false and sets err on failure.
	bool http_get(const Target& target, const Options& opts, std::string& body, int& status, std::string& err);
}
