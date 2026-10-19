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
		bool discovered = false;	//!< found by loopback autodetection rather than configured
		bool needs_auth = false;	//!< answered 401: an engine is there but needs a token
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
		double kv_usage = -1;		//!< KV cache / token pool utilization, 0-1 (engine gauge)
		double ctx_usage = -1;		//!< in-flight tokens / context window, 0-1 (derived)
									//!< engines without a KV gauge still answer "how full is the window"
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
		std::string state;				//!< "idle", "generating", "prefilling", ... (engine-reported)
		std::string derived_state;		//!< phase inferred from counter movement, used when the
		//!< engine reports no state itself (vLLM and friends have no live endpoint)
		bool has_baseline = false;		//!< false until one scrape has seeded the counters, so the
		//!< first sample cannot be mistaken for an idle engine
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

		//* Strata reports phase="thinking" for an entire reasoning generation, not as a
	//!< transient sub-step, so blanking throughput while it is set would hide the panel's
	//!< headline number for every request. Suppression is therefore opt-in (llm_hide_tok_s_thinking).
	bool thinking() const { return phase == "thinking"; }

	//* Defined in the .cpp: needs Options, which is declared after Engine
	double display_tok_s() const;

		//* Engine-reported state wins; otherwise the derived one, which may itself be empty
		const std::string& display_state() const {
			return state.empty() ? derived_state : state;
		}

		bool state_derived() const { return state.empty() and not derived_state.empty(); }
	};

	struct Options {
		int timeout_ms = 400;		//!< per-connection socket timeout
		int interval_ms = 1000;		//!< scrape cadence, independent of update_ms
		bool hide_tok_s_thinking = false;	//!< blank throughput while phase == "thinking"
		size_t max_bytes = 8U << 20;	//!< response body cap
		bool autodetect = true;		//!< find engines on loopback with no configuration at all
		int discover_ms = 30000;		//!< how often to rescan listening ports
		bool allow_token_env = true;		//!< fall back to $BTOP_LLM_TOKEN for discovered 401s
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

	//* Scan loopback listening sockets for Prometheus endpoints (Linux only, no-op elsewhere)
	void discover();

	//* Scrape all targets, refresh normalized fields. Called by the collector thread.
	void collect();

	void clear();

	//* Number of configured endpoints (fixed once set_targets has run)
	size_t target_count();

	//* True once per completed scrape round, so an open overlay repaints without the
	//!< collector thread ever naming Menu::
	bool new_data_since_menu_draw();

	//? Box geometry and rendering, defined in btop_draw.cpp alongside the other boxes
	extern int x, y, width, height, min_width, min_height;
	extern bool shown, redraw;
	extern std::string box;
	extern std::atomic<bool> menu_open;

	//* Rows the docked box needs: frame plus one dense line per engine. Also records the
	//!< count it was sized for, so collect() can ask for a reflow when that changes
	int box_height();

	//!< Endpoints the current box height was sized for, and a flag asking the main loop to
	//!< relayout: discovery fills in after calcSizes() has already run, which otherwise
	//!< leaves the box one row tall no matter how many engines were found
	extern std::atomic<size_t> layout_count;
	extern std::atomic<bool> layout_dirty;

	//* One dense line per engine, for the docked box
	std::vector<std::string> box_lines();
	//* Full detail for the selected engine, shared by the box overlay and Menu::llmMenu
	std::vector<std::string> detail_lines();
	std::string title();
	void nav(const std::string& key);
	std::string draw(bool force_redraw, bool no_update);

	//* Pure helpers, exposed for standalone testing
	bool parse_url(const std::string& url, Target& out);
	std::vector<Series> parse_prometheus(const std::string& text);

	//* Fetch <text/plain exposition body> over a plain HTTP GET. Returns false and sets err on failure.
	//!< tolerate_truncation lets discovery check a body's shape without downloading it all
	bool http_get(const Target& target, const Options& opts, std::string& body, int& status, std::string& err,
		bool tolerate_truncation = false);
}
