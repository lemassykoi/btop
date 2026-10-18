/* Standalone test harness for the Llm:: collector (not part of the btop build).

   build: g++ -std=c++23 -O1 -Wall -Wextra -pedantic -Isrc \
			  src/btop_llm.cpp tests/llm_probe.cpp -o /tmp/opencode/llm_probe
   usage: llm_probe ["label|url[|token-or-token-file]" ...]
		  llm_probe --parse-only <file>          (parse a saved exposition file)
*/
#include "btop_llm.hpp"

#include <chrono>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>

using namespace std::string_literals;
#include <thread>
#include <vector>

static std::vector<std::string> defaults() {
	//? Local engines only by default; pass other endpoints as argv to test remote ones.
	return {
		"strata|http://127.0.0.1:8013/metrics/prometheus|/home/clement/Strata/strata-unsloth-ud-iq4_xs.json",
	};
}

static void dump(const Llm::Engine& e) {
	auto num = [](double v, int prec = 1) {
		std::ostringstream os;
		if(v < 0)
			return std::string("--");
		os.precision(prec);
		os << std::fixed << v;
		return os.str();
	};
	std::cout << "=== " << e.target.label << " ===\n";
	std::cout << "  url          " << e.target.url << " (host=" << e.target.host
			  << " port=" << e.target.port << " path=" << e.target.path << ")\n";
	std::cout << "  valid        " << (e.target.valid ? "yes" : "no")
			  << (e.target.error.empty() ? "" : "  [" + e.target.error + "]") << "\n";
	std::cout << "  up           " << (e.up ? "yes" : "no") << "   http=" << e.status
			  << "   bytes=" << e.bytes << "   fails=" << e.fails << "\n";
	if(not e.error.empty())
		std::cout << "  error        " << e.error << "\n";
	std::cout << "  engine       " << (e.detected.empty() ? "--" : e.detected) << "\n";
	std::cout << "  model        " << (e.model.empty() ? "--" : e.model) << "\n";
	std::cout << "  tok/s        " << num(e.gen_tok_s) << "  (lifetime avg " << num(e.gen_tok_s_avg) << ")\n";
	std::cout << "  prefill tok/s " << num(e.prefill_tok_s) << "  (lifetime avg " << num(e.prefill_tok_s_avg) << ")\n";
	std::cout << "  running      " << (e.running < 0 ? "--" : std::to_string(e.running))
			  << "   waiting " << (e.waiting < 0 ? "--" : std::to_string(e.waiting))
			  << "   total   " << (e.requests_total < 0 ? "--" : std::to_string(e.requests_total)) << "\n";
	std::cout << "  kv usage     " << (e.kv_usage < 0 ? "--" : num(e.kv_usage * 100) + "%") << "\n";
	std::cout << "  cache hit    " << (e.hit_rate < 0 ? "--" : num(e.hit_rate * 100) + "%")
	          << "   [lifetime " << (e.hit_rate_lifetime < 0 ? "--" : num(e.hit_rate_lifetime * 100) + "%") + "]\n";
	std::cout << "  spec accept  " << (e.accept_rate < 0 ? "--" : num(e.accept_rate * 100, 0) + "%")
	          << "   [lifetime " << (e.accept_rate_lifetime < 0 ? "--" : num(e.accept_rate_lifetime * 100, 0) + "%") + "]\n";
	std::cout << "  engine rss   " << (e.rss_mib < 0 ? "--" : num(e.rss_mib / 1024.0, 2) + " GiB") << "\n";
	std::cout << "  live         " << (e.live_available
		? ("state=" + (e.state.empty() ? "?"s : e.state) + " phase=" + (e.phase.empty() ? "-"s : e.phase)
			+ " tok/s=" + num(e.live_tok_s) + " mean=" + num(e.live_tok_s_mean)
			+ " gen=" + (e.live_generated < 0 ? "--"s : std::to_string(e.live_generated))
			+ " prompt=" + (e.live_prompt_tokens < 0 ? "--"s : std::to_string(e.live_prompt_tokens))
			+ " read=" + (e.live_prompt_read < 0 ? "--"s : std::to_string(e.live_prompt_read))
			+ "/" + (e.live_prompt_total < 0 ? "--"s : std::to_string(e.live_prompt_total))
			+ " queued=" + (e.live_queued < 0 ? "--"s : std::to_string(e.live_queued))
			+ " elapsed=" + num(e.live_elapsed_s, 1) + "s")
		: std::string("--")) << "\n";
	std::cout << "  engine cfg   ctx=" << num(e.context_tokens, 0)
			  << " vram_free=" << num(e.vram_free_mib, 0) << " MiB"
			  << " experts=" << (e.expert_slots < 0 ? std::string("--") : std::to_string(e.expert_slots))
			  << " expert_cache=" << num(e.expert_cache_mib, 0) << " MiB"
			  << " kv=" << (e.kv_backend.empty() ? "--" : e.kv_backend)
			  << " ver=" << (e.engine_version.empty() ? "--" : e.engine_version) << "\n";
	std::cout << "  >> tok/s used by UI: " << num(e.current_tok_s()) << "\n";
	std::cout << "  raw metrics  " << e.gauges.size() << "\n";
	if(const auto* verbose = std::getenv("LLM_PROBE_RAW"))
		if(*verbose == '1' or *verbose == 'y')
			for(const auto& [name, value] : e.gauges)
				std::cout << "    " << name << " = " << value << "\n";
}

int main(int argc, char** argv) {
	std::vector<std::string> specs;

	if(argc > 2 and std::string(argv[1]) == "--parse-only") {
		std::ifstream file(argv[2]);
		std::stringstream buf;
		buf << file.rdbuf();
		const auto series = Llm::parse_prometheus(buf.str());
		std::cout << "parsed " << series.size() << " series\n";
		for(const auto& s : series)
			std::cout << "  " << s.key << " = " << s.value << "\n";
		return 0;
	}

	for(int i = 1; i < argc; ++i)
		specs.emplace_back(argv[i]);
	if(specs.empty())
		specs = defaults();

	Llm::options.timeout_ms = 1500;
	Llm::set_targets(specs);

	//? Two scrapes: the first seeds the counters, the second yields rates
	Llm::collect();
	std::this_thread::sleep_for(std::chrono::milliseconds(1500));
	Llm::collect();

	std::scoped_lock lock(Llm::mtx);
	for(const auto& e : Llm::engines)
		dump(e);
	return 0;
}
