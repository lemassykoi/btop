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

#include "btop_llm.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <charconv>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <fstream>
#include <map>
#include <set>
#include <sstream>
#include <thread>
#include <sys/socket.h>
#include <sys/types.h>
#include <fcntl.h>
#include <dirent.h>
#include <netdb.h>
#include <poll.h>
#include <unistd.h>

using namespace std::literals::string_literals;

namespace Llm {
	std::vector<Engine> engines;
	std::mutex mtx;
	Options options;

	//? Collector state, only touched by collect()/set_targets() (runner thread + startup)
	static std::vector<Engine> work;

	//* ---------------------------------------------------------------- helpers

	static uint64_t now_ms() {
		struct timespec ts{};
		clock_gettime(CLOCK_MONOTONIC, &ts);
		return static_cast<uint64_t>(ts.tv_sec) * 1000 + static_cast<uint64_t>(ts.tv_nsec) / 1000000;
	}

	static uint64_t epoch_ms() {
		struct timespec ts{};
		clock_gettime(CLOCK_REALTIME, &ts);
		return static_cast<uint64_t>(ts.tv_sec) * 1000 + static_cast<uint64_t>(ts.tv_nsec) / 1000000;
	}

	//! Locale-independent numeric parse. std::strtod honours LC_NUMERIC, and btop runs with the
	//! user's locale: under fr_FR it reads "72.05" as 72 and leaves ".05" unconsumed.
	static bool parse_double(const std::string& text, size_t& pos, double& out) {
		const char* first = text.data() + pos;
		const char* last = text.data() + text.size();
		const auto [ptr, ec] = std::from_chars(first, last, out);
		if(ec != std::errc() or ptr == first)
			return false;
		pos += static_cast<size_t>(ptr - first);
		return true;
	}

	static void trim(std::string& s) {
		const auto is_ws = [](unsigned char c) { return c == ' ' or c == '\t' or c == '\r' or c == '\n'; };
		while(not s.empty() and is_ws(s.front())) s.erase(s.begin());
		while(not s.empty() and is_ws(s.back())) s.pop_back();
	}

	static bool has_ctl(const std::string& s) {
		return std::any_of(s.begin(), s.end(), [](unsigned char c) { return c < 32 or c == 127; });
	}

	static std::string to_lower(std::string s) {
		std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) { return std::tolower(c); });
		return s;
	}

	//* Expand a leading ~ so token paths can be written the way users write them everywhere else
	static std::string expand_home(std::string path) {
		if(path.starts_with("~/") or path == "~") {
			if(const char* home = std::getenv("HOME"); home != nullptr and *home != '\0')
				return std::string(home) + path.substr(1);
		}
		return path;
	}

	//* Read a bearer token from a file: plain text, or JSON with an "api_key" member.
	static std::string read_token_file(const std::string& path, std::string& err) {
		std::ifstream file(path);
		if(not file.is_open()) {
			err = "cannot open token file "s + path;
			return {};
		}
		std::stringstream buf;
		buf << file.rdbuf();
		auto data = buf.str();
		trim(data);

		//? JSON config file: pull out "api_key": "<value>"
		if(data.starts_with('{')) {
			const auto key = data.find("\"api_key\"");
			if(key == std::string::npos) {
				err = "no \"api_key\" member in "s + path;
				return {};
			}
			const auto quote = data.find('"', data.find(':', key) + 1);
			if(quote == std::string::npos) {
				err = "malformed api_key in "s + path;
				return {};
			}
			const auto end = data.find('"', quote + 1);
			if(end == std::string::npos) {
				err = "unterminated api_key in "s + path;
				return {};
			}
			return data.substr(quote + 1, end - quote - 1);
		}

		if(data.empty())
			err = "empty token file "s + path;
		return data;
	}

	//* ---------------------------------------------------------------- parsing

	bool parse_url(const std::string& url, Target& out) {
		std::string rest = url;
		out.port = "80";

		if(auto scheme = to_lower(rest.substr(0, 8)); scheme == "https://") {
			out.error = "https not supported (no TLS): use an http:// endpoint or a local proxy";
			return false;
		}
		if(auto scheme = to_lower(rest.substr(0, 7)); scheme == "http://")
			rest.erase(0, 7);
		else if(scheme.starts_with("//"))
			rest.erase(0, 2);
		else {
			out.error = "missing http:// scheme";
			return false;
		}

		const auto slash = rest.find('/');
		auto authority = rest.substr(0, slash);
		out.path = slash == std::string::npos ? "/metrics"s : rest.substr(slash);

		if(authority.empty()) {
			out.error = "empty host";
			return false;
		}

		//? Only the last colon separates the port, so bare IPv6 literals need brackets
		if(authority.starts_with('[')) {
			const auto close = authority.find(']');
			if(close == std::string::npos) {
				out.error = "unbalanced IPv6 bracket in host";
				return false;
			}
			out.host = authority.substr(0, close + 1);
			if(close + 1 < authority.size()) {
				if(authority[close + 1] != ':') {
					out.error = "expected :port after IPv6 host";
					return false;
				}
				out.port = authority.substr(close + 2);
			}
		}
		else if(auto colon = authority.rfind(':'); colon != std::string::npos) {
			out.host = authority.substr(0, colon);
			out.port = authority.substr(colon + 1);
		}
		else
			out.host = authority;

		if(out.host.empty()) {
			out.error = "empty host";
			return false;
		}
		if(not std::all_of(out.port.begin(), out.port.end(), [](unsigned char c) { return std::isdigit(c); })
			or out.port.empty() or std::stoi(out.port) > 65535) {
			out.error = "invalid port: "s + out.port;
			return false;
		}
		if(has_ctl(out.host) or has_ctl(out.path)) {
			out.error = "control character in endpoint";
			return false;
		}
		if(out.path.front() != '/')
			out.path = "/"s + out.path;

		out.valid = true;
		return true;
	}

	std::vector<Series> parse_prometheus(const std::string& text) {
		std::vector<Series> series;
		std::string line;
		std::istringstream stream(text);

		while(std::getline(stream, line)) {
			if(not line.empty() and line.back() == '\r')
				line.pop_back();
			trim(line);
			if(line.empty() or line.front() == '#')
				continue;

			Series s;
			size_t pos = 0;

			//? Metric name: up to '{' or whitespace
			const auto name_end = line.find_first_of("{ \t", pos);
			if(name_end == std::string::npos)
				continue;
			s.name = line.substr(pos, name_end - pos);
			pos = name_end;
			if(s.name.empty())
				continue;

			//? Optional label set, respecting escaped quotes
			if(line[pos] == '{') {
				++pos;
				std::string key;
				std::string val;
				bool in_value = false;
				bool escaped = false;
				for(; pos < line.size(); ++pos) {
					const char c = line[pos];
					if(in_value) {
						if(escaped) {
							val += c;
							escaped = false;
						}
						else if(c == '\\') {
							escaped = true;
						}
						else if(c == '"') {
							in_value = false;
							s.tags[key] = val;
							key.clear();
							val.clear();
						}
						else
							val += c;
					}
					else if(c == '}') {
						++pos;
						break;
					}
					else if(c == '"')
						in_value = true;
					else if(c == '=')
						in_value = false;	//? next quoted run is the value
					else if(c == ',' or c == ' ')
						continue;
					else
						key += c;

					//? After a closing quote, '=' has already been consumed; skip separators
					if(not in_value and c == '"') {
						while(pos + 1 < line.size() and (line[pos + 1] == ',' or line[pos + 1] == ' '))
							++pos;
					}
				}
				trim(key);
				if(not key.empty())	//? unterminated label set
					continue;
			}

			while(pos < line.size() and (line[pos] == ' ' or line[pos] == '\t'))
				++pos;

			//? Sample value, optionally followed by a timestamp we ignore
			double value = 0;
			if(not parse_double(line, pos, value) or std::isnan(value))
				continue;
			s.value = value;

			//? Canonical key: name plus sorted labels, stable across scrapes
			std::vector<std::string> tags;
			tags.reserve(s.tags.size());
			for(const auto& [k, v] : s.tags)
				tags.push_back(k + "=" + v);
			std::ranges::sort(tags);
			s.key = s.name;
			for(const auto& t : tags)
				s.key += ',' + t;

			series.push_back(std::move(s));
		}
		return series;
	}

	//* ---------------------------------------------------------------- http

	//! Minimal blocking HTTP/1.1 GET over a plain socket: no TLS, no keep-alive, no gzip.
	//! btop links nothing but libc/libstdc++, so pulling in an HTTP client library is not an option.
	static bool send_all(int fd, const std::string& buf, int timeout_ms, std::string& err) {
		size_t sent = 0;
		while(sent < buf.size()) {
			struct pollfd pfd{fd, POLLOUT, 0};
			if(poll(&pfd, 1, timeout_ms) <= 0) {
				err = "send timeout";
				return false;
			}
			const auto n = ::send(fd, buf.data() + sent, buf.size() - sent, MSG_NOSIGNAL);
			if(n <= 0) {
				err = "send failed: "s + std::strerror(errno);
				return false;
			}
			sent += static_cast<size_t>(n);
		}
		return true;
	}

	//! Decode a chunked transfer-encoding body; returns false on malformed input.
	static bool dechunk(const std::string& in, std::string& out, size_t max_bytes) {
		size_t pos = 0;
		out.clear();
		while(true) {
			const auto eol = in.find("\r\n", pos);
			if(eol == std::string::npos)
				return false;
			const auto size = std::strtoul(in.c_str() + pos, nullptr, 16);
			pos = eol + 2;
			if(size == 0)
				return true;
			if(out.size() + size > max_bytes or pos + size > in.size())
				return false;
			out.append(in, pos, size);
			pos += size;
			if(in.compare(pos, 2, "\r\n") != 0)
				return false;
			pos += 2;
		}
	}

	bool http_get(const Target& target, const Options& opts, std::string& body, int& status, std::string& err,
		bool tolerate_truncation) {
		status = 0;
		body.clear();

		//? getaddrinfo wants "::1", not the "[::1]" form used in URLs and stored in Target::host
		std::string node = target.host;
		if(node.size() > 2 and node.front() == '[' and node.back() == ']')
			node = node.substr(1, node.size() - 2);

		struct addrinfo hints{};
		hints.ai_family = AF_UNSPEC;
		hints.ai_socktype = SOCK_STREAM;

		struct addrinfo* addresses = nullptr;
		const int rc = ::getaddrinfo(node.c_str(), target.port.c_str(), &hints, &addresses);
		if(rc != 0) {
			err = "resolving "s + node + ": " + gai_strerror(rc);
			return false;
		}

		int fd = -1;
		for(auto* addr = addresses; addr != nullptr and fd < 0; addr = addr->ai_next) {
			fd = ::socket(addr->ai_family, addr->ai_socktype, addr->ai_protocol);
			if(fd < 0)
				continue;

			const auto flags = ::fcntl(fd, F_GETFL, 0);
			::fcntl(fd, F_SETFL, flags | O_NONBLOCK);

			if(::connect(fd, addr->ai_addr, addr->ai_addrlen) != 0 and errno != EINPROGRESS) {
				::close(fd);
				fd = -1;
				continue;
			}
			if(errno == EINPROGRESS) {
				struct pollfd pfd{fd, POLLOUT, 0};
				if(poll(&pfd, 1, opts.timeout_ms) <= 0) {
					err = "connect timeout to "s + target.host + ":" + target.port;
					::close(fd);
					fd = -1;
					continue;
				}
				int so_error = 0;
				socklen_t len = sizeof(so_error);
				::getsockopt(fd, SOL_SOCKET, SO_ERROR, &so_error, &len);
				if(so_error != 0) {
					err = "connect failed: "s + std::strerror(so_error);
					::close(fd);
					fd = -1;
					continue;
				}
			}

			//? Back to blocking mode with explicit receive/send timeouts
			::fcntl(fd, F_SETFL, flags);
			struct timeval tv{};
			tv.tv_sec = opts.timeout_ms / 1000;
			tv.tv_usec = (opts.timeout_ms % 1000) * 1000;
			::setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));
			::setsockopt(fd, SOL_SOCKET, SO_SNDTIMEO, &tv, sizeof(tv));
		}
		::freeaddrinfo(addresses);

		if(fd < 0) {
			if(err.empty())
				err = "could not connect to "s + target.host + ":" + target.port;
			return false;
		}

		std::string request = "GET "s + target.path + " HTTP/1.1\r\n"
			"Host: " + target.host + ":" + target.port + "\r\n"
			"Accept: " + (target.accept.empty() ? "text/plain; version=0.0.4"s : target.accept) + "\r\n"
			"Accept-Encoding: identity\r\n"
			"User-Agent: btop-llm/1.0\r\n"
			"Connection: close\r\n";
		if(not target.token.empty())
			request += "Authorization: Bearer " + target.token + "\r\n";
		request += "\r\n";

		if(not send_all(fd, request, opts.timeout_ms, err)) {
			::close(fd);
			return false;
		}

		std::string raw;
		bool truncated = false;
		while(true) {
			char buffer[16384];
			const auto n = ::recv(fd, buffer, sizeof(buffer), 0);
			if(n < 0) {
				if(errno == EINTR)
					continue;
				err = "recv failed: "s + std::strerror(errno);
				::close(fd);
				return false;
			}
			if(n == 0)
				break;
			if(raw.size() + static_cast<size_t>(n) > opts.max_bytes) {
				truncated = true;
				break;
			}
			raw.append(buffer, static_cast<size_t>(n));
		}
		::close(fd);

		const auto header_end = raw.find("\r\n\r\n");
		if(header_end == std::string::npos) {
			err = truncated ? "response exceeds limit before headers" : "malformed HTTP response";
			return false;
		}

		const std::string head = raw.substr(0, header_end);
		std::string payload = raw.substr(header_end + 4);

		const auto first_line_end = head.find("\r\n");
		const std::string status_line = head.substr(0, first_line_end == std::string::npos ? head.size() : first_line_end);
		if(status_line.starts_with("HTTP/") and status_line.size() > 8)
			status = std::atoi(status_line.c_str() + 8);

		const std::string lower_head = to_lower(head);
		if(lower_head.find("transfer-encoding: chunked") != std::string::npos) {
			std::string decoded;
			if(not dechunk(payload, decoded, opts.max_bytes)) {
				err = "malformed chunked response";
				return false;
			}
			payload = std::move(decoded);
		}

		if(status < 200 or status > 299) {
			err = "HTTP "s + std::to_string(status);
			//? Keep a short fragment of the body: engines report the reason there
			const auto fragment = payload.substr(0, 120);
			if(not fragment.empty())
				err += " " + fragment;
			return false;
		}
		if(truncated and not tolerate_truncation) {
			err = "response exceeds "s + std::to_string(opts.max_bytes / (1024 * 1024)) + " MiB";
			return false;
		}

		body = std::move(payload);
		return true;
	}

	//* ---------------------------------------------------------------- targets

	void set_targets(const std::vector<std::string>& specs) {
		std::vector<Engine> next;
		for(const auto& spec : specs) {
			std::vector<std::string> fields;
			std::string field;
			std::stringstream stream(spec);
			while(std::getline(stream, field, '|'))
				fields.push_back(field);

			if(fields.size() < 2 or fields[0].empty() or fields[1].empty()) {
				Engine bad;
				bad.target.label = fields.empty() ? "invalid"s : fields[0];
				bad.target.url = fields.size() > 1 ? fields[1] : "";
				bad.target.error = "expected <label>|<url>[|<token-or-file>]";
				next.push_back(std::move(bad));
				continue;
			}

			Engine engine;
			engine.target.label = fields[0];
			engine.target.url = fields[1];
			trim(engine.target.label);
			trim(engine.target.url);
			parse_url(engine.target.url, engine.target);
			engine.error = engine.target.error;

			//? Third field: bearer token, or a path to read one from
			if(fields.size() > 2 and not fields[2].empty()) {
				std::string source = fields[2];
				trim(source);
				if(source.starts_with('/') or source.starts_with('~') or source.ends_with(".json")) {
					source = expand_home(source);
					engine.target.token_file = source;
					engine.target.token = read_token_file(source, engine.error);
					if(engine.target.token.empty())
						engine.target.valid = false;
				}
				else
					engine.target.token = source;
			}

			//? Fourth field: engine-specific JSON endpoint for live per-request state
			if(fields.size() > 3 and not fields[3].empty()) {
				engine.target.live_url = fields[3];
				trim(engine.target.live_url);
			}

			next.push_back(std::move(engine));
		}
		work = std::move(next);
	}

	bool enabled() {
		return not work.empty();
	}

	size_t target_count() {
		return work.size();
	}

	std::atomic<size_t> layout_count{0};
	std::atomic<bool> layout_dirty{false};

	void clear() {
		work.clear();
		std::scoped_lock lock(mtx);
		engines.clear();
	}

	//* ---------------------------------------------------------------- json

	//! Minimal read-only JSON reader for the optional live-state endpoint.
	//! btop links nothing but libc/libstdc++, so a JSON library is not an option.
	namespace json {
		struct JVal {
			enum class Kind { Null, Bool, Num, Str, Arr, Obj } kind = Kind::Null;
			bool boolean = false;
			double num = 0;
			std::string str;
			std::vector<std::pair<std::string, JVal>> members;
			std::vector<JVal> items;

			const JVal* member(std::string_view name) const {
				for(const auto& [key, value] : members)
					if(key == name)
						return &value;
				return nullptr;
			}
		};

		struct Guard {
			int& depth;
			explicit Guard(int& d) : depth(d) {
				if(++depth > 64)
					over = true;
			}
			~Guard() { --depth; }
			bool over = false;
		};

		struct Parse {
			const std::string& text;
			size_t pos = 0;
			int depth = 0;
			bool failed = false;

			explicit Parse(const std::string& t) : text(t) {}

			void skip_ws() {
				while(pos < text.size() and (text[pos] == ' ' or text[pos] == '\t' or text[pos] == '\n' or text[pos] == '\r'))
					++pos;
			}

			bool literal(std::string_view lit) {
				if(text.compare(pos, lit.size(), lit) == 0) {
					pos += lit.size();
					return true;
				}
				failed = true;
				return false;
			}

			bool string(std::string& out) {
				if(pos >= text.size() or text[pos] != '"') {
					failed = true;
					return false;
				}
				++pos;
				out.clear();
				while(pos < text.size()) {
					const char c = text[pos++];
					if(c == '"')
						return true;
					if(c != '\\') {
						out += c;
						continue;
					}
					if(pos >= text.size())
						break;
					const char e = text[pos++];
					switch(e) {
						case '"': out += '"'; break;
						case '\\': out += '\\'; break;
						case '/': out += '/'; break;
						case 'b': out += '\b'; break;
						case 'f': out += '\f'; break;
						case 'n': out += '\n'; break;
						case 'r': out += '\r'; break;
						case 't': out += '\t'; break;
						case 'u': {
							if(pos + 4 > text.size()) {
								failed = true;
								return false;
							}
							const auto code = std::strtoul(text.substr(pos, 4).c_str(), nullptr, 16);
							pos += 4;
							if(code < 0x80)
								out += static_cast<char>(code);
							else if(code < 0x800) {
								out += static_cast<char>(0xC0 | (code >> 6));
								out += static_cast<char>(0x80 | (code & 0x3F));
							}
							else {
								out += static_cast<char>(0xE0 | (code >> 12));
								out += static_cast<char>(0x80 | ((code >> 6) & 0x3F));
								out += static_cast<char>(0x80 | (code & 0x3F));
							}
							break;
						}
						default:
							failed = true;
							return false;
					}
				}
				failed = true;
				return false;
			}

			void value(JVal& out) {
				Guard guard(depth);
				if(guard.over) {
					failed = true;
					return;
				}
				skip_ws();
				if(pos >= text.size()) {
					failed = true;
					return;
				}

				switch(text[pos]) {
					case '{': {
						out.kind = JVal::Kind::Obj;
						++pos;
						skip_ws();
						if(pos < text.size() and text[pos] == '}') {
							++pos;
							return;
						}
						while(pos < text.size()) {
							skip_ws();
							std::string key;
							if(not string(key))
								return;
							skip_ws();
							if(pos >= text.size() or text[pos] != ':') {
								failed = true;
								return;
							}
							++pos;
							JVal member_value;
							value(member_value);
							if(failed)
								return;
							out.members.emplace_back(std::move(key), std::move(member_value));
							skip_ws();
							if(pos < text.size() and text[pos] == ',') {
								++pos;
								continue;
							}
							if(pos < text.size() and text[pos] == '}') {
								++pos;
								return;
							}
							failed = true;
							return;
						}
						failed = true;
						return;
					}
					case '[': {
						out.kind = JVal::Kind::Arr;
						++pos;
						skip_ws();
						if(pos < text.size() and text[pos] == ']') {
							++pos;
							return;
						}
						while(pos < text.size()) {
							JVal item;
							value(item);
							if(failed)
								return;
							out.items.push_back(std::move(item));
							skip_ws();
							if(pos < text.size() and text[pos] == ',') {
								++pos;
								continue;
							}
							if(pos < text.size() and text[pos] == ']') {
								++pos;
								return;
							}
							failed = true;
							return;
						}
						failed = true;
						return;
					}
					case '"': {
						out.kind = JVal::Kind::Str;
						string(out.str);
						return;
					}
					case 't': {
						out.kind = JVal::Kind::Bool;
						if(literal("true"))
							out.boolean = true;
						return;
					}
					case 'f': {
						out.kind = JVal::Kind::Bool;
						if(literal("false"))
							out.boolean = false;
						return;
					}
					case 'n': {
						out.kind = JVal::Kind::Null;
						literal("null");
						return;
					}
					default: {
						double number = 0;
						size_t at = pos;
						if(not parse_double(text, at, number) or std::isnan(number)) {
							failed = true;
							return;
						}
						out.kind = JVal::Kind::Num;
						out.num = number;
						pos = at;
						return;
					}
				}
			}
		};

		bool parse(const std::string& text, JVal& out) {
			Parse parser(text);
			parser.value(out);
			return not parser.failed;
		}

		const JVal* lookup(const JVal& root, std::string_view path) {
			const JVal* node = &root;
			size_t start = 0;
			while(true) {
				const auto dot = path.find('.', start);
				const auto key = path.substr(start, dot == std::string_view::npos ? std::string_view::npos : dot - start);
				if(node->kind != JVal::Kind::Obj or (node = node->member(key)) == nullptr)
					return nullptr;
				if(dot == std::string_view::npos)
					return node;
				start = dot + 1;
			}
		}

		bool number(const JVal& root, std::string_view path, double& out) {
			const auto* node = lookup(root, path);
			if(node == nullptr or node->kind != JVal::Kind::Num)
				return false;
			out = node->num;
			return true;
		}

		bool text(const JVal& root, std::string_view path, std::string& out) {
			const auto* node = lookup(root, path);
			if(node == nullptr or node->kind != JVal::Kind::Str or node->str.empty())
				return false;
			out = node->str;
			return true;
		}

		//* First numeric value found among the dotted paths, -1 when none is present
		double pick_num(const JVal& root, const std::array<const char*, 4>& paths) {
			for(const char* path : paths) {
				if(path == nullptr)
					break;
				double value = 0;
				if(number(root, path, value))
					return value;
			}
			return -1;
		}
	}

	//* ---------------------------------------------------------------- metrics

	//! Metric names for a normalized field are given as a null-terminated array, in
	//! preference order. Covers llama.cpp (and Strata, which mirrors it), vLLM, SGLang and TGI.
	static double pick(const std::unordered_map<std::string, double>& sums, const std::array<const char*, 8>& names) {
		for(const char* name : names) {
			if(name == nullptr)
				break;
			if(auto it = sums.find(name); it != sums.end())
				return it->second;
		}
		return -1;
	}

	static double counter_rate(Engine& engine,
		const std::unordered_map<std::string, double>& sums,
		const std::array<const char*, 8>& names,
		uint64_t now) {
		for(const char* name : names) {
			if(name == nullptr)
				break;
			const auto it = sums.find(name);
			if(it == sums.end())
				continue;
			const auto prev = engine.prev_values.find(it->first);
			const auto time = engine.prev_times.find(it->first);
			if(prev == engine.prev_values.end() or time == engine.prev_times.end())
				return 0.0;	//? first scrape: no rate yet
			const double dt = static_cast<double>(now - time->second);
			if(dt <= 0)
				return 0.0;
			const double delta = it->second - prev->second;
			//? Counter reset (engine restart or counter reload): report idle rather than a spike
			if(delta < 0)
				return 0.0;
			return delta * 1000.0 / dt;
		}
		return -1;
	}

	static std::string detect(const std::unordered_map<std::string, double>& sums) {
		for(const auto& [name, value] : sums) {
			if(name.starts_with("strata:"))
				return "strata";
		}
		for(const auto& [name, value] : sums) {
			if(name.starts_with("sglang:"))
				return "sglang";
			if(name.starts_with("vllm:"))
				return "vllm";
			if(name.starts_with("llamacpp:"))
				return "llamacpp";
			if(name.starts_with("tgi_"))
				return "tgi";
		}
		return sums.empty() ? ""s : "generic"s;
	}

	//!< One-hot state gauges (strata:live_state{state="generating"} 1) must be read by label:
	//!< summing them per name, as the gauge table does, would leave 1 and lose which state is set
	static std::string active_state(const std::vector<Series>& series, const char* name) {
		for(const auto& s : series) {
			if(s.name != name or s.value <= 0)
				continue;
			if(auto it = s.tags.find("state"); it != s.tags.end() and not it->second.empty())
				return it->second;
		}
		return {};
	}

	static std::string find_model(const std::vector<Series>& series) {
		static constexpr std::array<const char*, 6> keys = {
			"model_name", "model", "llm_model", "model_id", "served_model_name", "engine"
		};
		for(const auto& s : series)
			for(const char* key : keys)
				if(auto it = s.tags.find(key); it != s.tags.end() and not it->second.empty())
					return it->second;
		return {};
	}

	//! Pull instantaneous per-request state from an engine-specific JSON endpoint.
	//! Strata advances its Prometheus counters only when a request finishes, so live
	//! throughput has no other source; engines with live counters (vLLM, SGLang,
	//! llama.cpp) need no adapter and leave live_url empty.
	static void enrich_live(Engine& engine) {
		engine.live_available = false;
		engine.state.clear();
		engine.phase.clear();
		engine.live_tok_s = engine.live_tok_s_mean = -1;
		engine.live_prefill_tok_s_mean = engine.live_elapsed_s = -1;
		engine.live_generated = engine.live_prompt_tokens = -1;
		engine.live_prompt_read = engine.live_prompt_total = engine.live_queued = -1;
		engine.context_tokens = engine.vram_free_mib = engine.expert_cache_mib = -1;
		engine.ctx_usage = -1;
		engine.expert_slots = -1;

				if(engine.target.live_url.empty())
			return;

		Target source = engine.target;
		source.url = engine.target.live_url;
		source.accept = "application/json";
		source.error.clear();
		if(not parse_url(source.url, source))
			return;

		std::string body;
		int status = 0;
		std::string err;
		if(not http_get(source, options, body, status, err))
			return;

		json::JVal root;
		if(not json::parse(body, root))
			return;

		json::text(root, "live.state", engine.state);
		json::text(root, "live.phase", engine.phase);
		json::text(root, "engine.version", engine.engine_version);
		json::text(root, "engine.kv", engine.kv_backend);
		if(engine.model.empty())
			json::text(root, "engine.model", engine.model);

		engine.live_tok_s = json::pick_num(root, {"live.tok_s", nullptr});
		engine.live_tok_s_mean = json::pick_num(root, {"live.tok_s_mean", nullptr});
		engine.live_prefill_tok_s_mean = json::pick_num(root, {"live.prefill_tok_s_mean", nullptr});
		engine.live_elapsed_s = json::pick_num(root, {"live.elapsed_s", nullptr});
		engine.context_tokens = json::pick_num(root, {"engine.context", "engine.max_context", "max_model_len", nullptr});
		engine.vram_free_mib = json::pick_num(root, {"engine.vram_free_mib", nullptr});
		engine.expert_cache_mib = json::pick_num(root, {"engine.expert_cache_mib", nullptr});

		if(const auto value = json::pick_num(root, {"live.generated", nullptr}); value >= 0)
			engine.live_generated = static_cast<long long>(value);
		if(const auto value = json::pick_num(root, {"live.prompt_tokens", nullptr}); value >= 0)
			engine.live_prompt_tokens = static_cast<long long>(value);
		if(const auto value = json::pick_num(root, {"live.prompt_read", nullptr}); value >= 0)
			engine.live_prompt_read = static_cast<long long>(value);
		if(const auto value = json::pick_num(root, {"live.prompt_total", nullptr}); value >= 0)
			engine.live_prompt_total = static_cast<long long>(value);
		if(const auto value = json::pick_num(root, {"live.queued", nullptr}); value >= 0)
			engine.live_queued = static_cast<long long>(value);
		if(const auto value = json::pick_num(root, {"engine.expert_slots", nullptr}); value >= 0)
			engine.expert_slots = static_cast<long long>(value);

		//? In-flight prefix reuse: the tokens the engine skipped re-reading this request
		if(engine.live_prompt_total > 0 and engine.live_prompt_read >= 0
			and engine.live_prompt_read <= engine.live_prompt_total)
			engine.hit_rate = 1.0 - static_cast<double>(engine.live_prompt_read) / engine.live_prompt_total;

		//? How much of the context window the in-flight request occupies: the closest
		//!< honest stand-in for a KV utilization figure on engines that do not report one
		if(engine.context_tokens > 0) {
			const auto in_flight = static_cast<double>(std::max(0LL, engine.live_prompt_tokens)
				+ std::max(0LL, engine.live_generated));
			if(in_flight > 0)
				engine.ctx_usage = std::min(1.0, in_flight / engine.context_tokens);
		}

		engine.live_available = true;
	}

	void collect() {
		if(work.empty())
			return;

		const auto started = now_ms();
		for(auto& engine : work) {
			if(not engine.target.valid)
				continue;
			engine.last_attempt = epoch_ms();

			std::string body;
			int status = 0;
			std::string err;
			if(not http_get(engine.target, options, body, status, err)) {
				engine.status = status;
				engine.up = false;
				engine.error = err;
				++engine.fails;
				continue;
			}

			const auto series = parse_prometheus(body);
			if(series.empty()) {
				engine.up = false;
				engine.status = status;
				engine.error = "no metrics in response";
				++engine.fails;
				continue;
			}

			//? Sum series sharing a metric name: engines expose one series per model/worker
			std::unordered_map<std::string, double> sums;
			for(const auto& s : series)
				sums[s.name] += s.value;

			const auto finished = now_ms();

			engine.status = status;
			engine.bytes = body.size();
			engine.derived_state.clear();
			engine.error.clear();
			engine.up = true;
			engine.fails = 0;
			engine.last_ok = engine.last_attempt;
			engine.detected = detect(sums);
			if(auto model = find_model(series); not model.empty())
				engine.model = std::move(model);
			engine.gauges = sums;

			//? Counters are cumulative: instantaneous throughput comes from the delta since last scrape
			static const std::array<const char*, 8> gen_total = {
				"llamacpp:tokens_predicted_total", "vllm:generation_tokens_total",
				"sglang:generation_tokens_total", "tgi_token_generated", nullptr
			};
			static const std::array<const char*, 8> prompt_total = {
				"llamacpp:prompt_tokens_total", "vllm:prompt_tokens_total",
				"sglang:prompt_tokens_total", "tgi_prompt_tokens_total", nullptr
			};
			static const std::array<const char*, 8> cached_total = {
				"llamacpp:prompt_tokens_cached_total", "vllm:prefix_cache_hits_total",
				"sglang:cached_tokens_total", nullptr
			};
			static const std::array<const char*, 8> running = {
				"llamacpp:requests_processing", "vllm:num_requests_running",
				"sglang:num_running_reqs", "tgi_batch_current_size", nullptr
			};
			static const std::array<const char*, 8> queued = {
				"llamacpp:requests_deferred", "vllm:num_requests_waiting",
				"sglang:num_queue_reqs", "tgi_queue_size", nullptr
			};
			static const std::array<const char*, 8> requests = {
				"strata:requests_total", "vllm:request_success_total",
				"sglang:num_requests_total", "tgi_request_count", nullptr
			};
			static const std::array<const char*, 8> kv = {
				"vllm:gpu_cache_usage_perc", "vllm:kv_cache_usage_perc",
				"sglang:token_usage", nullptr
			};
			static const std::array<const char*, 8> gen_avg = {
				"llamacpp:predicted_tokens_seconds", "vllm:avg_generation_throughput_toks_per_s",
				"sglang:gen_throughput", nullptr
			};
			static const std::array<const char*, 8> prompt_avg = {
				"llamacpp:prompt_tokens_seconds", "vllm:avg_prompt_throughput_toks_per_s", nullptr
			};
			static const std::array<const char*, 8> rss = {
				"process_resident_memory_bytes", "strata:ram_used_bytes", nullptr
			};
			static const std::array<const char*, 8> draft_offered = {
				"strata:drafts_offered_total", "vllm:spec_decode_num_draft_tokens_total", nullptr
			};
			static const std::array<const char*, 8> draft_accepted = {
				"strata:drafts_accepted_total", "vllm:spec_decode_num_accepted_tokens_total", nullptr
			};

			engine.gen_tok_s = counter_rate(engine, sums, gen_total, started);
			engine.prefill_tok_s = counter_rate(engine, sums, prompt_total, started);

			//? Engines without a live state field (vLLM, SGLang, llama.cpp) still get a
			//!< "current step": which counter moved since the last scrape. Needs a baseline,
			//!< otherwise the very first sample would read as idle.
			if(engine.has_baseline) {
				if(engine.gen_tok_s > 0)
					engine.derived_state = "generating";
				else if(engine.prefill_tok_s > 0)
					engine.derived_state = "reading";
				else if(engine.running > 0)
					engine.derived_state = "scheduled";
				else
					engine.derived_state = "idle";
			}
			engine.gen_tok_s_avg = pick(sums, gen_avg);
			engine.prefill_tok_s_avg = pick(sums, prompt_avg);

			if(auto value = pick(sums, running); value >= 0)
				engine.running = static_cast<long long>(value);
			if(auto value = pick(sums, queued); value >= 0)
				engine.waiting = static_cast<long long>(value);
			if(auto value = pick(sums, requests); value >= 0)
				engine.requests_total = static_cast<long long>(value);
			engine.kv_usage = pick(sums, kv);
			if(const auto bytes = pick(sums, rss); bytes >= 0)
				engine.rss_mib = bytes / (1024.0 * 1024.0);
			else
				engine.rss_mib = -1;

			//? Prefix cache hit ratio: cached / (cached + actually read) over this interval
			{
				static const std::array<const char*, 8> hit_gauge = {
					"sglang:cache_hit_rate", "vllm:prefix_cache_hit_rate", "strata:last_hit_rate", nullptr
				};
				static const std::array<const char*, 8> queries_total = {
					"vllm:prefix_cache_queries_total", "sglang:prefix_cache_queries_total", nullptr
				};
				engine.hit_rate = pick(sums, hit_gauge);
				if(engine.hit_rate < 0) {
					//? vLLM/SGLang expose an explicit queries counter: hits / queries
					const auto hits = counter_rate(engine, sums, cached_total, started);
					const auto queries = counter_rate(engine, sums, queries_total, started);
					if(hits >= 0 and queries > 0)
						engine.hit_rate = hits / queries;
					else if(hits >= 0) {
						//? llama.cpp/Strata only report cached vs actually read tokens
						const auto read = counter_rate(engine, sums, prompt_total, started);
						if(read >= 0 and (hits + read) > 0)
							engine.hit_rate = hits / (hits + read);
					}
				}

				//? Same ratio since the engine started: counters are cumulative, so this stays
				//!< meaningful while the engine is idle and the interval rate has no traffic
				const auto hits_all = pick(sums, cached_total);
				const auto queries_all = pick(sums, queries_total);
				const auto read_all = pick(sums, prompt_total);
				if(hits_all >= 0 and queries_all > 0)
					engine.hit_rate_lifetime = hits_all / queries_all;
				else if(hits_all >= 0 and read_all >= 0 and (hits_all + read_all) > 0)
					engine.hit_rate_lifetime = hits_all / (hits_all + read_all);
			}

			//? Speculative decoding acceptance ratio over this interval, and since start
			{
				const auto offered = counter_rate(engine, sums, draft_offered, started);
				const auto accepted = counter_rate(engine, sums, draft_accepted, started);
				engine.accept_rate = (offered > 0 and accepted >= 0) ? accepted / offered : -1;

				const auto offered_all = pick(sums, draft_offered);
				const auto accepted_all = pick(sums, draft_accepted);
				if(offered_all > 0 and accepted_all >= 0)
					engine.accept_rate_lifetime = accepted_all / offered_all;
			}

			//? Strata reports throughput only in its JSON state, so default the adapter
			//? to the sibling /metrics endpoint of the same server
			if(engine.target.live_url.empty() and engine.detected == "strata")
				engine.target.live_url = "http://"s + engine.target.host + ":" + engine.target.port + "/metrics";
			enrich_live(engine);

			//? Fill what the JSON adapter could not provide straight from the exposition, so
			//!< engines without a JSON endpoint still get live state and throughput. rss and the
			//!< cache ratio are covered by the alias lists above and need no fallback here.
			static const std::array<const char*, 8> live_tok_names = {
				"strata:live_tok_s", "vllm:avg_generation_throughput_toks_per_s",
				"sglang:gen_throughput", nullptr
			};
			static const std::array<const char*, 8> context_names = {
				"strata:engine_max_context", "vllm:max_model_len", nullptr
			};
			if(engine.live_tok_s < 0)
				engine.live_tok_s = pick(sums, live_tok_names);
			if(engine.state.empty())
				engine.state = active_state(series, "strata:live_state");
			if(engine.context_tokens < 0)
				engine.context_tokens = pick(sums, context_names);

			//? Remember raw counter values for the next interval
			engine.prev_values.clear();
			for(const auto& [name, value] : sums)
				engine.prev_values[name] = value;
			engine.prev_times.clear();
			for(const auto& [name, value] : sums)
				engine.prev_times[name] = finished;
			engine.has_baseline = true;
		}

		{
			//? Discovery and engine disappearance change the row count after the initial
			//!< layout, so ask the main thread to reflow the panel instead of truncating it
			std::scoped_lock lock(mtx);
			engines = work;
			if(work.size() != layout_count.load()) {
				layout_count = work.size();
				layout_dirty = true;
			}
		}
	}

	//* ---------------------------------------------------------------- discovery

#ifdef __linux__
	namespace {
		//!< Never probed: everything privileged, plus local daemons that answer any request
		//!< (a redis reply or a web UI must not be mistaken for a metrics endpoint)
		constexpr std::array<int, 6> denied_ports = {22, 53, 67, 68, 111, 631};
		constexpr int max_probes_per_pass = 24;
		constexpr size_t sniff_bytes = 64U << 10;
		//!< Loopback services answer in milliseconds and anything that hangs is not an engine,
		//!< so discovery must not inherit the scrape timeout: 24 ports x 2 paths x 400ms left
		//!< the panel reading "scanning loopback..." for ~19s on a box with many listeners
		constexpr int discover_timeout_ms = 200;
		constexpr uint64_t discover_budget_ms = 1200;

		//!< Read a NUL-separated /proc file with a hard cap. Must stream rather than seek to
		//!< the end: /proc entries report st_size 0, so a tellg-based size reads nothing
		std::string read_capped(const std::string& path, size_t cap) {
			std::ifstream stream(path, std::ios::binary);
			if(not stream.is_open())
				return {};
			std::string out;
			std::array<char, 8192> buffer {};
			while(out.size() < cap and static_cast<bool>(stream.read(buffer.data(), buffer.size())))
				out.append(buffer.data(), static_cast<size_t>(stream.gcount()));
			out.append(buffer.data(), static_cast<size_t>(stream.gcount()));	//? final short read
			if(out.size() > cap)
				out.resize(cap);
			return out;
		}

		//* A running engine already publishes its own key where ps(1) can see it, so reuse it
		//!< instead of asking the user to hand-configure a token for a process on their own box
		std::string token_for_port(int port) {
			const auto port_str = std::to_string(port);
			const auto self = std::to_string(::getpid());
			DIR* dir = ::opendir("/proc");
			if(dir == nullptr)
				return {};
			std::string result;
			while(const auto* entry = ::readdir(dir)) {
				const std::string pid(entry->d_name);
				if(pid.empty() or pid == self or not std::all_of(pid.begin(), pid.end(),
					[](char c) { return std::isdigit(static_cast<unsigned char>(c)); }))
					continue;
				const auto base = "/proc/"s + pid;
				const auto cmdline = read_capped(base + "/cmdline", 32U << 10);
				if(cmdline.empty())
					continue;
				std::vector<std::string> args;
				for(size_t start = 0; start < cmdline.size(); ) {
					const auto end = cmdline.find('\0', start);
					const auto stop = end == std::string::npos ? cmdline.size() : end;
					if(stop > start)
						args.emplace_back(cmdline.substr(start, stop - start));
					start = stop + 1;
				}

				bool mentions = false;
				for(size_t i = 0; i < args.size() and not mentions; ++i) {
					if(args[i] == "--port" and i + 1 < args.size() and args[i + 1] == port_str)
						mentions = true;
					else if(args[i].starts_with("--port=") and args[i].substr(7) == port_str)
						mentions = true;
					else if(args[i].size() > port_str.size() + 1 and args[i].ends_with(":" + port_str))
						mentions = true;
				}
				if(not mentions)
					continue;

				for(size_t i = 0; i < args.size() and result.empty(); ++i) {
					if(args[i] == "--api-key" and i + 1 < args.size())
						result = args[i + 1];
					else if(args[i].starts_with("--api-key="))
						result = args[i].substr(10);
				}
				if(not result.empty())
					break;

				const auto env = read_capped(base + "/environ", 128U << 10);
				for(const char* name : {"VLLM_API_KEY=", "OPENAI_API_KEY=", "API_KEY="}) {
					const auto pos = env.find(name);
					if(pos == std::string::npos)
						continue;
					const auto from = pos + std::char_traits<char>::length(name);
					const auto to = env.find('\0', from);
					result = env.substr(from, to == std::string::npos ? std::string::npos : to - from);
					break;
				}
				if(not result.empty())
					break;

				//? Engines commonly keep the key in the config file named on their own
				//!< command line (Strata: --config <file>.json with an "api_key" member),
				//!< resolved against the process working directory
				std::string err;
				for(size_t i = 0; i < args.size() and result.empty(); ++i) {
					auto file = args[i];
					if(file.starts_with("--config="))
						file = file.substr(9);
					else if(file == "--config" and i + 1 < args.size())
						file = args[i + 1];
					else if(not file.ends_with(".json"))
						continue;
					if(file.empty() or file.front() != '/') {
						std::array<char, 4096> cwd {};
						const auto len = ::readlink((base + "/cwd").c_str(), cwd.data(), cwd.size() - 1);
						if(len > 0) {
							cwd[static_cast<size_t>(len)] = '\0';
							file = cwd.data() + "/"s + file;
						}
					}
					result = read_token_file(file, err);
				}
				if(not result.empty())
					break;
			}
			::closedir(dir);
			return result;
		}

		//* Well-known inference ports, tried first so a useful row lands in the first pass
		auto port_rank(int port) -> int {
			static const std::array<int, 10> preferred = {8013, 8000, 30000, 30001, 8080, 8081, 11434, 5000, 3000, 8001};
			const auto it = std::find(preferred.begin(), preferred.end(), port);
			return it == preferred.end() ? static_cast<int>(preferred.size()) : static_cast<int>(it - preferred.begin());
		}

		//* /proc/net/tcp stores addresses as host-order hex words: 0100007F is 127.0.0.1 and
		//!< 00000000 is 0.0.0.0 (wildcard); a wildcard listener is still a process on this host
		auto is_local_address(const std::string& address, bool ipv6) -> bool {
			if(ipv6)
				return address == "00000000000000000000000001000000"	//? ::1
					or address == "00000000000000000000000000000000";	//? :: wildcard
			return address == "0100007F" or address == "00000000";
		}

		//* Ports with a socket LISTENing on loopback, or on every interface of this host.
		//!< A wildcard listener is still a local process; what keeps the LAN out of scope is
		//!< that every probe below dials 127.0.0.1/::1, never the advertised address.
		void listen_ports(const char* file, bool ipv6, std::vector<int>& out) {
			std::ifstream stream(file);
			if(not stream.is_open())
				return;
			std::string line;
			while(std::getline(stream, line)) {
				std::istringstream fields(line);
				std::string seq, local, remote, state;
				if(not (fields >> seq >> local >> remote >> state))
					continue;
				if(state != "0A")	//? TCP_LISTEN
					continue;
				const auto colon = local.rfind(':');
				if(colon == std::string::npos)
					continue;
				if(not is_local_address(local.substr(0, colon), ipv6))
					continue;
				const auto port = static_cast<int>(std::strtoul(local.c_str() + colon + 1, nullptr, 16));
				if(port < 1024 or std::find(denied_ports.begin(), denied_ports.end(), port) != denied_ports.end())
					continue;
				out.push_back(port);
			}
		}

		//* The exposition format starts with "name value" or "name{labels} value"; JSON, HTML
		//!< and protocol error replies all fail the leading-character test
		auto looks_prometheus(const std::string& body) -> bool {
			std::istringstream lines(body);
			std::string line;
			int checked = 0;
			while(std::getline(lines, line) and ++checked < 40) {
				if(not line.empty() and line.back() == '\r')
					line.pop_back();
				if(line.empty() or line.front() == '#')
					continue;
				const auto valid = [](char c) {
					return std::isalnum(static_cast<unsigned char>(c)) or c == '_' or c == ':';
				};
				size_t i = 0;
				while(i < line.size() and valid(line[i]))
					++i;
				if(i == 0 or not std::isalpha(static_cast<unsigned char>(line[0])))
					return false;
				return i < line.size() and (line[i] == ' ' or line[i] == '\t' or line[i] == '{');
			}
			return false;
		}

		//* A 401 on /metrics alone proves nothing: an auth middleware 401s every route. Engines
		//!< worth showing are those that answer /health successfully without credentials —
		//!< vLLM does (200), whereas a generic API server 401s it too. Measured on two boxes:
		//!< vLLM 30000/30001 -> /health 200, opencode serve 4096 -> /health 401, so requiring a
		//!< 2xx here separates real engines from local web services that merely gate everything.
		auto has_engine_health(const std::string& port) -> bool {
			Target probe;
			probe.url = "http://127.0.0.1:"s + port + "/health";
			if(not parse_url(probe.url, probe))
				return false;
			Options sniff = options;
			sniff.max_bytes = 4U << 10;
			sniff.timeout_ms = discover_timeout_ms;
			std::string body;
			int status = 0;
			std::string err;
			http_get(probe, sniff, body, status, err, true);
			return status >= 200 and status < 300;
		}

		//* The first model name in the body, used only to recognise that two listeners are
		//!< the same engine (an OpenAI-compatible proxy republishes its upstream's labels)
		std::string model_hint(const std::string& body) {
			for(const char* name : {"model_name=\"", "model=\""}) {
				const auto pos = body.find(name);
				if(pos == std::string::npos)
					continue;
				const auto from = pos + std::char_traits<char>::length(name);
				const auto to = body.find('"', from);
				if(to == std::string::npos or to == from)
					continue;
				return body.substr(from, to - from);
			}
			return {};
		}

		auto engine_family(const std::string& body) -> std::string {
			static const std::array<std::pair<const char*, const char*>, 5> families = {{
				{"strata:", "strata"}, {"vllm:", "vllm"}, {"sglang:", "sglang"},
				{"llamacpp:", "llamacpp"}, {"tgi_", "tgi"}
			}};
			for(const auto& [prefix, name] : families)
				if(body.find(prefix) != std::string::npos)
					return name;
			return "other";	//? short on purpose: labels are "<family>-<port>" in a 12-column field
		}
	}
#endif

	//* Probe loopback for engines exposing Prometheus metrics, so a fresh install needs
	//!< no configuration. Linux only; elsewhere this is a no-op and manual endpoints work
	//!< exactly as before.
	void discover() {
#ifdef __linux__
		std::vector<int> ports;
		listen_ports("/proc/net/tcp", false, ports);
		listen_ports("/proc/net/tcp6", true, ports);
		std::ranges::sort(ports, [](int a, int b) {
			const auto ra = port_rank(a), rb = port_rank(b);
			return ra != rb ? ra < rb : a < b;
		});
		ports.erase(std::unique(ports.begin(), ports.end()), ports.end());
		const auto pass_deadline = now_ms() + discover_budget_ms;
		//!< Ports already judged and set aside, so a dropped listener is not re-probed forever
		static std::set<int> ignored_ports;
		//!< The listener chosen to represent each engine: one row per family+model, lowest port
		static std::map<std::string, int> primary_ports;


		//? 'ports' is ordered by probe priority, not numerically, so membership has to be
		//!< tested against a numerically sorted copy: binary_search on the priority order
		//!< reported live ports as absent and deleted the engines bound to high port numbers
		auto listening = ports;
		std::ranges::sort(listening);

		//? An engine that stopped listening disappears instead of leaving a dead row
		std::erase_if(work, [&](const Engine& e) {
			if(not e.target.discovered)
				return false;
			const auto port = static_cast<int>(std::atoi(e.target.port.c_str()));
			if(std::binary_search(listening.begin(), listening.end(), port))
				return false;
			//? Free the slot so a replacement on another port can be elected
			if(const auto it = primary_ports.find(e.target.engine_key);
				it != primary_ports.end() and it->second == port)
				primary_ports.erase(it);
			ignored_ports.erase(port);
			return true;
		});

		const char* env_token = options.allow_token_env ? std::getenv("BTOP_LLM_TOKEN") : nullptr;

		//? A reverse proxy in front of an engine shares its key but does not name it in
		//!< its own cmdline, so keys harvested from one local port are tried on the others
		static std::vector<std::string> known_tokens;
		const auto remember_token = [](const std::string& token) {
			if(token.empty() or token.size() > 512 or known_tokens.size() >= 8)
				return;
			if(std::find(known_tokens.begin(), known_tokens.end(), token) == known_tokens.end())
				known_tokens.push_back(token);
		};
		int probed = 0;
		for(const int port : ports) {
			const auto port_str = std::to_string(port);
			if(std::any_of(work.begin(), work.end(), [&](const Engine& e) {
				return e.target.port == port_str and not e.target.needs_auth
					and e.target.host.find('.') != std::string::npos;
			}))
				continue;
			if(++probed > max_probes_per_pass or now_ms() > pass_deadline)
				break;
			if(ignored_ports.contains(port))
				continue;

			//? Deliberately just the two conventional paths: looser guesses turn any local
			//!< service that 401s on some /api/... route into a bogus engine row
			static const std::array<const char*, 2> paths = {"/metrics", "/metrics/prometheus"};
			int auth_rows = std::count_if(work.begin(), work.end(),
				[](const Engine& e) { return e.target.needs_auth; });
			bool found = false;
			for(const char* path : paths) {
				Target probe;
				probe.url = "http://127.0.0.1:"s + port_str + path;
				if(not parse_url(probe.url, probe))
					continue;
				probe.accept = "text/plain; version=0.0.4";
				if(env_token != nullptr and *env_token != '\0')
					probe.token = env_token;

				Options sniff = options;
				sniff.max_bytes = sniff_bytes;
				sniff.timeout_ms = discover_timeout_ms;
				std::string body;
				int status = 0;
				std::string err;
				if(http_get(probe, sniff, body, status, err, true) and looks_prometheus(body)) {
					const auto family = engine_family(body);
					if(family == "other") {
						//? Metrics but no engine behind them (a model swapper exposing only
						//!< process counters): kept out of the panel, still reachable manually
						ignored_ports.insert(port);
						found = true;
						break;
					}
					std::string key;
					if(const auto model = model_hint(body); not model.empty()) {
						key = family + "|" + model;
						if(const auto it = primary_ports.find(key);
							it != primary_ports.end() and it->second != port) {
							ignored_ports.insert(port);	//? its proxy or a duplicate: keep the main one
							found = true;
							break;
						}
						primary_ports[key] = port;
					}
					Engine e;
					e.target = probe;
					e.target.label = family + "-" + port_str;
					e.target.engine_key = key;
					e.target.discovered = true;
					work.push_back(std::move(e));
					found = true;
					break;
				}
				if(status == 401) {
					//? First try to authenticate with the key the engine itself runs with, so a
					//!< protected /metrics still shows up live instead of as a dead row
					std::vector<std::string> candidates;
					if(env_token != nullptr and *env_token != '\0')
						candidates.emplace_back(env_token);
					else
						candidates = known_tokens;	//? a sibling port's key often works here
					if(options.token_scan) {
						const auto scanned = token_for_port(port);
						if(not scanned.empty()
							and std::find(candidates.begin(), candidates.end(), scanned) == candidates.end())
							candidates.push_back(scanned);
					}
					for(const auto& candidate : candidates) {
						Target retry = probe;
						retry.token = candidate;
						std::string auth_body;
						int auth_status = 0;
						std::string auth_err;
						if(http_get(retry, sniff, auth_body, auth_status, auth_err, true) and looks_prometheus(auth_body)) {
							remember_token(candidate);
							const auto family = engine_family(auth_body);
							std::string key;
							if(family == "other") {
								ignored_ports.insert(port);
								found = true;
								break;
							}
							if(const auto model = model_hint(auth_body); not model.empty()) {
								key = family + "|" + model;
								if(const auto it = primary_ports.find(key);
									it != primary_ports.end() and it->second != port) {
									ignored_ports.insert(port);
									found = true;
									break;
								}
								primary_ports[key] = port;
							}
							Engine e;
							e.target = retry;
							e.target.label = family + "-" + port_str;
							e.target.engine_key = key;
							e.target.discovered = true;
							std::erase_if(work, [&](const Engine& other) {
								return other.target.needs_auth and other.target.port == port_str;
							});
							work.push_back(std::move(e));
							found = true;
							break;
						}
					}
				}
				if(found)
					break;	//? authenticated already: no second row for this port
				if(status == 401 and not found and auth_rows < 4 and has_engine_health(port_str)) {
					//? Something is there but wants a token: surface it rather than hide it,
					//!< a few at most so an unrelated 401 cannot flood the panel
					++auth_rows;
					Engine e;
					e.target = probe;
					e.target.label = "auth-" + port_str;
					e.target.discovered = true;
					e.target.valid = false;
					e.error = "needs a token: set llm_endpoints or $BTOP_LLM_TOKEN";
					work.push_back(std::move(e));
					found = true;
					break;
				}
			}
			(void) found;
		}

		//? Known engines lead, unauthenticated or unknown-family endpoints go last
		std::stable_sort(work.begin(), work.end(), [](const Engine& a, const Engine& b) {
			const auto rank = [](const Engine& e) {
				if(e.target.needs_auth)
					return 2;
				const auto& label = e.target.label;
				return label.starts_with("other-") or label.starts_with("prometheus-") ? 1 : 0;
			};
			return rank(a) < rank(b);
		});
#endif
	}

	//* ---------------------------------------------------------------- thread

	static std::thread collector;
	static std::atomic<bool> running{false};
	std::atomic<uint64_t> scrape_seq{0};
	static uint64_t menu_seq{};

	//* True exactly once per completed scrape round, so an open overlay repaints without the
	//!< collector thread ever naming Menu::
	bool new_data_since_menu_draw() {
		const auto current = scrape_seq.load();
		if (current == menu_seq)
			return false;
		menu_seq = current;
		return true;
	}

	double Engine::display_tok_s() const {
		return (thinking() and options.hide_tok_s_thinking) ? -1 : current_tok_s();
	}

	void start() {
		if(running or (work.empty() and not options.autodetect))
			return;
		running = true;
		collector = std::thread([] {
			uint64_t next_discovery = 0;	//? 0 forces an immediate first scan
			while(running) {
				if(options.autodetect and now_ms() >= next_discovery) {
					discover();
					next_discovery = now_ms() + std::max(2000, options.discover_ms);
				}
				collect();
				++scrape_seq;
				//? Sleep in slices so stop() stays responsive on quit
				for(int slept = 0; running and slept < options.interval_ms; slept += 50)
					std::this_thread::sleep_for(std::chrono::milliseconds(50));
			}
		});
	}

	void stop() {
		running = false;
		if(collector.joinable())
			collector.join();
	}

}
