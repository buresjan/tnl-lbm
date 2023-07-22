#pragma once

#include <chrono>
#include <string>

#include <fmt/chrono.h>

static std::string timestamp()
{
	std::chrono::time_point<std::chrono::system_clock> now = std::chrono::system_clock::now();

	// convert from C++11 API to the C struct (milliseconds are dropped)
	std::time_t secs = std::chrono::system_clock::to_time_t(now);
	std::tm date = fmt::localtime(secs);

	// number of milliseconds since the last whole seconds (must be in C++11)
	auto last_sec = std::chrono::system_clock::from_time_t(std::mktime(&date));
	int milliseconds = std::chrono::duration_cast<std::chrono::milliseconds>(now - last_sec).count();

	return fmt::format("{:%Y|%m|%d} {:%H:%M:%S}.{:03d}", date, date, milliseconds);
}
