#pragma once

#include "Types.h"
#include "ConcurrentQueue.h"
#include "json.hpp"

using namespace TRT::Types;

void *xmalloc(size_t size);
void *xcalloc(size_t nmemb, size_t size);
void *xrealloc(void *ptr, size_t size);

nlohmann::json loadConfigurations(const std::string configPath);
timestamp getCurrentTime();
double calcTimeInterval(const timestamp begin, const timestamp end);
