#include "Utils.h"



void *xmalloc(size_t size) {
    void *ptr=malloc(size);
    if(!ptr) {
        std::cout << "Malloc error!\n" << std::endl;
        exit(EXIT_FAILURE);
    }
    return ptr;
}

void *xcalloc(size_t nmemb, size_t size) {
    void *ptr=calloc(nmemb,size);
    if(!ptr) {
        std::cout << "Calloc error!\n" << std::endl;
        exit(EXIT_FAILURE);
    }
    return ptr;
}

void *xrealloc(void *ptr, size_t size) {
    ptr=realloc(ptr,size);
    if(!ptr) {
        std::cout << "Realloc error!\n" << std::endl;
        exit(EXIT_FAILURE);
    }
    return ptr;
}

nlohmann::json loadConfigurations(const std::string configPath)
{
    std::ifstream configStream(configPath);
    nlohmann::json config;
    configStream >> config;
    configStream.close();
    return config;
}

timestamp getCurrentTime()
{
    return std::chrono::steady_clock::now();
}

double calcTimeInterval(const timestamp begin, const timestamp end)
{
    return std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
}