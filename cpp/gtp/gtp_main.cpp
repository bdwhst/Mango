// mango_gtp: GTP engine. M1: random legal-move player.
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <memory>
#include <string>

#include "gtp/gtp.h"

int main(int argc, char** argv) {
  int size = 9;
  float komi = 7.5f;
  uint64_t seed = 1;
  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    auto next = [&](const char* what) -> const char* {
      if (i + 1 >= argc) {
        std::cerr << "missing value for " << what << "\n";
        std::exit(2);
      }
      return argv[++i];
    };
    if (a == "--size") size = std::atoi(next("--size"));
    else if (a == "--komi") komi = static_cast<float>(std::atof(next("--komi")));
    else if (a == "--seed") seed = std::strtoull(next("--seed"), nullptr, 10);
    else if (a == "--help" || a == "-h") {
      std::cout << "usage: mango_gtp [--size N] [--komi K] [--seed S]\n";
      return 0;
    } else {
      std::cerr << "unknown option " << a << "\n";
      return 2;
    }
  }
  mango::GtpEngine engine(std::make_unique<mango::RandomPlayer>(seed), size, komi);
  return engine.run(std::cin, std::cout);
}
