/// HAFS Training Data Visualization - Main Entry Point
///
/// Usage: hafs_viz [data_path]
///   data_path: Path to training data directory (default: ~/.context/training)
///
/// Build:
///   cmake -B build -S src/cc -DHAFS_BUILD_VIZ=ON
///   cmake --build build
///
/// Keys:
///   F5 - Refresh data
///   Ctrl+Q - Quit
///   Ctrl+/ - Shortcut editor

#include <cstdlib>
#include <iostream>
#include <string>

#include "app.h"

int main(int argc, char* argv[]) {
  // Determine data path
  std::string data_path;

  if (argc > 1) {
    data_path = argv[1];
  } else {
    // Default to ~/.context/training
    const char* home = std::getenv("HOME");
    if (home) {
      data_path = std::string(home) + "/.context/training";
    } else {
      std::cerr << "Error: Cannot determine home directory\n";
      std::cerr << "Usage: " << argv[0] << " [data_path]\n";
      return 1;
    }
  }

  std::cout << "HAFS Training Data Visualization\n";
  std::cout << "Data path: " << data_path << "\n";
  std::cout << "Press F5 to refresh data\n\n";

  hafs::viz::App app(data_path);
  return app.Run();
}
