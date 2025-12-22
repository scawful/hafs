// SIMD-accelerated JSON loading using simdjson

#include "json_loader.h"

#ifdef HAFS_HAS_SIMDJSON
#include <simdjson.h>
#endif

#include <filesystem>
#include <fstream>
#include <sstream>

namespace fs = std::filesystem;

namespace hafs {

#ifdef HAFS_HAS_SIMDJSON

EmbeddingFileResult LoadEmbeddingFile(const std::string& path) {
  EmbeddingFileResult result;
  result.dimension = 0;
  result.success = false;

  try {
    simdjson::ondemand::parser parser;
    simdjson::padded_string json = simdjson::padded_string::load(path);
    simdjson::ondemand::document doc = parser.iterate(json);

    // Try format 1: {"embeddings": [...]}
    auto embeddings_array = doc["embeddings"];
    if (embeddings_array.error() == simdjson::SUCCESS) {
      for (auto item : embeddings_array.get_array()) {
        // Get ID
        std::string_view id_sv = item["id"].get_string();
        result.ids.emplace_back(id_sv);

        // Get vector (try "vector" then "embedding")
        std::vector<float> vec;
        auto vector_arr = item["vector"];
        if (vector_arr.error() != simdjson::SUCCESS) {
          vector_arr = item["embedding"];
        }

        for (auto val : vector_arr.get_array()) {
          vec.push_back(static_cast<float>(val.get_double()));
        }

        if (result.dimension == 0) {
          result.dimension = vec.size();
        }
        result.embeddings.push_back(std::move(vec));
      }
      result.success = true;
      return result;
    }

    // Try format 2: [{...}, {...}] (array at root)
    doc.rewind();
    auto root_array = doc.get_array();
    if (root_array.error() == simdjson::SUCCESS) {
      for (auto item : root_array) {
        std::string_view id_sv = item["id"].get_string();
        result.ids.emplace_back(id_sv);

        std::vector<float> vec;
        auto vector_arr = item["vector"];
        if (vector_arr.error() != simdjson::SUCCESS) {
          vector_arr = item["embedding"];
        }

        for (auto val : vector_arr.get_array()) {
          vec.push_back(static_cast<float>(val.get_double()));
        }

        if (result.dimension == 0) {
          result.dimension = vec.size();
        }
        result.embeddings.push_back(std::move(vec));
      }
      result.success = true;
      return result;
    }

    result.error = "Unknown JSON format";
  } catch (const simdjson::simdjson_error& e) {
    result.error = std::string("simdjson error: ") + e.what();
  } catch (const std::exception& e) {
    result.error = std::string("Error: ") + e.what();
  }

  return result;
}

#else  // No simdjson, use fallback (returns error)

EmbeddingFileResult LoadEmbeddingFile(const std::string& path) {
  EmbeddingFileResult result;
  result.dimension = 0;
  result.success = false;
  result.error = "simdjson not available - use Python JSON loader";
  return result;
}

#endif  // HAFS_HAS_SIMDJSON

EmbeddingBatchResult LoadEmbeddingFiles(const std::vector<std::string>& paths) {
  EmbeddingBatchResult result;
  result.dimension = 0;
  result.files_loaded = 0;
  result.files_failed = 0;

  for (const auto& path : paths) {
    auto file_result = LoadEmbeddingFile(path);
    if (file_result.success) {
      if (result.dimension == 0) {
        result.dimension = file_result.dimension;
      }
      result.ids.insert(result.ids.end(), file_result.ids.begin(),
                        file_result.ids.end());
      result.embeddings.insert(result.embeddings.end(),
                               file_result.embeddings.begin(),
                               file_result.embeddings.end());
      result.files_loaded++;
    } else {
      result.files_failed++;
      result.errors.push_back(path + ": " + file_result.error);
    }
  }

  return result;
}

EmbeddingBatchResult LoadEmbeddingsFromDirectory(const std::string& dir_path) {
  std::vector<std::string> paths;

  try {
    for (const auto& entry : fs::directory_iterator(dir_path)) {
      if (entry.path().extension() == ".json") {
        paths.push_back(entry.path().string());
      }
    }
  } catch (const std::exception& e) {
    EmbeddingBatchResult result;
    result.dimension = 0;
    result.files_loaded = 0;
    result.files_failed = 1;
    result.errors.push_back(std::string("Directory error: ") + e.what());
    return result;
  }

  return LoadEmbeddingFiles(paths);
}

// Python bindings
void RegisterIOBindings(py::module_& m) {
  // Check if simdjson is available
#ifdef HAFS_HAS_SIMDJSON
  m.attr("__has_simdjson__") = true;
#else
  m.attr("__has_simdjson__") = false;
#endif

  // Load single file - returns (ids, embeddings, dimension, success, error)
  m.def(
      "load_embedding_file",
      [](const std::string& path) {
        auto result = LoadEmbeddingFile(path);

        // Convert embeddings to numpy array
        py::array_t<float> embeddings;
        if (result.success && !result.embeddings.empty()) {
          size_t n = result.embeddings.size();
          size_t d = result.dimension;
          embeddings = py::array_t<float>({n, d});
          auto buf = embeddings.mutable_unchecked<2>();
          for (size_t i = 0; i < n; i++) {
            for (size_t j = 0; j < d; j++) {
              buf(i, j) = result.embeddings[i][j];
            }
          }
        }

        return py::make_tuple(result.ids, embeddings, result.dimension,
                              result.success, result.error);
      },
      py::arg("path"),
      "Load embeddings from a JSON file. Returns (ids, embeddings, dimension, "
      "success, error)");

  // Load from directory
  m.def(
      "load_embeddings_from_directory",
      [](const std::string& dir_path) {
        auto result = LoadEmbeddingsFromDirectory(dir_path);

        py::array_t<float> embeddings;
        if (!result.embeddings.empty()) {
          size_t n = result.embeddings.size();
          size_t d = result.dimension;
          embeddings = py::array_t<float>({n, d});
          auto buf = embeddings.mutable_unchecked<2>();
          for (size_t i = 0; i < n; i++) {
            for (size_t j = 0; j < d; j++) {
              buf(i, j) = result.embeddings[i][j];
            }
          }
        }

        return py::make_tuple(result.ids, embeddings, result.dimension,
                              result.files_loaded, result.files_failed,
                              result.errors);
      },
      py::arg("dir_path"),
      "Load all embeddings from a directory. Returns (ids, embeddings, "
      "dimension, files_loaded, files_failed, errors)");
}

}  // namespace hafs
