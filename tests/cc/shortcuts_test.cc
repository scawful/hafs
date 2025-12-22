#include "ui/shortcuts.h"

#include <chrono>
#include <filesystem>
#include <fstream>

#include "absl/strings/str_cat.h"
#include "gtest/gtest.h"

namespace hafs {
namespace viz {
namespace ui {
namespace {

class ImGuiContextFixture : public ::testing::Test {
 protected:
  void SetUp() override { ctx_ = ImGui::CreateContext(); }
  void TearDown() override { ImGui::DestroyContext(ctx_); }

 private:
  ImGuiContext* ctx_ = nullptr;
};

void WriteFile(const std::filesystem::path& path, const std::string& contents) {
  std::ofstream file(path, std::ios::trunc);
  file << contents;
}

std::filesystem::path UniqueTempPath(const std::string& suffix) {
  auto stamp = std::to_string(
      std::chrono::high_resolution_clock::now().time_since_epoch().count());
  return std::filesystem::temp_directory_path() /
         absl::StrCat("hafs_shortcuts_", stamp, suffix);
}

TEST_F(ImGuiContextFixture, SaveAndLoadShortcuts) {
  std::filesystem::path temp_path = UniqueTempPath(".conf");

  ShortcutManager shortcuts;
  shortcuts.SetConfigPath(temp_path.string());

  ASSERT_TRUE(shortcuts.SaveToDisk());
  ASSERT_TRUE(std::filesystem::exists(temp_path));

  WriteFile(temp_path,
            "toggle_sample_review = Ctrl+Shift+K\n"
            "workspace_analysis = Ctrl+2\n");

  ASSERT_TRUE(shortcuts.LoadFromDisk());
  Shortcut sample = shortcuts.GetShortcut(ActionId::ToggleSampleReview);
  EXPECT_EQ(sample.key, ImGuiKey_K);
  EXPECT_TRUE(sample.ctrl);
  EXPECT_TRUE(sample.shift);

  Shortcut workspace = shortcuts.GetShortcut(ActionId::WorkspaceAnalysis);
  EXPECT_EQ(workspace.key, ImGuiKey_2);
  EXPECT_TRUE(workspace.ctrl);
}

TEST_F(ImGuiContextFixture, RejectsInvalidLines) {
  std::filesystem::path temp_path = UniqueTempPath("_bad.conf");

  ShortcutManager shortcuts;
  shortcuts.SetConfigPath(temp_path.string());

  WriteFile(temp_path, "unknown_action = Ctrl+Q\n");
  EXPECT_FALSE(shortcuts.LoadFromDisk());
  EXPECT_FALSE(shortcuts.GetLastError().empty());
}

TEST_F(ImGuiContextFixture, FormatShortcutLabels) {
  ShortcutManager shortcuts;
  Shortcut custom;
  custom.key = ImGuiKey_J;
  custom.ctrl = true;
  shortcuts.SetShortcut(ActionId::ToggleInspector, custom);

  std::string label =
      shortcuts.FormatShortcut(ActionId::ToggleInspector, ImGui::GetIO());
  EXPECT_NE(label.find("Ctrl"), std::string::npos);
  EXPECT_NE(label.find("J"), std::string::npos);
}

TEST_F(ImGuiContextFixture, UsesAbslForPaths) {
  std::filesystem::path temp_path =
      std::filesystem::temp_directory_path() /
      absl::StrCat("hafs_", "shortcuts_",
                   std::chrono::high_resolution_clock::now()
                       .time_since_epoch()
                       .count(),
                   ".conf");

  ShortcutManager shortcuts;
  shortcuts.SetConfigPath(temp_path.string());
  EXPECT_TRUE(shortcuts.SaveToDisk());
}

}  // namespace
}  // namespace ui
}  // namespace viz
}  // namespace hafs
