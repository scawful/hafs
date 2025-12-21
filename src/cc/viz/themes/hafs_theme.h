#pragma once

#include "imgui.h"
#include "../app.h" // For ThemeProfile enum

namespace hafs::viz::themes {

/// Apply the HAFS dark theme to ImGui with specific profiles.
inline void ApplyHafsTheme(ThemeProfile profile = ThemeProfile::Cobalt) {
  ImGuiStyle& style = ImGui::GetStyle();

  // Rounding & Padding for a modern look
  style.WindowRounding = 6.0f;
  style.FrameRounding = 3.0f;
  style.GrabRounding = 3.0f;
  style.PopupRounding = 6.0f;
  style.ScrollbarRounding = 12.0f;
  style.TabRounding = 4.0f;
  style.WindowPadding = ImVec2(10, 10);
  style.FramePadding = ImVec2(6, 4);
  style.ItemSpacing = ImVec2(8, 6);

  ImVec4* colors = style.Colors;

  // Base background (Deep neutral)
  colors[ImGuiCol_WindowBg] = ImVec4(0.06f, 0.06f, 0.08f, 1.00f);
  colors[ImGuiCol_ChildBg] = ImVec4(0.05f, 0.05f, 0.07f, 0.85f); // Darker, less transparent
  colors[ImGuiCol_PopupBg] = ImVec4(0.08f, 0.08f, 0.10f, 0.98f);
  colors[ImGuiCol_Border] = ImVec4(1.0f, 1.0f, 1.0f, 0.08f);
  
  // Profile specific colors
  ImVec4 primary, secondary, accent;
  
  if (profile == ThemeProfile::Cobalt) {
      primary = ImVec4(0.0f, 0.48f, 1.0f, 1.00f);     // Vivid Azure
      secondary = ImVec4(0.12f, 0.14f, 0.18f, 1.00f); // Darker Midnight for contrast
      accent = ImVec4(0.0f, 0.85f, 1.0f, 1.0f);       // Neon Cyan
  } else if (profile == ThemeProfile::Amber) {
      primary = ImVec4(1.00f, 0.60f, 0.00f, 1.00f);   // Bright Orange
      secondary = ImVec4(0.20f, 0.10f, 0.05f, 1.00f); // Deep Rust
      accent = ImVec4(1.00f, 0.90f, 0.30f, 1.00f);    // Electric Gold
  } else { // Emerald
      primary = ImVec4(0.00f, 0.85f, 0.45f, 1.00f);   // Neon Green
      secondary = ImVec4(0.05f, 0.15f, 0.10f, 1.00f); // Dark Jungle
      accent = ImVec4(0.40f, 1.00f, 0.60f, 1.00f);    // Bright Mint
  }

  // Apply profile to components
  colors[ImGuiCol_Header] = secondary;
  colors[ImGuiCol_HeaderHovered] = primary;
  colors[ImGuiCol_HeaderActive] = accent;

  colors[ImGuiCol_Button] = secondary;
  colors[ImGuiCol_ButtonHovered] = primary;
  colors[ImGuiCol_ButtonActive] = accent;

  colors[ImGuiCol_FrameBg] = ImVec4(1.0f, 1.0f, 1.0f, 0.03f);
  colors[ImGuiCol_FrameBgHovered] = ImVec4(1.0f, 1.0f, 1.0f, 0.08f);
  colors[ImGuiCol_FrameBgActive] = ImVec4(1.0f, 1.0f, 1.0f, 0.12f);

  colors[ImGuiCol_Tab] = ImVec4(0.0f, 0.0f, 0.0f, 0.0f);
  colors[ImGuiCol_TabHovered] = primary;
  colors[ImGuiCol_TabActive] = secondary;
  colors[ImGuiCol_TabUnfocused] = ImVec4(0.0f, 0.0f, 0.0f, 0.0f);
  colors[ImGuiCol_TabUnfocusedActive] = secondary;

  colors[ImGuiCol_TitleBg] = colors[ImGuiCol_WindowBg];
  colors[ImGuiCol_TitleBgActive] = colors[ImGuiCol_WindowBg];
  colors[ImGuiCol_TitleBgCollapsed] = colors[ImGuiCol_WindowBg];

  colors[ImGuiCol_PlotLines] = primary;
  colors[ImGuiCol_PlotLinesHovered] = accent;
  colors[ImGuiCol_PlotHistogram] = primary;
  colors[ImGuiCol_PlotHistogramHovered] = accent;

  colors[ImGuiCol_Text] = ImVec4(0.95f, 0.95f, 1.00f, 1.00f);
  colors[ImGuiCol_TextDisabled] = ImVec4(0.50f, 0.50f, 0.50f, 1.00f);
  
  colors[ImGuiCol_Separator] = colors[ImGuiCol_Border];
}

/// Apply a light theme variant.
inline void ApplyHafsLightTheme() {
  ImGui::StyleColorsLight();
  ImGuiStyle& style = ImGui::GetStyle();

  style.WindowRounding = 4.0f;
  style.FrameRounding = 2.0f;
  style.GrabRounding = 2.0f;

  ImVec4* colors = style.Colors;
  colors[ImGuiCol_PlotLines] = ImVec4(0.20f, 0.50f, 0.80f, 1.00f);
  colors[ImGuiCol_PlotHistogram] = ImVec4(0.20f, 0.70f, 0.50f, 1.00f);
}

}  // namespace hafs::viz::themes
