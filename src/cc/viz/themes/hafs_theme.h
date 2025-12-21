#pragma once

/// HAFS Theme for ImGui
/// 
/// Provides a consistent dark theme matching the hafs TUI aesthetic.

#include "imgui.h"

namespace hafs::viz::themes {

/// Apply the HAFS dark theme to ImGui.
inline void ApplyHafsTheme() {
  ImGuiStyle& style = ImGui::GetStyle();

  // Rounding
  style.WindowRounding = 4.0f;
  style.FrameRounding = 2.0f;
  style.GrabRounding = 2.0f;
  style.PopupRounding = 4.0f;
  style.ScrollbarRounding = 2.0f;
  style.TabRounding = 4.0f;

  // Padding
  style.WindowPadding = ImVec2(8, 8);
  style.FramePadding = ImVec2(5, 4);
  style.ItemSpacing = ImVec2(8, 4);
  style.ItemInnerSpacing = ImVec2(4, 4);

  // Colors
  ImVec4* colors = style.Colors;

  // Backgrounds
  colors[ImGuiCol_WindowBg] = ImVec4(0.10f, 0.10f, 0.12f, 1.00f);
  colors[ImGuiCol_ChildBg] = ImVec4(0.08f, 0.08f, 0.10f, 1.00f);
  colors[ImGuiCol_PopupBg] = ImVec4(0.12f, 0.12f, 0.14f, 0.95f);

  // Headers
  colors[ImGuiCol_Header] = ImVec4(0.20f, 0.25f, 0.30f, 1.00f);
  colors[ImGuiCol_HeaderHovered] = ImVec4(0.25f, 0.30f, 0.40f, 1.00f);
  colors[ImGuiCol_HeaderActive] = ImVec4(0.30f, 0.40f, 0.50f, 1.00f);

  // Buttons
  colors[ImGuiCol_Button] = ImVec4(0.15f, 0.35f, 0.55f, 1.00f);
  colors[ImGuiCol_ButtonHovered] = ImVec4(0.20f, 0.45f, 0.70f, 1.00f);
  colors[ImGuiCol_ButtonActive] = ImVec4(0.25f, 0.50f, 0.80f, 1.00f);

  // Frame (inputs, etc)
  colors[ImGuiCol_FrameBg] = ImVec4(0.14f, 0.14f, 0.16f, 1.00f);
  colors[ImGuiCol_FrameBgHovered] = ImVec4(0.18f, 0.18f, 0.22f, 1.00f);
  colors[ImGuiCol_FrameBgActive] = ImVec4(0.22f, 0.22f, 0.28f, 1.00f);

  // Tabs
  colors[ImGuiCol_Tab] = ImVec4(0.14f, 0.14f, 0.16f, 1.00f);
  colors[ImGuiCol_TabHovered] = ImVec4(0.25f, 0.35f, 0.50f, 1.00f);
  colors[ImGuiCol_TabActive] = ImVec4(0.20f, 0.30f, 0.45f, 1.00f);

  // Titles
  colors[ImGuiCol_TitleBg] = ImVec4(0.08f, 0.08f, 0.10f, 1.00f);
  colors[ImGuiCol_TitleBgActive] = ImVec4(0.12f, 0.15f, 0.20f, 1.00f);

  // Scrollbar
  colors[ImGuiCol_ScrollbarBg] = ImVec4(0.08f, 0.08f, 0.10f, 1.00f);
  colors[ImGuiCol_ScrollbarGrab] = ImVec4(0.25f, 0.25f, 0.30f, 1.00f);
  colors[ImGuiCol_ScrollbarGrabHovered] = ImVec4(0.35f, 0.35f, 0.40f, 1.00f);
  colors[ImGuiCol_ScrollbarGrabActive] = ImVec4(0.45f, 0.45f, 0.50f, 1.00f);

  // Separator
  colors[ImGuiCol_Separator] = ImVec4(0.25f, 0.25f, 0.30f, 1.00f);
  colors[ImGuiCol_SeparatorHovered] = ImVec4(0.35f, 0.40f, 0.50f, 1.00f);
  colors[ImGuiCol_SeparatorActive] = ImVec4(0.45f, 0.55f, 0.70f, 1.00f);

  // Resize grip
  colors[ImGuiCol_ResizeGrip] = ImVec4(0.25f, 0.25f, 0.30f, 0.50f);
  colors[ImGuiCol_ResizeGripHovered] = ImVec4(0.40f, 0.50f, 0.70f, 0.70f);
  colors[ImGuiCol_ResizeGripActive] = ImVec4(0.50f, 0.60f, 0.80f, 0.90f);

  // Plot colors
  colors[ImGuiCol_PlotLines] = ImVec4(0.40f, 0.70f, 1.00f, 1.00f);
  colors[ImGuiCol_PlotLinesHovered] = ImVec4(0.60f, 0.85f, 1.00f, 1.00f);
  colors[ImGuiCol_PlotHistogram] = ImVec4(0.30f, 0.85f, 0.70f, 1.00f);
  colors[ImGuiCol_PlotHistogramHovered] = ImVec4(0.50f, 1.00f, 0.85f, 1.00f);

  // Text
  colors[ImGuiCol_Text] = ImVec4(0.90f, 0.90f, 0.92f, 1.00f);
  colors[ImGuiCol_TextDisabled] = ImVec4(0.50f, 0.50f, 0.55f, 1.00f);

  // Border
  colors[ImGuiCol_Border] = ImVec4(0.25f, 0.25f, 0.30f, 0.50f);
  colors[ImGuiCol_BorderShadow] = ImVec4(0.00f, 0.00f, 0.00f, 0.00f);

  // Menu bar
  colors[ImGuiCol_MenuBarBg] = ImVec4(0.12f, 0.12f, 0.14f, 1.00f);

  // Check mark
  colors[ImGuiCol_CheckMark] = ImVec4(0.40f, 0.80f, 0.60f, 1.00f);

  // Slider
  colors[ImGuiCol_SliderGrab] = ImVec4(0.30f, 0.55f, 0.80f, 1.00f);
  colors[ImGuiCol_SliderGrabActive] = ImVec4(0.40f, 0.65f, 0.90f, 1.00f);
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
