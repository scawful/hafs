#include "chat_panel.h"
#include "../core.h"
#include "../../icons.h"
#include <imgui.h>
#include <vector>

namespace hafs::viz::ui {

void RenderChatPanel(AppState& state, std::function<void(const std::string&, const std::string&, const std::string&)> log_callback) {
  // Toolbar
  ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, ImVec2(4, 4));
  if (ImGui::Button(ICON_MD_DELETE " Clear")) {
      state.logs.clear();
  }
  ImGui::SameLine();
  ImGui::SetNextItemWidth(150);
  ImGui::InputTextWithHint("##LogFilter", ICON_MD_FILTER_LIST " Filter...", state.log_filter.data(), state.log_filter.size());
  
  // Agent Selector
  ImGui::SameLine();
  std::vector<const char*> agent_labels;
  agent_labels.push_back("All Channels");
  for (const auto& agent : state.agents) {
    agent_labels.push_back(agent.name.c_str());
  }
  if (state.log_agent_index < 0 || state.log_agent_index >= static_cast<int>(agent_labels.size())) {
    state.log_agent_index = 0;
  }
  ImGui::SetNextItemWidth(120);
  ImGui::Combo("##Target", &state.log_agent_index, agent_labels.data(), static_cast<int>(agent_labels.size()));
  ImGui::PopStyleVar();

  ImGui::Separator();

  // Log Table
  float footer_height = 40.0f; // Input bar height
  if (ImGui::BeginTable("ChatLogTable", 3, ImGuiTableFlags_ScrollY | ImGuiTableFlags_RowBg | ImGuiTableFlags_BordersInnerH | ImGuiTableFlags_Resizable, ImVec2(0, ImGui::GetContentRegionAvail().y - footer_height))) {
    ImGui::TableSetupColumn("Source", ImGuiTableColumnFlags_WidthFixed, 100.0f);
    ImGui::TableSetupColumn("Message", ImGuiTableColumnFlags_WidthStretch);
    ImGui::TableSetupColumn("Time", ImGuiTableColumnFlags_WidthFixed, 60.0f);
    
    // Auto-scroll logic
    if (ImGui::GetScrollY() >= ImGui::GetScrollMaxY()) {
        ImGui::SetScrollHereY(1.0f);
    }
    
    std::string filter(state.log_filter.data());
    for (const auto& entry : state.logs) {
      // Filtering
      if (state.log_agent_index > 0 && entry.agent != agent_labels[state.log_agent_index]) continue;
      if (!filter.empty()) {
        if (entry.message.find(filter) == std::string::npos && entry.agent.find(filter) == std::string::npos) continue;
      }

      ImGui::TableNextRow();
      
      // Source Column
      ImGui::TableNextColumn();
      ImVec4 color = ImVec4(0.7f, 0.7f, 0.7f, 1.0f);
      if (entry.kind == "user") color = ImVec4(0.4f, 0.8f, 1.0f, 1.0f);
      else if (entry.kind == "system") color = ImVec4(1.0f, 0.6f, 0.2f, 1.0f);
      else if (entry.kind == "agent") color = ImVec4(0.4f, 1.0f, 0.6f, 1.0f);
      
      ImGui::TextColored(color, "%s", entry.agent.c_str());

      // Message Column
      ImGui::TableNextColumn();
      ImGui::PushTextWrapPos(0.0f);
      ImGui::TextUnformatted(entry.message.c_str());
      ImGui::PopTextWrapPos();

      // Time Column (Mock)
      ImGui::TableNextColumn();
      ImGui::TextDisabled("12:00"); 
    }
    ImGui::EndTable();
  }

  // Input Area
  ImGui::Separator();
  ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, ImVec2(8, 6)); // Taller input
  ImGui::SetNextItemWidth(ImGui::GetContentRegionAvail().x - 70);
  if (ImGui::InputTextWithHint("##ChatInput", "Message swarm...", state.chat_input.data(), state.chat_input.size(), ImGuiInputTextFlags_EnterReturnsTrue)) {
     goto send_msg;
  }
  ImGui::SameLine();
  if (ImGui::Button("SEND")) {
  send_msg:
    std::string message(state.chat_input.data());
    if (!message.empty()) {
      std::string target = agent_labels[state.log_agent_index];
      if (log_callback) log_callback("user", message, "user");
      
      // Simulate response
      if (state.log_agent_index > 0) {
        if (log_callback) log_callback(target, "Processing: " + message, "agent");
      } else {
        if (log_callback) log_callback("Orchestrator", "Routing: " + message, "system");
      }
      state.chat_input[0] = '\0';
      ImGui::SetItemDefaultFocus();
      ImGui::SetKeyboardFocusHere(-1); // Auto focus back
    }
  }
  ImGui::PopStyleVar();
}

} // namespace hafs::viz::ui
