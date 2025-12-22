#pragma once

#include "../../models/state.h"
#include <functional>
#include <string>

namespace hafs::viz::ui {

// A clean, log-based chat interface.
void RenderChatPanel(AppState& state, std::function<void(const std::string&, const std::string&, const std::string&)> log_callback);

} // namespace hafs::viz::ui
