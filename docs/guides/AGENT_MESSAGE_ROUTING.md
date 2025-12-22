# Agent Message Routing (Halext + Terminal Mail)

This guide describes a lightweight notification path for hAFS agents:

- **Primary:** Send agent updates into Halext message threads (seen on iOS app).
- **Fallback:** Send plain terminal mail for SSH-only sessions.

The delivery is implemented by user plugins (see `docs/plugins/QUICK_START.md`).

## Architecture

1. hAFS agent emits a short status update.
2. Plugin script posts a message to Halext (`/messages/quick`).
3. iOS app receives the message in the "Messages" UI.
4. Optional local mail message is delivered for terminal-only workflows.

## Plugin Setup (Generic)

In your plugin `config.toml`, configure a notification block:

```toml
[notify.halext]
enabled = true
api_base = "https://api.halext.org"
sender_username = "hafs-agent"
target_username = "your-username"
token_env = "HALEXT_AGENT_TOKEN"

[notify.terminal_mail]
enabled = true
command = "mail"
to = "your-user"
```

Set the token in your shell profile or CI environment:

```bash
export HALEXT_AGENT_TOKEN="eyJhbGciOi..."
```

## Sending a Message

Your plugin can expose a helper (example):

```bash
notify_agent_message.py "Campaign finished with 1.2K samples"
```

## Notes

- The Halext endpoint currently expects a Bearer token; use a dedicated agent account.
- Terminal mail uses the configured command (`mail`, `mailx`, or similar) and depends on local MTA setup.
- Future enhancement: add a sync-token-protected agent inbox endpoint to avoid storing JWTs in plugin config.
