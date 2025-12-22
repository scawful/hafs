# Agent Message Delivery via halext-org

This document proposes a simple system for delivering HAFS agent messages
through halext-org, with downstream delivery to iOS and terminal mail.

## Goals

- Agents can send you a message with one API call.
- halext-org persists messages and streams them to clients.
- iOS gets push notifications when the app is offline.
- Terminal mail receives a copy for archival and CLI workflows.

## Architecture

1. **HAFS agent -> halext-org API**
   - Use `POST /messages/quick` to send a message to your username.
   - Alternatively target a specific conversation with
     `POST /conversations/{conversation_id}/messages`.

2. **halext-org -> WebSocket**
   - Backend already broadcasts new messages to the conversation channel.
   - iOS can subscribe and update the UI immediately.

3. **halext-org -> Push (iOS)**
   - Add APNS support in the backend and store device tokens per user.
   - On new message, enqueue a push job.

4. **halext-org or HAFS -> Terminal Mail**
   - Either send mail from halext-org via SMTP
   - Or have HAFS send `mailx`/`sendmail` locally as a fallback.

## Required Additions

### 1) Device Token Registration (iOS -> backend)

- iOS registers for remote notifications.
- On token refresh, call a new endpoint:
  - `POST /devices/register` (payload: device token, platform, app version)

### 2) Backend Push Worker

- Add a `device_tokens` table keyed by user id.
- Add a background job queue (Celery/RQ/async task).
- On message create, enqueue a push job with:
  - title: "HAFS"
  - body: message summary
  - data: `{ conversation_id, message_id }`

### 3) iOS Notification Handling

- Implement `didRegisterForRemoteNotificationsWithDeviceToken` and
  `didFailToRegisterForRemoteNotificationsWithError`.
- Route taps to the corresponding conversation/message.

### 4) Terminal Mail Fallback

Two options:

A) **Server-side SMTP** (recommended)
- Configure SMTP credentials in halext-org.
- On message create, send email to `alerts@` or your mailbox.

B) **Local mail via HAFS**
- Add a notifier adapter that calls `mailx` or `sendmail` locally.
- Triggered by message priority (warning/critical).

## Suggested Message Payload

```json
{
  "title": "HAFS Alert",
  "body": "Training run stalled on GPU host",
  "severity": "warning",
  "tags": ["training", "gpu"],
  "metadata": {
    "conversation_id": 42,
    "message_id": 1337
  }
}
```

## Minimal MVP

1. HAFS agent sends `POST /messages/quick` to your username.
2. iOS app receives via WebSocket when open.
3. Add APNS token registration and push sending for offline delivery.
4. Add SMTP mail from halext-org for a terminal-friendly copy.

## Notes

- Keep routing rules in `~/.config/hafs/config.toml` (local overrides).
- If using local mail, document the required MTA setup in your plugin repo.
