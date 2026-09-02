---
name: calendar
description: "Manage calendar events and scheduling via MCP tools."
version: 1.0.0
author: Hermes Agent
license: MIT
platforms: [linux, macos, windows]
metadata:
  hermes:
    tags: [calendar, scheduling, events, meetings, mcp]
    related_skills: [maps, meeting-action-items]
---

# Calendar Skill

Manage calendar events, check availability, and schedule meetings through MCP-integrated calendar services. Works with Calendly and other calendar MCP providers.

## MCP Integration

This skill delegates calendar operations to MCP tools:
- **Calendly MCP** — Schedule meetings, check availability, manage event types
- **Other calendar MCPs** — Extend with additional calendar providers as needed

## When to use

Use when the user asks to:
- Check calendar availability
- Schedule or reschedule meetings
- View upcoming events
- Create event invites
- Check meeting conflicts

## Usage

The skill routes calendar requests to available MCP tools. Specify:
- **Action:** list, create, reschedule, cancel, check-availability
- **Details:** dates, times, participants, event title/description
- **Calendar:** which calendar/service to use (if multiple available)

## Example

```
User: "Check my availability tomorrow afternoon"
→ Queries calendar MCP for availability on tomorrow's afternoon slots
→ Returns open time windows

User: "Schedule a 30-min meeting with alice@example.com next Tuesday at 2pm"
→ Creates event via calendar MCP
→ Sends invite to participant
→ Returns confirmation with event details
```

## Limitations

- Requires active MCP calendar service connection
- Availability depends on configured calendar providers
- Some features limited by specific MCP implementation

## Related

- **meeting-action-items** — Extract action items from scheduled meetings
- **maps** — Check location/travel time for meetings
