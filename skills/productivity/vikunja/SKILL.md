---
name: vikunja
description: "Task management and project tracking via Vikunja MCP."
version: 1.0.0
author: Hermes Agent
license: MIT
platforms: [linux, macos, windows]
metadata:
  hermes:
    tags: [vikunja, tasks, projects, todo, mcp]
    related_skills: [obsidian, notion]
---

# Vikunja Skill

Manage tasks, projects, and lists in Vikunja through MCP integration. Create tasks, organize work, track progress, and collaborate on projects.

## MCP Integration

This skill connects to Vikunja via MCP:
- **Vikunja MCP** — Full task/project management operations
- Works with self-hosted or cloud Vikunja instances

## When to use

Use when the user asks to:
- Create or update tasks
- Organize tasks into projects/lists
- Check task status or deadlines
- Assign tasks
- Track project progress
- Add subtasks or dependencies

## Usage

The skill routes Vikunja operations through MCP. Specify:
- **Action:** create, update, list, complete, delete, assign
- **Task details:** title, description, due date, priority, assignee
- **Project/List:** target project or list for the task
- **Relationships:** parent task, subtasks, dependencies

## Example

```
User: "Add 'Review pull request' to my urgent tasks, due tomorrow"
→ Creates task in Vikunja via MCP
→ Sets priority to high
→ Sets due date to tomorrow
→ Returns task ID and confirmation

User: "What tasks am I assigned for this week?"
→ Queries Vikunja MCP for assigned tasks
→ Filters by due date range
→ Returns formatted task list

User: "Move 'Design mockups' to the website redesign project"
→ Relocates task between projects
→ Updates project associations
→ Returns confirmation
```

## Features

- **Task management** — Create, read, update, complete, delete tasks
- **Organization** — Manage projects, lists, buckets
- **Collaboration** — Assign tasks, add comments, track changes
- **Tracking** — Due dates, priorities, progress indicators
- **Relationships** — Subtasks, dependencies, parent tasks

## Limitations

- Requires active Vikunja instance and MCP connection
- Features depend on Vikunja MCP implementation
- Bulk operations may have rate limits

## Related

- **obsidian** — Note-taking alongside task management
- **notion** — Alternative all-in-one workspace
