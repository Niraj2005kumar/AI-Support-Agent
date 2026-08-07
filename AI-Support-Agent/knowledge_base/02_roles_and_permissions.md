# Roles and Permissions

## Overview

OrbitDesk uses role-based access control (RBAC). Every workspace member is
assigned exactly one role, which determines what they can view and do.

## The Four Roles

### 1. Owner
The workspace Owner has full control, including billing, security settings,
deleting the workspace, and managing all members. There is always exactly one
Owner per workspace.

### 2. Admin
Admins can manage team members, roles, automation rules, and workspace
settings. Admins **can** create workspace API credentials and manage
integrations. Admins cannot delete the workspace or change the Owner.

### 3. Editor
Editors can create and edit tickets, respond to customers, and manage the
knowledge base articles. Editors **cannot** change workspace settings, manage
members, or create API credentials.

### 4. Read-only
Read-only users can view tickets, reports, and documentation but **cannot**
modify anything. Read-only users **cannot** create API credentials, edit
tickets, or change settings.

## Permission Summary Table

| Action                              | Owner | Admin | Editor | Read-only |
|-------------------------------------|:-----:|:-----:|:------:|:---------:|
| View tickets and reports            |  Yes  |  Yes  |  Yes   |    Yes    |
| Create / edit tickets               |  Yes  |  Yes  |  Yes   |    No     |
| Manage members and roles            |  Yes  |  Yes  |   No   |    No     |
| Create workspace API credentials    |  Yes  |  Yes  |   No   |    No     |
| Manage billing / delete workspace   |  Yes  |  No   |   No   |    No     |

## Key Rules

- **Only Owners and Admins can create workspace API credentials.**
- Read-only users cannot create or edit anything.
- Editors cannot manage members or create API credentials.
- The Owner role cannot be transferred to another user without contacting
  support.
