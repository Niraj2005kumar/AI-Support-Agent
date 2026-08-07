# API Credentials

## Overview

OrbitDesk provides a REST API for programmatically managing tickets, workspace
settings, and reporting. To authenticate with the API, you need a set of
workspace API credentials: an **API key** and an **API secret**.

## Who Can Create API Credentials?

**Only users with the Owner or Admin role can create workspace API
credentials.**

- Editors **cannot** create API credentials.
- Read-only users **cannot** create API credentials.

## Creating API Credentials

1. Go to **Settings → API**.
2. Click **Create credential**.
3. Give the credential a name (e.g., "Production server").
4. OrbitDesk generates an API key and API secret.

> **Important**: The API secret is shown only once. If you lose it, you must
> regenerate the credential.

## Regenerating Credentials

To regenerate a credential:

1. Go to **Settings → API**.
2. Find the credential you wish to rotate.
3. Click **Regenerate**.
4. Confirm the action. The old secret is invalidated immediately.

## Revoking Credentials

To revoke a credential, click **Revoke** on the credential. This immediately
invalidates the key and secret, and any requests using them will fail.

## Security Best Practices

- Store the API secret in a secure vault; never commit it to source control.
- Rotate credentials regularly.
- Use workspace-scoped credentials rather than sharing personal ones.
- Revoke unused credentials promptly.
