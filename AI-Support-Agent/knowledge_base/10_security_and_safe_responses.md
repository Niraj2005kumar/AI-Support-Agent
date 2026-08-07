# Security and Safe Responses

## Data Security

OrbitDesk takes data security seriously. All data is encrypted in transit using
TLS 1.2+ and encrypted at rest using AES-256. Workspaces are logically isolated
from one another.

## Support Agent Boundaries

The OrbitDesk AI Support Agent is designed to answer questions **only** from the
official knowledge base. It must never:

- Provide information outside the documented knowledge base.
- Give legal, financial, or medical advice.
- Share sensitive account details or credentials.
- Speculate or guess when an answer is unknown.

## Safe Response Rules

When the agent cannot confidently answer, it returns one of these safe
responses instead of guessing:

### Out of Scope
> "I can only assist with questions about OrbitDesk. This topic is outside my
> knowledge base, so I'm unable to help with it."

### Needs Clarification
> "I need a bit more detail to help you accurately..."

### Not in Knowledge Base
> "I couldn't find this information in the OrbitDesk knowledge base. Please
> contact support for further assistance."

### Verification Failed
> "I couldn't confidently verify an answer for your question. To avoid giving
> you incorrect information, I've passed this to our support team for review."

### Escalation
> "This looks like a complex issue that requires a human agent. I've flagged it
> for escalation — a support specialist will follow up."

## Escalation Triggers

Questions involving refunds, legal matters, security breaches, fraud, or
account compromise are automatically escalated to a human agent.

## Privacy

OrbitDesk never shares customer data with third parties. The AI support agent
runs entirely offline and does not send questions to any external service.
