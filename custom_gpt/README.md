# Custom GPT Module (Phase 2: Auth Context + Role Headers)

This module implements enterprise-ready frontend auth context handling for the Custom GPT tab.

## Included
- Session auth context model (`token`, `user_id`, `tenant_id`, `role`, `scopes`)
- JWT apply/decode helper
- Role/Tenant/API key header builder
- UI card for current auth context
- Clear auth/logout action
- Header preview + backend ping using current auth headers

## Files
- `config.py`: env-backed settings
- `types.py`: pydantic models
- `auth_context.py`: session + JWT utilities
- `client.py`: request header builder
- `ui_custom_gpt.py`: streamlit tab renderer

## Notes
- This phase focuses on auth context and request-header readiness.
- RBAC/ACL policy enforcement remains backend responsibility.
