from __future__ import annotations

import sib_api_v3_sdk
from sib_api_v3_sdk.rest import ApiException

from ..config import settings


def send_via_brevo(
    api_instance: sib_api_v3_sdk.TransactionalEmailsApi,
    receiver_email: str,
    subject: str,
    body: str,
) -> tuple[bool, str | None]:
    send_smtp_email = sib_api_v3_sdk.SendSmtpEmail(
        to=[{"email": receiver_email}],
        sender={"email": settings.sender_email, "name": "Enterprise AI Support"},
        subject=subject,
        text_content=body,
    )
    try:
        api_instance.send_transac_email(send_smtp_email)
        return True, None
    except ApiException as exc:
        return False, str(exc)
