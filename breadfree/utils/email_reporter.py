"""
SMTP 邮件发送器 — 支持 HTML 正文 + 内嵌图片附件

配置优先级: .env 环境变量 > config.yaml smtp 段
"""

import os
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.image import MIMEImage
from typing import Dict, List, Optional

from .config import get_config
from .logger import get_logger

logger = get_logger(__name__)


def _load_smtp_config() -> Dict:
    """加载 SMTP 配置, 环境变量优先"""
    cfg = get_config().get("smtp", {})
    return {
        "host": os.getenv("SMTP_HOST", cfg.get("host", "smtp.qq.com")),
        "port": int(os.getenv("SMTP_PORT", cfg.get("port", 465))),
        "use_ssl": os.getenv("SMTP_USE_SSL", str(cfg.get("use_ssl", True))).lower() in ("true", "1", "yes"),
        "user": os.getenv("SMTP_USER", cfg.get("user", "")),
        "password": os.getenv("SMTP_PASSWORD", cfg.get("password", "")),
        "recipients": _parse_recipients(cfg),
    }


def _parse_recipients(cfg: Dict) -> List[str]:
    """从环境变量或 config 解析收件人列表"""
    env_val = os.getenv("REPORT_RECIPIENTS", "")
    if env_val:
        return [r.strip() for r in env_val.split(",") if r.strip()]
    return cfg.get("report_recipients", [])


def send_report_email(
    subject: str,
    html_body: str,
    images: Optional[Dict[str, bytes]] = None,
    recipients: Optional[List[str]] = None,
) -> bool:
    """
    发送 HTML 邮件, 可选内嵌图片.

    Args:
        subject: 邮件主题
        html_body: HTML 正文 (图片引用 <img src="cid:{cid_key}">)
        images: {cid_key: png_bytes} 内嵌图片字典
        recipients: 覆盖配置的收件人列表

    Returns:
        True 发送成功, False 失败
    """
    smtp_cfg = _load_smtp_config()
    to_addrs = recipients or smtp_cfg["recipients"]

    if not smtp_cfg["user"] or not smtp_cfg["password"]:
        logger.error("[Email] SMTP 账号或密码未配置, 跳过发送. "
                     "请在 .env 设置 SMTP_USER / SMTP_PASSWORD")
        return False
    if not to_addrs:
        logger.error("[Email] 收件人列表为空, 跳过发送. "
                     "请在 .env 设置 REPORT_RECIPIENTS 或 config.yaml smtp.report_recipients")
        return False

    msg = MIMEMultipart("related")
    msg["Subject"] = subject
    msg["From"] = smtp_cfg["user"]
    msg["To"] = ", ".join(to_addrs)

    msg_alt = MIMEMultipart("alternative")
    msg.attach(msg_alt)
    msg_alt.attach(MIMEText(html_body, "html", "utf-8"))

    if images:
        for cid, img_bytes in images.items():
            img_part = MIMEImage(img_bytes, _subtype="png")
            img_part.add_header("Content-ID", f"<{cid}>")
            img_part.add_header("Content-Disposition", "inline", filename=f"{cid}.png")
            msg.attach(img_part)

    try:
        if smtp_cfg["use_ssl"]:
            server = smtplib.SMTP_SSL(smtp_cfg["host"], smtp_cfg["port"], timeout=30)
        else:
            server = smtplib.SMTP(smtp_cfg["host"], smtp_cfg["port"], timeout=30)
            server.starttls()

        server.login(smtp_cfg["user"], smtp_cfg["password"])
        server.sendmail(smtp_cfg["user"], to_addrs, msg.as_string())
        server.quit()
        logger.info(f"[Email] 邮件已发送: {subject} → {to_addrs}")
        return True
    except Exception as e:
        logger.error(f"[Email] 发送失败: {e}")
        return False
