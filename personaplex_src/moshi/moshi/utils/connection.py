# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

import socket
import subprocess
import platform
from shutil import which
import urllib.request
from pathlib import Path
from .logging import setup_logger

logger = setup_logger(__name__)

def get_lan_ip():
    """
    get_lan_ip()
    
    Get the LAN IP address of the current machine.

    Returns
    -------
    str
        LAN IP address as a string.
    """
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))
        return s.getsockname()[0]
    finally:
        s.close()


def ensure_mkcert():
    """
    Ensure mkcert exists; if not, download the correct binary.
    
    Returns
    -------
    str | None
        Path to mkcert binary, or None if installation failed.
    """

    # mkcert already installed?
    mkcert_bin_path = which("mkcert")
    if mkcert_bin_path:
        return mkcert_bin_path

    logger.info("[auto-cert] mkcert not found, attempting to download...")

    # Determine OS + arch
    system = platform.system().lower()   # "linux", "darwin", "windows"
    machine = platform.machine().lower() # "x86_64", "amd64", "arm64", etc.

    # Normalize architecture
    if machine in ("x86_64", "amd64"):
        arch = "amd64"
    elif machine in ("aarch64", "arm64"):
        arch = "arm64"
    elif machine.startswith("arm"):
        arch = "arm"
    else:
        logger.error(f"[auto-cert] Unsupported architecture: {machine}")
        return None

    # Determine filename and URL
    if system == "linux":
        filename = f"mkcert-v1.4.4-linux-{arch}"
    elif system == "darwin":
        filename = f"mkcert-v1.4.4-darwin-{arch}"
    elif system == "windows":
        filename = f"mkcert-v1.4.4-windows-{arch}.exe"
    else:
        logger.error(f"[auto-cert] Unsupported OS: {system}")
        return None

    url = f"https://github.com/FiloSottile/mkcert/releases/download/v1.4.4/{filename}"
    dest_dir = Path.home() / ".local" / "bin"
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / ("mkcert.exe" if system == "windows" else "mkcert")

    logger.info(f"[auto-cert] Downloading: {url}")
    try:
        urllib.request.urlretrieve(url, dest)
    except Exception as e:
        logger.error(f"[auto-cert] Failed to download mkcert: {e}")
        return None

    # Make executable if Unix
    if system != "windows":
        dest.chmod(0o755)

    logger.info(f"[auto-cert] mkcert installed at {dest}")

    # Verify it works
    try:
        subprocess.check_call([str(dest), "-help"],
                              stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception:
        logger.error("[auto-cert] mkcert failed to run after install.")
        return None

    return str(dest)


def _run_command(cmd):
    """Run command, return True on success."""
    try:
        subprocess.check_call(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return True
    except subprocess.CalledProcessError:
        return False


def ensure_mkcert_ca(mkcert_bin: str):
    """
    Install mkcert CA if not installed yet.

    Parameters
    ----------
    mkcert_bin : str
        Path to mkcert binary.
    """
    _run_command([mkcert_bin, "-install"])


def _create_self_signed_cert(cert_file: Path, key_file: Path):
    """Fallback: generate a self-signed certificate using Python's ssl/cryptography libs."""
    try:
        from cryptography import x509
        from cryptography.x509.oid import NameOID
        from cryptography.hazmat.primitives import hashes, serialization
        from cryptography.hazmat.primitives.asymmetric import rsa
        import datetime
        import ipaddress

        lan_ip = get_lan_ip()
        logger.info(f"[auto-cert] Generating self-signed certificate for localhost and {lan_ip}...")

        key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
        subject = issuer = x509.Name([
            x509.NameAttribute(NameOID.COMMON_NAME, "PersonaPlex Self-Signed"),
        ])
        san_list = [
            x509.DNSName("localhost"),
            x509.IPAddress(ipaddress.IPv4Address("127.0.0.1")),
            x509.IPAddress(ipaddress.IPv6Address("::1")),
        ]
        try:
            san_list.append(x509.IPAddress(ipaddress.ip_address(lan_ip)))
        except ValueError:
            san_list.append(x509.DNSName(lan_ip))

        cert = (
            x509.CertificateBuilder()
            .subject_name(subject)
            .issuer_name(issuer)
            .public_key(key.public_key())
            .serial_number(x509.random_serial_number())
            .not_valid_before(datetime.datetime.now(datetime.timezone.utc))
            .not_valid_after(datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(days=365))
            .add_extension(x509.SubjectAlternativeName(san_list), critical=False)
            .sign(key, hashes.SHA256())
        )

        key_file.write_bytes(key.private_bytes(
            serialization.Encoding.PEM,
            serialization.PrivateFormat.TraditionalOpenSSL,
            serialization.NoEncryption(),
        ))
        cert_file.write_bytes(cert.public_bytes(serialization.Encoding.PEM))
        logger.info("[auto-cert] Self-signed certificate generated.")
        return cert_file, key_file
    except ImportError:
        logger.warning("[auto-cert] 'cryptography' package not installed, cannot generate self-signed cert.")
        return None, None
    except Exception as e:
        logger.warning(f"[auto-cert] Self-signed cert generation failed: {e}")
        return None, None


def create_cert_if_needed(cert_dir: str):
    """
    Create cert.pem and key.pem using mkcert if they don't already exist.
    Falls back to a Python-generated self-signed certificate if mkcert is unavailable.
    Returns (cert_file, key_file) or (None, None) if all methods fail.

    Parameters
    ----------
    cert_dir : str
        Directory to store or find cert.pem and key.pem.
    
    Returns
    -------
    pathlib.Path | None
        Path to cert.pem, or None if not created.
    pathlib.Path | None
        Path to key.pem, or None if not created.
    """
    cert_dir = Path(cert_dir)
    cert_dir.mkdir(parents=True, exist_ok=True)

    cert_file = cert_dir / "cert.pem"
    key_file = cert_dir / "key.pem"

    # Already exists → nothing to do
    if cert_file.exists() and key_file.exists():
        return cert_file, key_file

    mkcert_bin = ensure_mkcert()

    if mkcert_bin:
        logger.info("[auto-cert] mkcert detected. Ensuring local CA installed...")
        ensure_mkcert_ca(mkcert_bin)

        # Create cert for localhost, loopback, and LAN IP
        lan_ip = get_lan_ip()
        logger.info(f"[auto-cert] Generating certificate for localhost and {lan_ip}...")

        success = _run_command([
            mkcert_bin,
            "-cert-file", str(cert_file),
            "-key-file", str(key_file),
            "localhost",
            "127.0.0.1",
            "::1",
            lan_ip
        ])

        if success:
            logger.info("[auto-cert] Certificate generated via mkcert.")
            return cert_file, key_file
        else:
            logger.warning("[auto-cert] mkcert cert generation failed, trying self-signed fallback...")
    else:
        logger.warning("[auto-cert] mkcert not available, trying self-signed fallback...")

    return _create_self_signed_cert(cert_file, key_file)


def create_ssl_context(cert_dir: str):
    """
    Main entry: create SSL context if certificates can be created.
    Returns (ssl_context, protocol_str) where protocol_str is "http" or "https".

    Parameters
    ----------
    cert_dir : str
        Directory to store or find cert.pem and key.pem.

    Returns
    -------
    ssl.SSLContext | None
        SSL context if HTTPS is available, else None.
    str
        "https" if SSL context created, else "http".
    """
    cert_file, key_file = create_cert_if_needed(cert_dir)

    if cert_file is None:
        # mkcert missing → use HTTP
        return None, "http"

    import ssl
    ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    ctx.load_cert_chain(certfile=str(cert_file), keyfile=str(key_file))
    return ctx, "https"
