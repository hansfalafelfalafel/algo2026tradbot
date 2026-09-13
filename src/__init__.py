"""Пакет src.

Здесь же — слой совместимости с разными версиями SDK Т-Банка. Официальный SDK
переехал с пакета ``tinkoff-investments`` (импорт ``tinkoff.invest``) на
``t-tech-investments`` (импорт ``t_tech.invest``). Весь код проекта написан под
``tinkoff.invest``; этот шим делает так, чтобы он работал и с новым SDK — без
правок в остальных файлах. Если установлен старый SDK — шим ничего не меняет.
"""
import importlib
import importlib.util
import os
import sys


def _configure_grpc_ssl_roots() -> None:
    """Указать gRPC системное хранилище сертификатов.

    gRPC по умолчанию использует собственный набор корневых сертификатов и не
    видит установленные в систему (в т.ч. корневой НУЦ Минцифры, которым подписан
    сертификат сервера Т-Банка). Если переменная не задана, а системный bundle
    существует — направляем gRPC на него.
    """
    if os.environ.get("GRPC_DEFAULT_SSL_ROOTS_FILE_PATH"):
        return
    for path in ("/etc/ssl/certs/ca-certificates.crt",   # Debian/Ubuntu
                 "/etc/pki/tls/certs/ca-bundle.crt"):     # RHEL/CentOS
        if os.path.exists(path):
            os.environ["GRPC_DEFAULT_SSL_ROOTS_FILE_PATH"] = path
            return


def _install_sdk_alias() -> None:
    # Старый SDK установлен -> ничего не делаем, всё работает нативно.
    if importlib.util.find_spec("tinkoff") is not None:
        return
    # Нового SDK нет -> пусть импорт упадёт штатно с понятной ошибкой.
    if importlib.util.find_spec("t_tech") is None:
        return
    try:
        import t_tech
        import t_tech.invest  # noqa: F401
    except Exception:
        return
    # Делаем `tinkoff.invest...` псевдонимом `t_tech.invest...`.
    sys.modules.setdefault("tinkoff", t_tech)
    sys.modules.setdefault("tinkoff.invest", t_tech.invest)
    for sub in (
        "utils", "schemas", "services", "exceptions",
        "sandbox", "sandbox.client", "market_data_stream",
    ):
        try:
            module = importlib.import_module(f"t_tech.invest.{sub}")
            sys.modules[f"tinkoff.invest.{sub}"] = module
        except Exception:
            pass


_configure_grpc_ssl_roots()
_install_sdk_alias()
