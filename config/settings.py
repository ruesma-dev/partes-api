# config/settings.py
from __future__ import annotations

from typing import Literal

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

ProviderName = Literal["gemini", "claude", "openai"]


class Settings(BaseSettings):
    """Configuracion del servicio 2 de partes (partes-extractor).

    Version simplificada respecto al sv2 de albaranes: una sola fase de
    extraccion con UN proveedor (gemini por defecto, claude opcional).
    El proveedor activo lo elige ``IA_PROVIDER`` y debe estar habilitado.
    """

    # --- Flags de proveedor --- #
    gemini_enabled: bool = Field(True, alias="ENABLE_GEMINI")
    claude_enabled: bool = Field(False, alias="ENABLE_CLAUDE")
    openai_enabled: bool = Field(False, alias="ENABLE_OPENAI")

    # --- Azure Storage (colas/blobs del pipeline). En Azure las inyecta la
    # Container App; en local se leen del .env como el resto de config. --- #
    # Connection strings para LOCAL (Azurite o cuenta con clave). Si estan,
    # MANDAN sobre las account_url + identidad. BLOBS se deriva de COLAS si
    # falta (Azurite: puerto 10001 -> 10000).
    colas_connection_string: str | None = Field(
        None, alias="COLAS_CONNECTION_STRING"
    )
    blobs_connection_string: str | None = Field(
        None, alias="BLOBS_CONNECTION_STRING"
    )

    colas_account_url: str | None = Field(None, alias="COLAS_ACCOUNT_URL")
    blobs_account_url: str | None = Field(None, alias="BLOBS_ACCOUNT_URL")

    gemini_api_key: str | None = Field(None, alias="GEMINI_API_KEY")
    gemini_model: str = Field("gemini-2.5-flash", alias="GEMINI_MODEL")
    # Resolucion de medios (Gemini 3). Valores: default|unspecified|low|medium|high.
    # "default" deja decidir al SDK (NO fuerza tokens extra). "high" mejora la
    # lectura de manuscritos finos pero sube tokens/pagina y consumo de cuota.
    gemini_media_resolution: str = Field("default", alias="GEMINI_MEDIA_RESOLUTION")

    anthropic_api_key: str | None = Field(None, alias="ANTHROPIC_API_KEY")
    anthropic_model: str = Field("claude-sonnet-4-5", alias="ANTHROPIC_MODEL")
    anthropic_max_tokens: int = Field(8192, alias="ANTHROPIC_MAX_TOKENS")
    anthropic_timeout_s: int = Field(120, alias="ANTHROPIC_TIMEOUT_S")

    openai_api_key: str | None = Field(None, alias="OPENAI_API_KEY")
    openai_model: str = Field("gpt-4o", alias="OPENAI_MODEL")
    openai_max_output_tokens: int | None = Field(
        None, alias="OPENAI_MAX_OUTPUT_TOKENS"
    )
    openai_timeout_s: int = Field(120, alias="OPENAI_TIMEOUT_S")

    # --- Proveedor activo y prompt --- #
    ia_provider: ProviderName = Field("gemini", alias="IA_PROVIDER")
    prompt_key: str = Field("parte_trabajo_es", alias="PROMPT_KEY")
    prompts_yaml_path: str = Field("config/prompts.yaml", alias="PROMPTS_YAML_PATH")

    # --- Reintentos LLM --- #
    llm_max_retries: int = Field(2, alias="LLM_MAX_RETRIES")
    llm_backoff_base_s: float = Field(2.0, alias="LLM_BACKOFF_BASE_S")
    llm_backoff_cap_s: float = Field(30.0, alias="LLM_BACKOFF_CAP_S")

    # --- IA call logging (tuning de prompts) --- #
    ia_logging_enabled: bool = Field(False, alias="IA_LOGGING_ENABLED")
    ia_logging_dir: str = Field("logs/ia", alias="IA_LOGGING_DIR")

    # --- Catalogo de codigos de hora (auxhor) desde sigrid-api --- #
    # Si se cablean las tres SIGRID_API_*, sv2 lee el catalogo de codigos de
    # hora, filtra maquinaria/becario y lo inyecta en el prompt para que la IA
    # PROPONGA el codigo por categoria + normal/extra + incidencia.
    sigrid_api_base_url: str | None = Field(None, alias="SIGRID_API_BASE_URL")
    sigrid_api_function_key: str | None = Field(
        None, alias="SIGRID_API_FUNCTION_KEY"
    )
    sigrid_api_database: str | None = Field(None, alias="SIGRID_API_DATABASE")
    sigrid_api_timeout_s: float = Field(30.0, alias="SIGRID_API_TIMEOUT_S")
    auxhor_cache_ttl_s: int = Field(600, alias="AUXHOR_CACHE_TTL_S")
    # Palabras clave (en la descripcion/codigo) a EXCLUIR del catalogo antes
    # de mandarlo a la IA. CSV. Por defecto maquinaria + becario.
    auxhor_exclude_keywords: str = Field(
        "MAQUINARIA,MAQUINA,BECARIO,BECARIA", alias="AUXHOR_EXCLUDE_KEYWORDS"
    )
    auxhor_exclude_codes: str | None = Field(
        None, alias="AUXHOR_EXCLUDE_CODES"
    )

    # --- API + meta --- #
    api_host: str = Field("127.0.0.1", alias="API_HOST")
    api_port: int = Field(8010, alias="API_PORT")
    max_file_mb: int = Field(25, alias="MAX_FILE_MB")
    cors_allow_origins: str | None = Field(None, alias="CORS_ALLOW_ORIGINS")
    log_level: str = Field("INFO", alias="LOG_LEVEL")
    log_dir: str = Field("logs", alias="LOG_DIR")
    service_version: str = Field("1.0.0", alias="SERVICE_VERSION")

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    @property
    def sigrid_catalog_enabled(self) -> bool:
        return bool(
            (self.sigrid_api_base_url or "").strip()
            and (self.sigrid_api_function_key or "").strip()
            and (self.sigrid_api_database or "").strip()
        )

    @property
    def auxhor_exclude_keywords_list(self) -> tuple[str, ...]:
        return tuple(
            k.strip() for k in (self.auxhor_exclude_keywords or "").split(",")
            if k.strip()
        )

    @property
    def auxhor_exclude_codes_list(self) -> tuple[str, ...]:
        return tuple(
            c.strip() for c in (self.auxhor_exclude_codes or "").split(",")
            if c.strip()
        )

    @model_validator(mode="after")
    def _ensure_provider_enabled_and_keyed(self) -> "Settings":
        flag_by_provider = {
            "gemini": self.gemini_enabled,
            "claude": self.claude_enabled,
            "openai": self.openai_enabled,
        }
        if not flag_by_provider.get(self.ia_provider, False):
            raise ValueError(
                f"IA_PROVIDER='{self.ia_provider}' pero "
                f"ENABLE_{self.ia_provider.upper()} no esta a true."
            )
        if self.ia_provider == "gemini" and not (self.gemini_api_key or "").strip():
            raise ValueError("IA_PROVIDER=gemini requiere GEMINI_API_KEY.")
        if self.ia_provider == "claude" and not (self.anthropic_api_key or "").strip():
            raise ValueError("IA_PROVIDER=claude requiere ANTHROPIC_API_KEY.")
        if self.ia_provider == "openai" and not (self.openai_api_key or "").strip():
            raise ValueError("IA_PROVIDER=openai requiere OPENAI_API_KEY.")
        return self
