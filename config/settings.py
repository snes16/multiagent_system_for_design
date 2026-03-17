from __future__ import annotations
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # API keys
    anthropic_api_key: str = ""
    firecrawl_api_key: str = ""

    # Model selection per agent (can be overridden via .env)
    research_model: str = "claude-sonnet-4-5"
    research_temperature: float = 0.3

    analyst_model: str = "claude-opus-4-5"
    analyst_temperature: float = 0.8

    generator_model: str = "claude-opus-4-5"
    generator_temperature: float = 0.9

    critic_model: str = "claude-sonnet-4-5"
    critic_temperature: float = 0.2

    orchestrator_model: str = "claude-opus-4-5"
    orchestrator_temperature: float = 0.2

    # Pipeline settings
    min_quality_score: float = 7.0
    output_dir: str = "./output"
    max_research_iter: int = 6
    max_generation_iter: int = 3
    max_critic_iter: int = 2
    max_orchestrator_iter: int = 12

    # Retry settings
    llm_max_retries: int = 3
    llm_retry_wait_seconds: int = 5


settings = Settings()
