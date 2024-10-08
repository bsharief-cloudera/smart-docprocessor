from typing import Optional

from pydantic_settings import BaseSettings, SettingsConfigDict


class FlaskSettings(BaseSettings):
    host: str = "127.0.0.1"
    port: int = 8100


class LLMModelSettings(BaseSettings):
    model_name: str = (
        "/home/cdsw/.cache/huggingface/hub/models--TheBloke--Karen_TheEditor_V2_STRICT_Mistral_7B-GGUF/snapshots/6c654aa207a4b673379db7a928b87a01d644676c/karen_theeditor_v2_strict_mistral_7b.Q8_0.gguf"
    )


class Settings(BaseSettings):
    flask: FlaskSettings = FlaskSettings()
    llm_model: LLMModelSettings = LLMModelSettings()


settings = Settings()
