from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    github_app_id: str
    github_app_private_key_path: str = "./keys/github-app.pem"
    github_webhook_secret: str
    anthropic_api_key: str
    target_branch: str = "develop"

    model_config = {"env_file": ".env"}


settings = Settings()
