from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    github_app_id: str
    github_app_private_key_path: str = "./keys/github-app.pem"
    github_webhook_secret: str
    anthropic_api_key: str = ""
    target_branch: str = "develop"
    # "api" uses the Anthropic SDK (requires credits); "cli" uses the claude CLI subprocess
    review_backend: str = "cli"

    model_config = {"env_file": ".env"}


settings = Settings()
