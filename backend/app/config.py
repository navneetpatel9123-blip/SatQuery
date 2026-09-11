"""
Configuration settings for SatQuery AI backend.
"""
from pathlib import Path


class Settings:
    """Application settings."""

    # Application
    app_name: str = "SatQuery AI"
    app_version: str = "0.1.0"
    debug: bool = True

    # Server
    host: str = "0.0.0.0"
    port: int = 8000

    # Storage paths
    base_dir: Path = Path(__file__).parent.parent
    upload_dir: Path = base_dir / "data" / "uploads"
    results_dir: Path = base_dir / "data" / "results"
    thumbnails_dir: Path = base_dir / "data" / "thumbnails"

    # Analysis settings
    max_file_size_mb: int = 500
    thumbnail_size: tuple[int, int] = (512, 512)
    change_threshold: float = 0.15

    # CORS
    frontend_url: str = "http://localhost:3000"

    def ensure_dirs(self):
        """Create required directories if they don't exist."""
        for d in [self.upload_dir, self.results_dir, self.thumbnails_dir]:
            d.mkdir(parents=True, exist_ok=True)


settings = Settings()
settings.ensure_dirs()
