"""
LLM Configuration Examples

Shows how Pydantic Settings handles complex LLM configurations.
Compare with OmegaConf to see which fits better.
"""

from typing import Literal, Optional, Dict, Any
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


# ============================================================================
# PYDANTIC SETTINGS APPROACH (Current)
# ============================================================================

class OpenAIConfig(BaseSettings):
    """OpenAI API configuration."""

    api_key: str = Field(default="", description="OpenAI API key")
    model: str = Field(default="gpt-4", description="Model name")
    temperature: float = Field(default=0.7, ge=0.0, le=2.0, description="Sampling temperature")
    max_tokens: int = Field(default=2000, ge=1, le=32000, description="Max tokens to generate")
    top_p: float = Field(default=1.0, ge=0.0, le=1.0, description="Nucleus sampling")
    frequency_penalty: float = Field(default=0.0, ge=-2.0, le=2.0)
    presence_penalty: float = Field(default=0.0, ge=-2.0, le=2.0)

    model_config = SettingsConfigDict(
        env_prefix="OPENAI_",
        env_file=".env",
        extra="ignore"
    )


class AnthropicConfig(BaseSettings):
    """Anthropic Claude API configuration."""

    api_key: str = Field(default="", description="Anthropic API key")
    model: Literal["claude-3-5-sonnet-20241022", "claude-3-5-haiku-20241022", "claude-3-opus-20240229"] = Field(
        default="claude-3-5-haiku-20241022",
        description="Claude model"
    )
    temperature: float = Field(default=1.0, ge=0.0, le=1.0)
    max_tokens: int = Field(default=2000, ge=1, le=8000)

    model_config = SettingsConfigDict(
        env_prefix="ANTHROPIC_",
        env_file=".env",
        extra="ignore"
    )


class EmbeddingConfig(BaseSettings):
    """Embedding model configuration."""

    model_name: str = Field(
        default="all-mpnet-base-v2",
        description="Sentence transformer model name"
    )
    device: Literal["cpu", "cuda", "mps"] = Field(default="cpu")
    batch_size: int = Field(default=32, ge=1, le=512)
    normalize: bool = Field(default=True, description="Normalize embeddings")

    # Cache settings
    cache_dir: Optional[str] = Field(default=None, description="Model cache directory")

    model_config = SettingsConfigDict(
        env_prefix="EMBEDDING_",
        env_file=".env",
        extra="ignore"
    )


class SkillExtractionConfig(BaseSettings):
    """Configuration for LLM-based skill extraction."""

    # Which LLM to use
    provider: Literal["openai", "anthropic", "local"] = Field(
        default="anthropic",
        description="LLM provider for skill extraction"
    )

    # Extraction settings
    max_retries: int = Field(default=3, ge=1, le=10)
    timeout: int = Field(default=30, ge=5, le=120, description="API timeout in seconds")

    # Prompt settings
    include_context: bool = Field(default=True, description="Include context in extraction")
    extract_years: bool = Field(default=True, description="Extract years of experience")
    extract_level: bool = Field(default=True, description="Extract proficiency level")

    model_config = SettingsConfigDict(
        env_prefix="SKILL_EXTRACTION_",
        env_file=".env",
        extra="ignore"
    )


class MatchingConfig(BaseSettings):
    """Job matching algorithm configuration."""

    # Embedding weights for multi-aspect matching
    weight_full_match: float = Field(default=0.20, ge=0.0, le=1.0)
    weight_skill_match: float = Field(default=0.35, ge=0.0, le=1.0)
    weight_required_skill_match: float = Field(default=0.30, ge=0.0, le=1.0)
    weight_semantic_match: float = Field(default=0.15, ge=0.0, le=1.0)

    # Matching thresholds
    min_match_score: float = Field(default=0.5, ge=0.0, le=1.0, description="Minimum score to show")
    top_k_results: int = Field(default=10, ge=1, le=100, description="Number of results to return")

    @field_validator("weight_full_match", "weight_skill_match", "weight_required_skill_match", "weight_semantic_match")
    @classmethod
    def validate_weights_sum(cls, v, info):
        """Validate that all weights sum to 1.0 (checked after all fields loaded)."""
        # Note: This is a simplified validator. For proper validation of sum,
        # you'd use model_validator with mode='after'
        return v

    model_config = SettingsConfigDict(
        env_prefix="MATCHING_",
        env_file=".env",
        extra="ignore"
    )


class LLMSettings(BaseSettings):
    """Complete LLM configuration."""

    openai: OpenAIConfig = Field(default_factory=OpenAIConfig)
    anthropic: AnthropicConfig = Field(default_factory=AnthropicConfig)
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    skill_extraction: SkillExtractionConfig = Field(default_factory=SkillExtractionConfig)
    matching: MatchingConfig = Field(default_factory=MatchingConfig)

    model_config = SettingsConfigDict(
        env_file=".env",
        extra="ignore"
    )


# ============================================================================
# USAGE EXAMPLES
# ============================================================================

def example_pydantic_usage():
    """Example of using Pydantic Settings for LLM config."""

    # Load config
    config = LLMSettings()

    # Type-safe access with full IDE autocomplete
    print(f"Using {config.skill_extraction.provider} for skill extraction")
    print(f"Model: {config.anthropic.model}")
    print(f"Temperature: {config.anthropic.temperature}")

    # Validation happens automatically
    # config.anthropic.temperature = 5.0  # ✗ ValidationError: must be <= 1.0

    # Easy to pass to LLM clients
    import anthropic
    client = anthropic.Anthropic(api_key=config.anthropic.api_key)

    response = client.messages.create(
        model=config.anthropic.model,
        max_tokens=config.anthropic.max_tokens,
        temperature=config.anthropic.temperature,
        messages=[{"role": "user", "content": "Extract skills"}]
    )


# ============================================================================
# OMEGACONF COMPARISON (Alternative)
# ============================================================================

"""
OmegaConf approach would look like:

```yaml
# config/llm.yaml
openai:
  api_key: ${oc.env:OPENAI_API_KEY}
  model: gpt-4
  temperature: 0.7
  max_tokens: 2000

anthropic:
  api_key: ${oc.env:ANTHROPIC_API_KEY}
  model: claude-3-5-haiku-20241022
  temperature: 1.0
  max_tokens: 2000

embedding:
  model_name: all-mpnet-base-v2
  device: cpu
  batch_size: 32

skill_extraction:
  provider: anthropic
  max_retries: 3
  timeout: 30

matching:
  weights:
    full_match: 0.20
    skill_match: 0.35
    required_skill_match: 0.30
    semantic_match: 0.15
  min_match_score: 0.5
  top_k_results: 10
```

```python
from omegaconf import OmegaConf

# Load config
config = OmegaConf.load("config/llm.yaml")

# Access values (but no type safety!)
print(config.anthropic.model)  # IDE doesn't know this exists
print(config.anthropic.temperature)  # Could be string, float, who knows?

# No validation - typos fail at runtime
print(config.anthropic.modle)  # ✗ No error until you access it!

# Have to manually validate
assert 0.0 <= config.anthropic.temperature <= 1.0
```
"""


# ============================================================================
# DECISION GUIDE
# ============================================================================

"""
## When to Use Pydantic Settings (RECOMMENDED for Jobfit):

✅ LLM API configurations (keys, models, params)
✅ Embedding model settings
✅ Matching algorithm weights/thresholds
✅ Simple hyperparameters
✅ Production deployments
✅ When you want type safety and validation

**Pros:**
- Type-safe (IDE autocomplete, type checking)
- Automatic validation (ranges, types, required fields)
- .env file support (12-factor app)
- Simple, Pythonic
- No extra config files needed
- Perfect for API configs

**Cons:**
- Less flexible for complex hierarchies
- Can't easily override nested values from CLI
- More boilerplate for very complex configs


## When to Use OmegaConf:

✅ Complex ML experiment configs with many variants
✅ Need config composition/merging
✅ Hydra integration (multi-run experiments)
✅ Research/experimentation phase
✅ Many YAML config files to manage

**Pros:**
- Powerful config composition
- Easy CLI overrides (hydra)
- Good for ML experiments
- YAML is human-readable

**Cons:**
- No type safety
- No IDE autocomplete
- No automatic validation
- Extra dependency (OmegaConf + Hydra)
- Steeper learning curve
- Need separate YAML files


## Recommendation for Jobfit:

**Use Pydantic Settings!**

Reasons:
1. Our LLM configs are relatively simple
2. We prioritize type safety and validation
3. We use .env files (12-factor app best practice)
4. We're building a production app, not running experiments
5. Better developer experience (IDE support)

If we later need complex experiment configs, we can:
- Add OmegaConf for experiments only
- Keep Pydantic Settings for production config
- Both can coexist!
"""


# ============================================================================
# EXAMPLE .ENV FILE FOR LLM CONFIG
# ============================================================================

"""
# .env file example

# OpenAI Configuration
OPENAI_API_KEY=sk-...
OPENAI_MODEL=gpt-4
OPENAI_TEMPERATURE=0.7
OPENAI_MAX_TOKENS=2000

# Anthropic Configuration
ANTHROPIC_API_KEY=sk-ant-...
ANTHROPIC_MODEL=claude-3-5-haiku-20241022
ANTHROPIC_TEMPERATURE=1.0
ANTHROPIC_MAX_TOKENS=2000

# Embedding Configuration
EMBEDDING_MODEL_NAME=all-mpnet-base-v2
EMBEDDING_DEVICE=cpu
EMBEDDING_BATCH_SIZE=32
EMBEDDING_NORMALIZE=true

# Skill Extraction
SKILL_EXTRACTION_PROVIDER=anthropic
SKILL_EXTRACTION_MAX_RETRIES=3
SKILL_EXTRACTION_TIMEOUT=30

# Matching Weights
MATCHING_WEIGHT_FULL_MATCH=0.20
MATCHING_WEIGHT_SKILL_MATCH=0.35
MATCHING_WEIGHT_REQUIRED_SKILL_MATCH=0.30
MATCHING_WEIGHT_SEMANTIC_MATCH=0.15
MATCHING_MIN_MATCH_SCORE=0.5
MATCHING_TOP_K_RESULTS=10
"""
