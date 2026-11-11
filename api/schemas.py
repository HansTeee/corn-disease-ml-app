"""
Pydantic models describing what each endpoint returns.

WHY bother: FastAPI uses these to (1) validate our own responses, so a typo in
a key fails loudly instead of shipping a broken response, and (2) generate the
/docs page, so anyone calling the API can see the exact response shape.
"""
from pydantic import BaseModel, Field


class RootResponse(BaseModel):
    service: str
    version: str
    description: str
    endpoints: dict[str, str]


class HealthResponse(BaseModel):
    status: str = Field(description='"healthy" or "unhealthy"')
    model_loaded: bool
    error: str | None = Field(None, description="Why the model failed to load, if it did")


class ModelInfoResponse(BaseModel):
    model_file: str
    model_version: str = Field(description="Training date plus the first 12 chars of the file's SHA-256")
    sha256: str
    classes: list[str] = Field(description="In the same order as the model's output indices")
    input_size: list[int]
    training_date: str | None
    test_accuracy: float | None
    per_class: dict | None
    n_test: int | None


class PredictionResponse(BaseModel):
    predicted_class: str
    confidence: float = Field(description="Probability of the predicted class (0-1)")
    probabilities: dict[str, float] = Field(description="Every class, highest first. Sums to 1.")
    inference_ms: float


class ErrorResponse(BaseModel):
    detail: str
