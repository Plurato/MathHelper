"""Pydantic schemas for MathCoach-Agent data exchange."""

from mathcoach.schemas.inputs import UserQuery
from mathcoach.schemas.plan import SolvingPlan
from mathcoach.schemas.problem import ProblemUnderstanding
from mathcoach.schemas.teaching import TeachingExplanation
from mathcoach.schemas.verification import (
    SolvingVerification,
    VerifiableArtifact,
    VerifiableKind,
    VerificationResult,
)

__all__ = [
    "ProblemUnderstanding",
    "SolvingPlan",
    "SolvingVerification",
    "TeachingExplanation",
    "UserQuery",
    "VerifiableArtifact",
    "VerifiableKind",
    "VerificationResult",
]
