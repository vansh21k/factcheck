"""Failures that must be loud.

Every error here covers a case whose natural failure mode is a plausible-looking
result rather than a crash: a silently short corpus, a stale index, a mismatched
embedding model, a swept index-time knob. None is detectable downstream by reading
verdicts, so each gets an assertion at the boundary.
"""

from __future__ import annotations


class FactCheckError(Exception):
    """Base class for every error this system raises deliberately."""


class EmptyCorpusError(FactCheckError):
    """Ingest resolved zero documents.

    Left unchecked this reads as appropriate conservatism — every claim returns
    unknown — rather than as a broken corpus.
    """


class MissingIndexError(FactCheckError):
    """No manifest at the given directory. Never an implicit rebuild."""


class StaleIndexError(FactCheckError):
    """The index was built from different documents than the ones on disk."""


class EmbeddingModelMismatchError(FactCheckError):
    """Configured embedding model differs from the one recorded in the manifest.

    A dimension change errors loudly on its own; a different model at the same
    dimension returns confident nonsense in plausible rank order, so identity is
    checked explicitly and refusal is at startup.
    """


class IndexTimeConfigError(FactCheckError):
    """An index-time setting was varied against a prebuilt index."""


class ConfigError(FactCheckError):
    """A config file or override could not be applied."""


class MissingAPIKeyError(FactCheckError):
    """No Anthropic API key available; reported at startup, not mid-claim."""


class ModelCallError(FactCheckError):
    """A model call failed -- rate limit, billing, network, or a service error.

    Translated at the client boundary so nothing above it imports the SDK's exception
    types, and so an interactive session reports the failure and stays alive instead
    of dying with a stack trace mid-claim.
    """
