"""Prompts as versioned config values.

A prompt change is as traceable as a threshold change: the version string is part of
the config, so it is part of the config hash recorded on every evaluation row.
"""

from __future__ import annotations

from ..errors import ConfigError

VERIFIER_TOOL = {
    "name": "submit_verdict",
    "description": "Record the verdict and the passages that justify it.",
    "input_schema": {
        "type": "object",
        "properties": {
            "reasoning": {
                "type": "string",
                "description": "Brief reasoning over the supplied passages only.",
            },
            "verdict": {
                "type": "string",
                "enum": ["supported", "contradicted", "unknown"],
            },
            "evidence": {
                "type": "array",
                "description": (
                    "One entry per passage that bears on the claim. Empty when the "
                    "passages do not address the claim."
                ),
                "items": {
                    "type": "object",
                    "properties": {
                        "doc_id": {
                            "type": "string",
                            "description": "The exact bracketed identifier of the source passage.",
                        },
                        "quote": {
                            "type": "string",
                            "description": (
                                "A verbatim span copied character-for-character from that "
                                "passage. Never paraphrase, never join separated sentences."
                            ),
                        },
                        "stance": {"type": "string", "enum": ["entails", "contradicts", "neither"]},
                    },
                    "required": ["doc_id", "quote", "stance"],
                },
            },
        },
        # Requiring `evidence` is what makes an uncited verdict structurally
        # unrepresentable. The model cannot return a bare answer.
        "required": ["verdict", "evidence"],
    },
}

AUDITOR_TOOL = {
    "name": "submit_stance",
    "description": "Record what the quoted passage does to the claim.",
    "input_schema": {
        "type": "object",
        "properties": {
            "stance": {
                "type": "string",
                "enum": ["entails", "contradicts", "neither"],
                "description": (
                    "entails: the quote asserts the claim. contradicts: the quote asserts "
                    "its opposite. neither: the quote is about something else, or is "
                    "merely related without settling the claim."
                ),
            }
        },
        "required": ["stance"],
    },
}

_VERIFIER_PROMPTS = {
    "v3": (
        "You are a fact-checker with exactly one source of truth: the passages supplied "
        "in this message.\n\n"
        "Rules:\n"
        "1. Your own knowledge of these papers is inadmissible. If the passages do not "
        "settle the claim, the answer is 'unknown' -- even if you are confident the claim "
        "is true.\n"
        "2. Every quote must be copied verbatim from a supplied passage. Quotes are "
        "checked mechanically against the source; an invented or paraphrased quote is "
        "discarded and takes its verdict with it.\n"
        "3. Quote a claim-bearing span -- roughly one clause to one sentence. A few words "
        "prove nothing; an entire passage pinpoints nothing.\n"
        "4. 'contradicted' requires a passage asserting the opposite of the claim. A claim "
        "the passages are simply silent about is 'unknown', never 'contradicted'.\n"
        "5. Attribute each quote to the exact bracketed identifier it came from."
    )
}

_AUDITOR_PROMPTS = {
    "v2": (
        "You are given one claim and one quoted passage, and nothing else.\n\n"
        "Decide what the quote does to the claim:\n"
        "- entails: the quote asserts the claim, or states something that makes it true.\n"
        "- contradicts: the quote asserts the opposite of the claim.\n"
        "- neither: the quote is on a related topic but does not settle the claim.\n\n"
        "Judge only the text in front of you. Topical similarity is not entailment; a "
        "quote about the same subject that does not assert the claim is 'neither'."
    )
}


def verifier_prompt(version: str) -> str:
    return _lookup(_VERIFIER_PROMPTS, version, "verifier")


def auditor_prompt(version: str) -> str:
    return _lookup(_AUDITOR_PROMPTS, version, "auditor")


def _lookup(table: dict[str, str], version: str, kind: str) -> str:
    try:
        return table[version]
    except KeyError:
        raise ConfigError(
            f"unknown {kind} prompt version '{version}'. Known: {', '.join(sorted(table))}"
        ) from None
