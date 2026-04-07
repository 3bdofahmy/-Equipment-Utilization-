"""
motion/llm_verifier.py
───────────────────────
LLM verification layer for activity classification.
Runs only when rule-based confidence is below threshold.

Supported providers: openai | gemini | groq | ollama
"""

import json
from core.config import LLMConfig
from core.logger import get_logger

log = get_logger("llm_verifier")

VALID_ACTIVITIES = ("Digging", "Swinging/Loading", "Dumping", "Traveling", "Waiting")


# ── LLM Factory ───────────────────────────────────────────────────────────────

class LLMFactory:
    """
    Returns a callable client based on config.provider.

    Usage:
        client = LLMFactory.create(config)
        response = client("Your prompt here")
    """

    @staticmethod
    def create(config: LLMConfig):
        provider = config.provider.lower().strip()

        if provider == "openai":
            return OpenAIClient(config)
        if provider == "gemini":
            return GeminiClient(config)
        if provider == "groq":
            return GroqClient(config)
        if provider == "ollama":
            return OllamaClient(config)

        raise ValueError(
            f"Unknown LLM provider '{provider}'. "
            f"Choose from: openai | gemini | groq | ollama"
        )


# ── Clients ───────────────────────────────────────────────────────────────────

class OpenAIClient:
    def __init__(self, config: LLMConfig):
        from openai import OpenAI
        self.client = OpenAI(api_key=config.api_key)
        self.model  = config.model or "gpt-4o-mini"
        self.config = config

    def __call__(self, prompt: str) -> str:
        resp = self.client.chat.completions.create(
            model       = self.model,
            messages    = [{"role": "user", "content": prompt}],
            temperature = self.config.temperature,
            max_tokens  = self.config.max_tokens,
        )
        return resp.choices[0].message.content.strip()


class GeminiClient:
    def __init__(self, config: LLMConfig):
        import google.generativeai as genai
        genai.configure(api_key=config.api_key)
        self.model  = genai.GenerativeModel(config.model or "gemini-1.5-flash")
        self.config = config

    def __call__(self, prompt: str) -> str:
        resp = self.model.generate_content(prompt)
        return resp.text.strip()


class GroqClient:
    def __init__(self, config: LLMConfig):
        from groq import Groq
        self.client = Groq(api_key=config.api_key)
        self.model  = config.model or "llama3-8b-8192"
        self.config = config

    def __call__(self, prompt: str) -> str:
        resp = self.client.chat.completions.create(
            model       = self.model,
            messages    = [{"role": "user", "content": prompt}],
            temperature = self.config.temperature,
            max_tokens  = self.config.max_tokens,
        )
        return resp.choices[0].message.content.strip()


class OllamaClient:
    def __init__(self, config: LLMConfig):
        self.base_url = config.base_url or "http://localhost:11434"
        self.model    = config.model or "llama3"
        self.config   = config

    def __call__(self, prompt: str) -> str:
        import requests
        resp = requests.post(
            f"{self.base_url}/api/generate",
            json    = {"model": self.model, "prompt": prompt, "stream": False},
            timeout = 10,
        )
        resp.raise_for_status()
        return resp.json().get("response", "").strip()


# ── Prompt Builder ────────────────────────────────────────────────────────────

class PromptBuilder:

    @staticmethod
    def build(
        equipment_type:    str,
        track_id:          str,
        rule_prediction:   str,
        rule_confidence:   float,
        motion_score:      float,
        zone_scores:       dict,
        recent_activities: list[str],
        frame_index:       int,
    ) -> str:
        recent = " → ".join(recent_activities[-5:]) if recent_activities else "none"
        zones  = json.dumps(zone_scores, indent=2) if zone_scores else "not available (mask-based)"

        return f"""You are analyzing construction equipment activity from motion sensor data.

Equipment:        {equipment_type} (ID: {track_id})
Frame:            {frame_index}
Motion score:     {motion_score:.3f}  (0=no motion, 1=max motion)
Zone motion:
{zones}

Rule-based prediction: {rule_prediction}
Rule confidence:       {rule_confidence:.2f}

Recent activity history (last 5): {recent}

Based on this motion data, what activity is this {equipment_type} most likely performing?

Valid activities: {", ".join(VALID_ACTIVITIES)}

Respond ONLY with a JSON object in this exact format:
{{"activity": "one of the valid activities", "reason": "one sentence explanation"}}"""


# ── Verifier ──────────────────────────────────────────────────────────────────

class LLMVerifier:
    """
    Verifies rule-based activity classification using an LLM.
    Only called when confidence is below threshold.
    """

    def __init__(self, config: LLMConfig):
        self.config  = config
        self.enabled = config.enabled
        self._client = None

        if self.enabled:
            try:
                self._client = LLMFactory.create(config)
                log.info("LLM verifier ready — provider: %s  model: %s",
                         config.provider, config.model)
            except Exception as e:
                log.warning("LLM verifier init failed: %s — disabled", e)
                self.enabled = False

    def verify(
        self,
        equipment_type:    str,
        track_id:          str,
        rule_prediction:   str,
        rule_confidence:   float,
        motion_score:      float,
        zone_scores:       dict,
        recent_activities: list[str],
        frame_index:       int,
    ) -> tuple[str, bool]:
        """
        Returns (activity, llm_verified).
        If LLM is disabled or fails, returns rule prediction.
        """
        if not self.enabled or rule_confidence >= self.config.__class__.__dataclass_fields__:
            return rule_prediction, False

        try:
            prompt   = PromptBuilder.build(
                equipment_type, track_id, rule_prediction, rule_confidence,
                motion_score, zone_scores, recent_activities, frame_index,
            )
            response = self._client(prompt)
            data     = json.loads(response)
            activity = data.get("activity", rule_prediction)

            if activity not in VALID_ACTIVITIES:
                log.warning("LLM returned invalid activity: %s", activity)
                return rule_prediction, False

            if activity != rule_prediction:
                log.info("LLM corrected %s → %s (conf %.2f) for %s",
                         rule_prediction, activity, rule_confidence, track_id)

            return activity, True

        except Exception as e:
            log.warning("LLM verification failed: %s", e)
            return rule_prediction, False
