"""
LLM Interface for GreenRouting.
Connects to local Ollama instance and provides a robust, context-grounded simulation fallback
when Ollama is offline or models are unavailable.
"""
import logging
import requests
import json
import time
from typing import Dict, Any, List, Optional

logger = logging.getLogger(__name__)


class OllamaClient:
    """Client for querying Ollama models with automatic health check and fallback."""

    def __init__(self, base_url: str = "http://localhost:11434", enable_simulation_fallback: bool = True):
        self.base_url = base_url.rstrip("/")
        self.enable_simulation_fallback = enable_simulation_fallback
        self.is_connected = False
        self.available_models: List[str] = []
        self.check_connection()

    def check_connection(self) -> bool:
        """Probe Ollama endpoint to verify connectivity and list available models."""
        try:
            resp = requests.get(f"{self.base_url}/api/tags", timeout=2.0)
            if resp.status_code == 200:
                data = resp.json()
                self.available_models = [m.get("name", "") for m in data.get("models", [])]
                self.is_connected = True
                logger.info("Connected to Ollama! Available models: %s", self.available_models)
                return True
        except Exception:
            self.is_connected = False
            self.available_models = []

        return False

    def is_model_available(self, model_name: str) -> bool:
        """Check if a specific model is pulled in Ollama."""
        if not self.is_connected:
            self.check_connection()
        # Also check without tag suffix if omitted
        base_name = model_name.split(":")[0]
        return any(model_name == m or base_name == m.split(":")[0] for m in self.available_models)

    def generate(self, prompt: str, model_name: str, system_prompt: Optional[str] = None,
                 temperature: float = 0.2) -> Dict[str, Any]:
        """
        Send a generation request to Ollama or fall back to simulated response.
        Returns a dictionary containing answer_text, model_used, is_simulated, and latency.
        """
        start_time = time.perf_counter()

        # If connected and model is available, use real Ollama
        if self.is_connected and (self.is_model_available(model_name) or len(self.available_models) > 0):
            try:
                payload = {
                    "model": model_name,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": temperature
                    }
                }
                if system_prompt:
                    payload["system"] = system_prompt

                resp = requests.post(f"{self.base_url}/api/generate", json=payload, timeout=60.0)
                if resp.status_code == 200:
                    data = resp.json()
                    elapsed = time.perf_counter() - start_time
                    return {
                        "answer": data.get("response", "").strip(),
                        "model_used": model_name,
                        "is_simulated": False,
                        "latency_seconds": elapsed,
                        "eval_count": data.get("eval_count", 0),
                        "prompt_eval_count": data.get("prompt_eval_count", 0)
                    }
                else:
                    logger.warning("Ollama returned status %d: %s. Using fallback.", resp.status_code, resp.text)
            except Exception as e:
                logger.warning("Ollama query failed: %s. Using fallback.", e)

        # Fallback to simulation if enabled
        if self.enable_simulation_fallback:
            return self._simulated_generate(prompt, model_name, start_time)

        raise ConnectionError(f"Ollama is unreachable at {self.base_url} and simulation fallback is disabled.")

    def _simulated_generate(self, prompt: str, model_name: str, start_time: float) -> Dict[str, Any]:
        """
        Deterministic, context-grounded response generator for offline execution,
        testing, and environments without local LLM daemons.
        Synthesizes real information from the retrieved context embedded in prompt.
        """
        # Brief pause to simulate model compute latency
        is_lightweight = any(k in model_name.lower() for k in ["0.5b", "1b", "1.5b", "tiny", "mini"])
        simulated_delay = 0.08 if is_lightweight else 0.18
        time.sleep(simulated_delay)

        elapsed = time.perf_counter() - start_time

        # Extract Context and Question from prompt if formatted standardly
        context_part = ""
        question_part = prompt
        if "Context:" in prompt and "Question:" in prompt:
            parts = prompt.split("Question:")
            question_part = parts[1].strip() if len(parts) > 1 else ""
            context_part = parts[0].replace("Context:", "").strip()

        # If context is empty
        if not context_part:
            return {
                "answer": "I do not have sufficient context from the uploaded documents to answer this question accurately.",
                "model_used": f"{model_name} (Simulated Offline)",
                "is_simulated": True,
                "latency_seconds": elapsed,
                "eval_count": 22,
                "prompt_eval_count": 35
            }

        # Extract sentences from context that share keywords with the question
        q_words = set(w.lower() for w in question_part.replace("?", "").replace(".", "").split()
                      if len(w) > 3 and w.lower() not in {"what", "which", "where", "when", "how", "does", "explain", "describe"})
        
        sentences = [s.strip() for s in context_part.replace("\n", " ").split(".") if len(s.strip()) > 10]
        matching_sentences = []
        for s in sentences:
            s_lower = s.lower()
            overlap = sum(1 for w in q_words if w in s_lower)
            if overlap > 0:
                matching_sentences.append((overlap, s))

        matching_sentences.sort(key=lambda x: x[0], reverse=True)
        top_sentences = [s[1] for s in matching_sentences[:3]]

        if not top_sentences:
            # Fallback to the first few sentences of the retrieved chunk
            top_sentences = sentences[:2] if sentences else [context_part[:150]]

        grounded_core = ". ".join(top_sentences) + "."

        if is_lightweight:
            answer = f"Based on the provided document: {grounded_core}"
        else:
            answer = (
                f"According to the source documentation:\n\n"
                f"{grounded_core}\n\n"
                f"In summary, this directly addresses the inquiry regarding '{question_part[:60]}...' "
                f"using the verified document context."
            )

        return {
            "answer": answer,
            "model_used": f"{model_name} (Simulated Offline)",
            "is_simulated": True,
            "latency_seconds": elapsed,
            "eval_count": len(answer.split()) * 2,
            "prompt_eval_count": len(prompt.split()) * 2
        }
