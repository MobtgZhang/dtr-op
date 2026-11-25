"""Common agent scaffolding."""
from typing import Dict, List, Optional

from ..case_library import CaseLibrary
from ..llm_client import LLMClient
from ..prompts import parse_numbered


class BaseAgent:
    name = "base"

    def __init__(self, llm: LLMClient, dataset: str):
        assert dataset in ("alfworld", "scienceworld", "webshop")
        self.llm = llm
        self.dataset = dataset
        self.library: Optional[CaseLibrary] = None

    def bootstrap(self, library: CaseLibrary) -> None:
        self.library = library

    def query(self, task: str) -> Dict:
        raise NotImplementedError

    # shared helpers -----------------------------------------------------
    def _gen(self, prompt: str, max_tokens: int = 1024) -> str:
        return self.llm.chat([{"role": "user", "content": prompt}], max_tokens=max_tokens)

    @staticmethod
    def _parse(text: str) -> List[str]:
        return parse_numbered(text)
