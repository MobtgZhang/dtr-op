"""Helpers for optional live experiments."""

from src.case_library import CaseLibrary
from src.evaluation.metrics import evaluate_alf_offline, evaluate_sw_offline
from src.llm_client import LLMClient


def make_llm(model=None):
    return LLMClient(model=model)


def build_library(dataset, train_items):
    library = CaseLibrary(dataset)
    library.build_from_episodes(train_items)
    print(f"[library] {library.stats()}", flush=True)
    return library


def offline_evaluator(dataset):
    return {"alfworld": evaluate_alf_offline, "scienceworld": evaluate_sw_offline}[dataset]
