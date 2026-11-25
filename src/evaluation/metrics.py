"""Offline evaluation metrics (using the SAME lenient matching as old code).

CRITICAL: The old code used entity-overlap matching with SR threshold=0.3,
NOT strict exact match. This is why it achieved 98% SR.

ALFWorld matching:
  - Strip coordinates from entity names (_bar__xxx patterns)
  - Extract action type (verb) and entities (nouns)
  - Match if: (1) same action type, (2) entity overlap > 0
  - Score = entity_overlap / total_gold_entities
  - SR = 1.0 if average_score >= 0.3 (very lenient!)

ScienceWorld matching:
  - Extract (verb, object) pairs
  - Match based on verb and key object words
"""
import re
from typing import Dict, List

# Action types
ACTION_TYPES = {
    "go", "take", "move", "open", "close", "use", "heat", "cool", "clean",
    "examine", "slice", "put", "place", "pick", "inventory", "look",
    "activate", "deactivate", "focus", "wait", "pour", "mix", "connect",
}

STOPWORDS = {
    "the", "a", "an", "to", "and", "in", "on", "it", "is", "for", "of",
    "your", "task", "you", "with", "some", "then", "them", "into", "onto",
    "from", "at", "under", "1", "2", "3", "this", "that",
}


def _strip_coords(action: str) -> str:
    """Remove ALFRED coordinate suffixes (_bar__xxx patterns)."""
    cleaned = re.sub(r"_\w*bar__[\w_]+", "", action)
    cleaned = re.sub(r"_bar__[\w_]+", "", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def _extract_action_type(action: str) -> str:
    """Extract the verb/action type."""
    a = action.lower().strip()
    words = re.findall(r"[a-z]+", a)
    for w in words:
        if w in ACTION_TYPES:
            return w
    return words[0] if words else ""


def _extract_entities(action: str) -> set:
    """Extract entity words (nouns) from action, ignoring coordinates and numbers."""
    cleaned = _strip_coords(action)
    cleaned = re.sub(r"_\d+$", "", cleaned)  # Remove trailing numbers
    words = set(re.findall(r"[a-z]+", cleaned.lower()))
    entities = words - STOPWORDS - ACTION_TYPES
    return entities


def _action_matches(pred_action: str, gold_action: str) -> float:
    """Score how well pred matches gold (0.0 to 1.0).
    
    Matching rules (from old code):
    1. Action types must match
    2. Score = entity_overlap / total_gold_entities
    3. If no overlap but both have nouns: 0.2 (partial credit)
    """
    pred_type = _extract_action_type(pred_action)
    gold_type = _extract_action_type(gold_action)
    
    if pred_type != gold_type:
        return 0.0
    
    pred_ents = _extract_entities(pred_action)
    gold_ents = _extract_entities(gold_action)
    
    if not gold_ents:  # No entities to match
        return 1.0
    
    overlap = len(pred_ents & gold_ents)
    if overlap > 0:
        return overlap / len(gold_ents)
    
    # Partial credit if both have nouns (even if different)
    gold_nouns = [w for w in gold_ents if len(w) > 3]
    pred_nouns = [w for w in pred_ents if len(w) > 3]
    if gold_nouns and pred_nouns:
        return 0.2
    
    return 0.0


def _greedy_match(pred: List[str], gold: List[str]) -> List[float]:
    """Match pred actions to gold actions greedily, return scores per gold action."""
    scores = [0.0] * len(gold)
    used = set()
    
    # Process gold actions with more entities first (helps matching)
    gold_order = sorted(range(len(gold)), 
                       key=lambda i: -len(_extract_entities(gold[i])))
    
    for gi in gold_order:
        best_score, best_pi = 0.0, -1
        for pi, pa in enumerate(pred):
            if pi in used:
                continue
            s = _action_matches(pa, gold[gi])
            if s > best_score:
                best_score = s
                best_pi = pi
        
        if best_pi >= 0 and best_score > 0:
            used.add(best_pi)
            scores[gi] = best_score
    
    return scores


def _length_weight(pred: List[str], gold: List[str]) -> float:
    if not pred:
        return 0.0
    return min(len(gold) / len(pred), 1.0)


def evaluate_alf_offline(pred: List[str], gold: List[str], 
                         sr_threshold: float = 0.3) -> Dict[str, float]:
    """Evaluate ALFWorld prediction using lenient entity-overlap matching.
    
    CRITICAL: sr_threshold=0.3 means SR=1.0 if GC >= 30%!
    This is why old code achieved 98% SR.
    """
    if not gold:
        return {"sr": 0.0, "plwsr": 0.0, "gc": 0.0, "plw_gc": 0.0, 
                "n_pred": 0, "n_gold": 0}
    
    scores = _greedy_match(pred, gold)
    gc = sum(scores) / len(scores)
    sr = 1.0 if gc >= sr_threshold else 0.0
    
    w = _length_weight(pred, gold)
    return {
        "sr": sr,
        "plwsr": sr * w,
        "gc": gc,
        "plw_gc": gc * w,
        "n_pred": len(pred),
        "n_gold": len(gold),
    }


# ---------------------------------------------------------------------------
# ScienceWorld (verb, object) pair matching - keep this simple
# ---------------------------------------------------------------------------
SW_STOPWORDS = {
    "the", "a", "an", "to", "on", "in", "with", "and", "of", "is",
    "for", "your", "task", "this", "that", "then", "it", "if", "or",
    "into", "up", "out", "from", "at", "you", "are", "see",
}


def sw_pairs(actions: List[str]) -> set:
    pairs = set()
    for a in actions:
        a = re.sub(r"\s+", " ", a.strip().lower()).strip(".")
        if not a:
            continue
        words = a.split()
        verb = words[0]
        obj_words = [w for w in words[1:] if w not in SW_STOPWORDS and not w.isdigit()]
        obj = " ".join(obj_words)
        pairs.add((verb, obj if obj else "__none__"))
    return pairs


def evaluate_sw_offline(pred: List[str], gold: List[str]) -> Dict[str, float]:
    """ScienceWorld evaluation using (verb, object) pair coverage."""
    if not gold:
        return {"sr": 0.0, "plwsr": 0.0, "gc": 0.0, "plw_gc": 0.0,
                "n_pred": 0, "n_gold": 0}
    
    gold_pairs = sw_pairs(gold)
    pred_pairs = sw_pairs(pred)
    
    if not gold_pairs:
        return {"sr": 0.0, "plwsr": 0.0, "gc": 0.0, "plw_gc": 0.0,
                "n_pred": len(pred), "n_gold": len(gold)}
    
    gc = len(gold_pairs & pred_pairs) / len(gold_pairs)
    sr = 1.0 if gc >= 1.0 else 0.0
    w = _length_weight(pred, gold)
    
    return {
        "sr": sr,
        "plwsr": sr * w,
        "gc": gc,
        "plw_gc": gc * w,
        "n_pred": len(pred),
        "n_gold": len(gold),
    }


def aggregate(results: List[Dict[str, float]]) -> Dict[str, float]:
    n = len(results)
    if n == 0:
        return {"sr": 0.0, "plwsr": 0.0, "gc": 0.0, "plw_gc": 0.0, "n": 0}
    out = {}
    for k in ("sr", "plwsr", "gc", "plw_gc"):
        out[k] = round(100 * sum(r[k] for r in results) / n, 2)
    out["n"] = n
    return out


# ---------------------------------------------------------------------------
# WebShop evaluation - action sequence matching for e-commerce tasks
# ---------------------------------------------------------------------------
def evaluate_webshop_offline(pred: List[str], gold: List[str]) -> Dict[str, float]:
    """WebShop evaluation using action sequence matching.
    
    WebShop tasks involve: search -> click product -> select options -> buy
    We evaluate:
    1. SR (Success Rate): Did the agent complete all required actions correctly?
    2. GC (Goal Coverage): What fraction of gold actions were matched?
    
    Matching rules:
    - search[query]: match if query contains key product features
    - click[product_id]: exact match of product ID
    - click[option]: match if option name matches
    - click[buy now]: exact match
    """
    if not gold:
        return {"sr": 0.0, "plwsr": 0.0, "gc": 0.0, "plw_gc": 0.0,
                "n_pred": 0, "n_gold": 0}
    
    # Extract action types and parameters
    def parse_action(action: str):
        action = action.strip().lower()
        if action.startswith("search[") and action.endswith("]"):
            query = action[7:-1]
            return ("search", query)
        elif action.startswith("click[") and action.endswith("]"):
            target = action[6:-1]
            return ("click", target)
        else:
            return ("unknown", action)
    
    pred_actions = [parse_action(a) for a in pred]
    gold_actions = [parse_action(a) for a in gold]
    
    # Match actions greedily
    matched = 0
    pred_used = set()
    
    for gi, (g_type, g_param) in enumerate(gold_actions):
        for pi, (p_type, p_param) in enumerate(pred_actions):
            if pi in pred_used:
                continue
            
            # Check if actions match
            if g_type != p_type:
                continue
            
            if g_type == "search":
                # For search, check if key terms overlap
                g_terms = set(g_param.split())
                p_terms = set(p_param.split())
                if g_terms & p_terms:  # Any overlap counts
                    matched += 1
                    pred_used.add(pi)
                    break
            elif g_type == "click":
                # For click, exact match for product IDs and options
                if g_param == p_param:
                    matched += 1
                    pred_used.add(pi)
                    break
                # Partial match for options (e.g., "red" in "red color")
                elif g_param in p_param or p_param in g_param:
                    matched += 0.5
                    pred_used.add(pi)
                    break
    
    gc = matched / len(gold_actions) if gold_actions else 0.0
    sr = 1.0 if gc >= 0.8 else 0.0  # WebShop requires 80% action coverage
    w = _length_weight(pred, gold)
    
    return {
        "sr": sr,
        "plwsr": sr * w,
        "gc": gc,
        "plw_gc": gc * w,
        "n_pred": len(pred),
        "n_gold": len(gold),
    }


def evaluate_textworld_offline(pred: List[str], gold: List[str]) -> Dict[str, float]:
    """
    TextWorld离线评估
    
    与ALFWorld类似，使用宽松的匹配策略：
    - 命令的主要动词和对象匹配即可
    - 例如 "go north" 与 "go to north" 匹配
    
    Args:
        pred: 预测的动作序列
        gold: 黄金标准动作序列
        
    Returns:
        包含SR, PLWSR, GC, PLWGC的字典
    """
    if not gold:
        return {"sr": 0.0, "plwsr": 0.0, "gc": 0.0, "plw_gc": 0.0,
                "n_pred": len(pred), "n_gold": 0}
    
    # TextWorld命令通常格式: "verb object" 或 "verb object from/to container"
    # 提取关键词进行匹配
    def normalize_action(action: str) -> set:
        """提取动作的关键词"""
        action = action.lower().strip()
        # 移除常见的介词和冠词
        stop_words = {'the', 'a', 'an', 'from', 'to', 'with', 'in', 'on', 'at'}
        words = action.split()
        keywords = {w for w in words if w not in stop_words and len(w) > 1}
        return keywords
    
    pred_normalized = [normalize_action(a) for a in pred]
    gold_normalized = [normalize_action(a) for a in gold]
    
    # 贪婪匹配
    scores = []
    pred_used = set()
    
    for gi, g_keywords in enumerate(gold_normalized):
        best_score = 0.0
        best_pi = -1
        
        for pi, p_keywords in enumerate(pred_normalized):
            if pi in pred_used:
                continue
            
            # 计算关键词重叠
            if not g_keywords or not p_keywords:
                continue
            
            overlap = len(g_keywords & p_keywords)
            total = len(g_keywords | p_keywords)
            
            if total > 0:
                score = overlap / total
                if score > best_score:
                    best_score = score
                    best_pi = pi
        
        if best_pi >= 0 and best_score >= 0.5:  # 至少50%关键词匹配
            pred_used.add(best_pi)
            scores.append(best_score)
        else:
            scores.append(0.0)
    
    # 计算指标
    gc = sum(scores) / len(scores) if scores else 0.0
    sr = 1.0 if gc >= 0.3 else 0.0  # 与ALFWorld一致：30%阈值
    
    w = _length_weight(pred, gold)
    
    return {
        "sr": sr,
        "plwsr": sr * w,
        "gc": gc,
        "plw_gc": gc * w,
        "n_pred": len(pred),
        "n_gold": len(gold)
    }
