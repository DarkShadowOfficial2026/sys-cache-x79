"""
what ever i dont have time to write full details lol u can write one but not so much needed XD
"""

!pip install -q --upgrade unsloth # type: ignore
!pip install -q --upgrade xformers # type: ignore
!pip install -q triton # type: ignore

from __future__ import annotations

import json
import logging
import math
import os
import random
import re
import threading
import unsloth # type: ignore
from threading import Thread
import time
from collections import defaultdict, deque
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Callable, Deque, Dict, List, NamedTuple, Optional, Set
import torch
from transformers import pipeline
from transformers.utils import logging as hf_logging
import sys
from unsloth import FastLanguageModel
from transformers import TextIteratorStreamer


# ---------------------------------------------------------------------------
# Default model names (all encoder-only)
# ---------------------------------------------------------------------------


main_generative_model = "skshmjn/unsloth_llama-3.2-3B-instruct-uncenssored" # can be"bunnycore/Llama-3.2-3B-All-Mix"
core_reasoning_model = "huggingkot/Llama-3-8B-Instruct-abliterated-v2-bnb-4bit"

token_classification_model = "dslim/bert-base-NER"
text_reranker_model = "Alibaba-NLP/gte-reranker-modernbert-base"
cross_encoder_base_model = "cross-encoder/nli-deberta-v3-large" 
sentence_embeder_model = "sentence-transformers/all-mpnet-base-v2"
emotion_classification_model = "SamLowe/roberta-base-go_emotions"
text_classification_model = "MoritzLaurer/deberta-v3-large-zeroshot-v2.0" 




# ---------------------------------------------------------------------------
# Global models -- loaded once, right here, at import time
# ---------------------------------------------------------------------------

inference_labels = ["contradiction", "entailment", "neutral"]

#-------------------------------------------------------
# Logger start
#-------------------------------------------------------
hf_logging.set_verbosity_error()  
logger1 = logging.getLogger("silence_detector")
_device_ = 0 if torch.cuda.is_available() else -1
logger1.info("Loading NLI model '%s' on device %s", text_classification_model, _device_)

logger2 = logging.getLogger("state_inactivity_detector")
device_index = 0 if torch.cuda.is_available() else -1
logger2.info("Loading emotion model '%s' on device %s", emotion_classification_model, device_index)


#-------------------------------------------------------
# Logger end
#-------------------------------------------------------

#---------------------------------------------
#   Main LLM loads Start
#---------------------------------------------

print("Loading Model 1...")
engine_1, tokenizer_1 = FastLanguageModel.from_pretrained(
    model_name=main_generative_model, max_seq_length=1024,
    dtype=torch.float16, load_in_4bit=True,
)
FastLanguageModel.for_inference(engine_1)
engine_1.generation_config.pad_token_id = tokenizer_1.eos_token_id
engine_1.generation_config.max_length = None

print("Loading Model 2...")
engine_2, tokenizer_2 = FastLanguageModel.from_pretrained(
    model_name=core_reasoning_model, max_seq_length=1024,
    dtype=torch.float16, load_in_4bit=True,
)
FastLanguageModel.for_inference(engine_2)
engine_2.generation_config.pad_token_id = tokenizer_2.eos_token_id
engine_2.generation_config.max_length = None

print("Large Models loaded.")



# ------------------ Main loads of model FACTORY ------------------
def main_engine_config(name, engine, tokenizer, system_prompt,
               max_new_tokens=256, temperature=0.0, top_p=0.9,
               repetition_penalty=1.0, color="\033[0m"):
    """Returns a function f(user_text) -> str. temperature=0 means greedy (fastest, deterministic)."""

    def run(user_text, show_stats=True):
        chat = [{"role": "system", "content": system_prompt},
                {"role": "user", "content": user_text}]
        prompt = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

        streamer = CountingStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

        gen_kwargs = dict(
            **inputs, streamer=streamer, max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.eos_token_id, num_beams=1,
            repetition_penalty=repetition_penalty,
        )
        if temperature and temperature > 0:
            gen_kwargs.update(do_sample=True, temperature=temperature, top_p=top_p)
        else:
            gen_kwargs.update(do_sample=False)

        def _generate():
            with torch.inference_mode():
                engine.generate(**gen_kwargs)

        t = Thread(target=_generate)
        print(f"\n--------------------------------------------------\n🤖 [{name}]: {color}", end="")
        t0 = time.time()
        t.start()

        text = ""
        for chunk in streamer:
            text += chunk
            sys.stdout.write(chunk)
            sys.stdout.flush()
        t.join()
        t_end = time.time()
        print("\033[0m")

        if show_stats and streamer.first_token_time and streamer.n_tokens > 1:
            decode_time = t_end - streamer.first_token_time
            speed = (streamer.n_tokens - 1) / max(decode_time, 1e-6)
            print(f"\033[92m⚡ {name}: {speed:.2f} tok/s | {streamer.n_tokens} tokens | "
                  f"first token {streamer.first_token_time - t0:.2f}s | total {t_end - t0:.2f}s\033[0m")
        print("--------------------------------------------------")
        return text

    return run

print(torch.cuda.memory_allocated() / 1e9, "GB")


#---------------------------------------------
#   Main LLM loads Ends
#---------------------------------------------


try:
    from transformers import pipeline
    NER_MODEL = pipeline("ner", model=token_classification_model, aggregation_strategy="simple")
except Exception as e:
    print(f"[warn] failed to load NER model: {e}")
    NER_MODEL = None

try:
    from sentence_transformers import CrossEncoder
    RERANKER_MODEL = CrossEncoder(text_reranker_model)
except Exception as e:
    print(f"[warn] failed to load reranker model: {e}")
    RERANKER_MODEL = None

try:
    from sentence_transformers import CrossEncoder
    ENCODER_MODEL = CrossEncoder(cross_encoder_base_model)
except Exception as e:
    print(f"[warn] failed to load NLI model: {e}")
    ENCODER_MODEL = None




try:
    from sentence_transformers import SentenceTransformer
    EMBED_MODEL = SentenceTransformer(sentence_embeder_model)
except Exception as e:
    print(f"[warn] failed to load embedder model: {e}")
    EMBED_MODEL = None



try:
    CLASSIFICATION_MODEL = pipeline("zero-shot-classification", model=text_classification_model, device=_device_)
except Exception as e:
    logger1.warning("failed to load NLI model '%s': %s", text_classification_model, e)
    CLASSIFICATION_MODEL = None
try:
    emotion_classification = pipeline(
        "text-classification",
        model=emotion_classification_model,
        top_k=None,
        device=device_index,
    )
except Exception as exc:
    logger2.warning("failed to load emotion model '%s': %s", emotion_classification_model, exc)
    emotion_classification = None



print(torch.cuda.memory_allocated() / 1e9, "GB")
# ---------------------------------------------------------------------------
# Global models -- load end
# ---------------------------------------------------------------------------

# Non-semantic preprocessing only (Unicode range strip) -- not part of the
# intent decision, so it doesn't conflict with "no keyword matching for
# meaning".
_emoji_noise_cleaner = re.compile(
    "["
    "\U0001F300-\U0001FAFF"
    "\U00002600-\U000027BF"
    "\U0001F1E6-\U0001F1FF"
    "]+",
    flags=re.UNICODE,
)

NEUTRAL_LABEL = "neutral"
UNKNOWN_SPEAKER = "unknown"
if emotion_classification is None:
    logger2.error(
        "EMOTION_CLASSIFIER (emotion_classification) failed to load — the affect engine (layer 2) will "
        "never change mood and the back-to-back trigger (layer 1) will never "
        "fire. Check model download / network access / torch install."
    )
log = logging.getLogger(__name__)
_EPS = 1e-9  # float tolerance so 50 * 0.02 really counts as 1.0


def strip_emoji(text: str) -> str:
    return _emoji_noise_cleaner.sub("", text).strip()

def split_line(line: str):
    """'felix: did you finish...' -> ('felix', 'did you finish...')"""
    speaker, text = line.split(":", 1)
    return speaker.strip().lower(), text.strip()

def parse_chat_line(line: str) -> tuple[str, str]:
    """'felix: did you finish...' -> ('felix', 'did you finish...').
    Lines without a 'speaker:' prefix are attributed to UNKNOWN_SPEAKER."""
    if ":" not in line:
        return UNKNOWN_SPEAKER, line.strip()
    speaker, text = line.split(":", 1)
    return speaker.strip().lower(), text.strip()
class Stage(NamedTuple):
    name: str       # stage name exposed to other layers
    emotion: str    # Layer-2 emotion that seeds the meter (read-only)
    meaning: str    # what she's thinking at this stage


def _clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, x))



#----------------------------------
#
# VALUES START
#
#----------------------------------

# (valence, activation) per emotion label, both in [-1, 1]
#   valence    : unpleasant (-1) ... pleasant (+1)
#   activation : calm / low energy (-1) ... energized / high energy (+1)
EMOTION_VALENCE_ACTIVATION: dict[str, tuple[float, float]] = {
    "admiration": (0.7, 0.4), "amusement": (0.8, 0.5), "anger": (-0.8, 0.8),
    "annoyance": (-0.5, 0.5), "approval": (0.5, 0.2), "caring": (0.6, 0.2),
    "confusion": (-0.2, 0.4), "curiosity": (0.3, 0.5), "desire": (0.4, 0.6),
    "disappointment": (-0.6, -0.1), "disapproval": (-0.5, 0.3), "disgust": (-0.7, 0.4),
    "embarrassment": (-0.5, 0.5), "excitement": (0.8, 0.9), "fear": (-0.7, 0.8),
    "gratitude": (0.8, 0.2), "grief": (-0.9, -0.2), "joy": (0.9, 0.6),
    "love": (0.9, 0.4), "nervousness": (-0.4, 0.7), "optimism": (0.7, 0.4),
    "pride": (0.7, 0.5), "realization": (0.1, 0.3), "relief": (0.6, -0.3),
    "remorse": (-0.6, 0.0), "sadness": (-0.8, -0.3), "surprise": (0.2, 0.8),
    NEUTRAL_LABEL: (0.0, 0.0),
}
# How strongly each emotion pulls kanako into talking again right after her own
# line (0-1). Emotions that are not listed can never be the focus of a back-to-back.
BACK_TO_BACK_TRIGGER_WEIGHTS: dict[str, float] = {
    "excitement": 1.00, "surprise": 0.95, "confusion": 0.95, "anger": 0.90,
    "amusement": 0.85, "embarrassment": 0.80, "nervousness": 0.80, "fear": 0.75,
    "realization": 0.75, "annoyance": 0.75, "joy": 0.70, "curiosity": 0.60,
    "desire": 0.60, "disgust": 0.60, "disappointment": 0.55, "love": 0.50,
}



DEFAULT_LADDER: tuple[Stage, ...] = (
    Stage("curiosity",   "curiosity",      "they might just be busy — where are they?"),
    Stage("nervousness", "nervousness",    "something might be off"),
    Stage("anxiety",     "fear",           "something might be wrong"),
    Stage("sadness",     "sadness",        "they forgot / they don't care"),
    Stage("anger",       "anger",          "how dare they make me wait"),
    Stage("coldness",    "disappointment", "I'm done, whatever"),
)

# stage name -> Layer-2 emotion it writes when its interject fires (or lapses
# silently), and how strongly (0-1). Only used by the waiting callbacks.
STAGE_EMOTION: dict[str, str] = {s.name: s.emotion for s in DEFAULT_LADDER}
WAITING_EMOTION_LEVEL = 0.6

# which kind of line the current feeling produces (the line writer can use this)
INTENT_BY_EMOTION: dict[str, str] = {
    "curiosity": "playful_poke", "nervousness": "check_in", "fear": "worried_check_in",
    "sadness": "hurt", "anger": "irritated", "disappointment": "cold",
}




#----------------------------------
#
# VALUES END
#
#----------------------------------



# ------------------ STREAMER THAT COUNTS REAL TOKENS ------------------
class CountingStreamer(TextIteratorStreamer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.n_tokens = 0
        self.first_token_time = None

    def put(self, value):
        # the first put() call is the prompt, skip it
        if not (self.skip_prompt and self.next_tokens_are_prompt):
            if self.first_token_time is None:
                self.first_token_time = time.time()
            self.n_tokens += value.numel()
        super().put(value)




class AddresseeDetector:
    def __init__(
        self,
        ner=None,
        reranker=None,
        nli=None,
        embedder=None,
        ner_confidence_threshold: float = 0.80,
        rrf_k: int = 60,
    ):
        """
        Pass `ner` / `reranker` / `nli` / `embedder` yourself (e.g. fakes,
        or models you already loaded elsewhere) to override. Otherwise this
        falls back to the module-level globals loaded once at import time.
        """
        self.ner = ner if ner is not None else NER_MODEL
        self.reranker = reranker if reranker is not None else RERANKER_MODEL
        self.nli = nli if nli is not None else ENCODER_MODEL
        self.embedder = embedder if embedder is not None else EMBED_MODEL

        self.ner_confidence_threshold = ner_confidence_threshold
        self.rrf_k = rrf_k

    # -- name discovery (replaces the hardcoded extra_known_speakers) -----

    def arg_named(self, conversations: List[str]) -> Set[str]:
        """
        Everyone who has posted a line, PLUS everyone the NER model finds
        named inside the message text itself.

        Chat text is casual and usually lowercase, and BERT-NER checkpoints
        lean heavily on capitalization cues from their training data
        (CoNLL-2003 newswire). Title-casing each line before tagging
        restores that cue while the model still uses surrounding context
        to decide. It's a mitigation, not a fix -- unusual names in very
        short messages can still be missed, which is why this sits
        *alongside* the ensemble layer rather than being the only thing
        standing between you and a wrong answer. A confidence threshold
        (not a keyword list) is what filters out slang words that get
        capitalized into looking name-shaped.
        """
        names = {split_line(line)[0] for line in conversations}

        for line in conversations:
            _, text = split_line(line)
            titled = " ".join(w.capitalize() for w in text.split())
            for ent in self.ner(titled):
                if ent.get("entity_group") != "PER":
                    continue
                if ent.get("score", 1.0) < self.ner_confidence_threshold:
                    continue
                name = re.sub(r"[^a-z']", "", ent["word"].strip().lower())
                # a one-letter "name" is an NER artifact: title-casing makes the pronoun
                # "i" look like a person, which then fired the explicit-mention rule and
                # the phantom-name bypass on every "i will ..." line
                if len(name) > 1:
                    names.add(name)

        return names

    # -- layer 1: rule-based explicit mention ------------------------------

    @staticmethod
    def rule_based_addressee(text: str, speaker: str, speakers: Set[str]) -> Optional[str]:
        """
        A name mentioned directly is the strongest signal available.
        Vocative position ("dana, can you...") is checked first, mid-
        sentence mention ("...ask dana...") second. Either way this beats
        the encoder layer when it fires -- it's cheap and near-deterministic.
        """
        words = re.findall(r"[a-zA-Z']+", text.lower())
        if not words:
            return None

        if words[0] in speakers and words[0] != speaker:
            return words[0]

        for w in words:
            if w in speakers and w != speaker:
                return w

        return None

    # -- topic profiles -----------------------------------------------------

    @staticmethod
    def speaker_topic_context(conversations: List[str], self_speaker: str) -> Dict[str, str]:
        """Everything each OTHER speaker has said so far, as one text blob per person."""
        context: Dict[str, List[str]] = defaultdict(list)
        for line in conversations:
            s, t = split_line(line)
            if s != self_speaker:
                context[s].append(t)
        return {s: " ".join(lines) for s, lines in context.items()}

    # -- layer 2: three independent encoder scorers -------------------------

    def score_reranker(self, text: str, profiles: Dict[str, str]) -> Dict[str, float]:
        names = list(profiles.keys())
        pairs = [(text, profiles[n]) for n in names]
        scores = self.reranker.predict(pairs)
        return {n: float(s) for n, s in zip(names, scores)}

    def score_nli(self, text: str, profiles: Dict[str, str]) -> Dict[str, float]:
        """
        Zero-shot reuse of NLI (Yin et al. 2019 style): premise = the
        candidate's profile, hypothesis = the incoming message, score =
        P(entailment). This isn't "classic" NLI usage -- there's no crisp
        logical entailment between two chat blurbs -- but entailment
        probability from an NLI-trained encoder correlates well with
        topical/semantic fit, which is exactly the well-known trick behind
        NLI-based zero-shot classification. Kept as one vote among three,
        not the sole signal, since it's a repurposing rather than the
        model's original objective.
        """
        import torch

        names = list(profiles.keys())
        pairs = [(profiles[n], text) for n in names]
        logits = self.nli.predict(pairs)
        probs = torch.softmax(torch.tensor(logits), dim=1)
        entail_idx = inference_labels.index("entailment")
        return {n: float(probs[i, entail_idx]) for i, n in enumerate(names)}

    def score_embedding(self, text: str, profiles: Dict[str, str]) -> Dict[str, float]:
        from sentence_transformers import util

        names = list(profiles.keys())
        msg_emb = self.embedder.encode([text], convert_to_tensor=True)
        profile_embs = self.embedder.encode([profiles[n] for n in names], convert_to_tensor=True)
        sims = util.cos_sim(msg_emb, profile_embs)[0]
        return {n: float(sims[i]) for i, n in enumerate(names)}

    # -- fusion ---------------------------------------------------------------

    def reciprocal_rank_fusion(self, score_dicts: List[Dict[str, float]]) -> Dict[str, float]:
        """
        Combine several rankings into one WITHOUT assuming the raw scores
        are on comparable scales -- a reranker logit, an NLI probability
        and a cosine similarity are not directly comparable, but "1st
        place vs 2nd place" is. Fusing on rank sidesteps calibration
        entirely, which is the standard reason RRF is used to combine
        heterogeneous retrieval/ranking signals.
        """
        fused: Dict[str, float] = defaultdict(float)
        for scores in score_dicts:
            ranked = sorted(scores, key=lambda n: -scores[n])
            for rank, name in enumerate(ranked):
                fused[name] += 1.0 / (self.rrf_k + rank + 1)
        return dict(fused)

    TEMP, TURN_DECAY, MIN_MARGIN = 0.10, 0.60, 0.10
    AFFINITY_DECAY, AFFINITY_SHARE = 0.90, 0.15
    W_RERANK, W_EMBED, W_NLI, W_TURN = 0.45, 0.15, 0.10, 0.30

    # ---- "ask" signal (experimental, switch off to get the old behaviour back) ----
    # The topical scorers above cannot tell "a question for the bot" from "chat
    # between two friends": both are on the same topic. This adds ONE separate
    # zero-shot check that looks at the SHAPE of the line instead of its topic.
    # It only ever moves kanako's score (up for info-seeking lines, down for
    # chatter), and it is made of two independent checks that must agree.
    BOT = "kanako"
    USE_ASK_SIGNAL = True
    W_ASK = 0.30             # max push on kanako's fused score (+/-)
    ASK_SOLO_THRESH = 0.70   # only one human in the window: still answer if the line clearly asks

    # ---- ask GATE: stops the topical scorers from deciding "is this for the bot" ----
    # rerank/embed/nli/turn measure "same topic / same pair", which cannot separate a
    # question to the group from a reply inside a pair. The ask signal can, so it gates.
    # Tune LO/HI/TIE with replay_gate.py on the full field-test log before trusting them.
    USE_ASK_GATE = True
    ASK_GATE_LO = 0.60       # ask below this: line is chatter, the bot is never the addressee
    ASK_HI = 0.85            # ask above this: line clearly seeks outside help, the bot answers
    ASK_TIE = 0.65           # fused scores too close to call: the bot only if ask is at least this
    # Optional extra chatter hypothesis for lines that GIVE advice ("you could check...").
    # Changes every ask value, so turn on only after a full re-run of the field test.
    USE_ADVICE_HYP = False
    ADVICE_HYPOTHESIS = "The speaker is giving advice, an answer, or information to someone who asked them."

    # ---- context check (added after the hard test) ---------------------------------
    # ask_signal only sees ONE line, so "how long did it take YOU" or "shorten THAT"
    # look the same as any other question/command. This second zero-shot check scores
    # the line together with the last few lines as the premise (no keywords). It only
    # acts at the edges of the gate, so lines the gate already gets right are untouched:
    #   instr   : the line tells the bot to change/shorten/continue its last answer
    #             -> rescue the bot even when the line looks like chatter
    #   partner : the line asks the person they were just talking to about THEMSELVES
    #             -> take the bot out when the gate would have picked her
    #   react   : the line builds on what the bot just said. Riskiest rule (it can
    #             wake the bot on pair follow-ups), so it is LOG ONLY until you set a
    #             threshold from the [ctx] lines in your logs with replay_gate.py.
    # RESULT OF THE COMBINED TEST: these three scores did not separate the cases
    # (instr 0.02-0.67 on real instructions, react overlapped between wanted and unwanted
    # lines, and the partner veto flipped two lines that used to pass without fixing the
    # two it was built for). They are OFF; the code stays so a calibrated version can
    # be tried later with replay_gate.py.
    USE_CTX = False
    LOG_CTX_ALWAYS = False   # True = score every model-route line so logs can calibrate
    CTX_LINES = 3            # how many previous lines go into the premise
    CTX_INSTR_THR = None     # None = rule off
    CTX_PARTNER_THR = None   # None = veto off
    CTX_REACT_THR = None     # None = rule off

    # ---- follow-up from the person she just answered (dialogue state, no thresholds) ----
    # 14 of the 14 first-in-chain misses in the combined test were lines like "shorten
    # that" or "in that case i will..." sent by the very person Kanako had just answered,
    # on the line right after her reply. Every unwanted line in the same position came
    # from a DIFFERENT person. main_execution records who she last answered in
    # `last_served`; the same person speaking again straight after her answer is still
    # talking to her unless a name says otherwise (the name rule runs before this).
    USE_SERVED_ASKER = True
    last_served = None
    CTX_HYPOTHESES = {
        "react": "The last speaker is reacting to or building on what {bot} just said.",
        "instr": "The last speaker is telling {bot} to change, shorten, repeat or continue {bot}'s previous answer.",
        "partner": "The last speaker is asking the person they were just talking to about that person's own life, experience, or belongings.",
    }
    ASK_HYPOTHESES = (
        "The speaker is asking a question that needs knowledge, advice, or an explanation.",
        "The speaker is asking for a recommendation or for help solving a problem.",
    )
    CHATTER_HYPOTHESES = (
        "The speaker is arranging plans or logistics with a friend.",
        "The speaker is commenting on or agreeing with what someone else just said.",
        # added after the hard test: questions that look like questions but are
        # about the other person's own life or about the group's plans
        "The speaker is asking someone about their own personal life, experiences, or belongings.",
        "The speaker is asking the group about plans, schedules, or who is bringing what.",
    )

    def ask_signal(self, text: str) -> float:
        """0..1. 1 = clearly seeks information or advice, 0 = clearly friend chatter,
        0.5 = no opinion (model missing or failed). Zero-shot NLI, no keywords."""
        if CLASSIFICATION_MODEL is None or not text:
            return 0.5
        try:
            chatter = self.CHATTER_HYPOTHESES + ((self.ADVICE_HYPOTHESIS,) if self.USE_ADVICE_HYP else ())
            hyps = list(self.ASK_HYPOTHESES + chatter)
            out = CLASSIFICATION_MODEL(text, hyps, hypothesis_template="{}", multi_label=True)
            sc = dict(zip(out["labels"], out["scores"]))
            ask = max(sc[h] for h in self.ASK_HYPOTHESES)
            chat = max(sc[h] for h in chatter)
            return _clamp(0.5 + 0.5 * (ask - chat))
        except Exception as exc:
            logger1.warning("ask_signal failed: %s", exc)
            return 0.5

    def context_signals(self, incoming: str, conversations: List[str]):
        """{'react','instr','partner'} in 0..1, or None if the model is missing/fails.
        Premise = the last few lines + the new line (the duplicate bot lines that
        main_execution writes into `conversations` are collapsed)."""
        if CLASSIFICATION_MODEL is None:
            return None
        try:
            lines = []
            for ln in conversations[-(self.CTX_LINES + 3):] + [incoming]:
                if not lines or lines[-1] != ln:
                    lines.append(ln)
            premise = "\n".join(lines[-(self.CTX_LINES + 1):])
            hyps = {k: v.format(bot=self.BOT) for k, v in self.CTX_HYPOTHESES.items()}
            out = CLASSIFICATION_MODEL(premise, list(hyps.values()),
                                       hypothesis_template="{}", multi_label=True)
            sc = dict(zip(out["labels"], out["scores"]))
            return {k: float(sc[v]) for k, v in hyps.items()}
        except Exception as exc:
            logger1.warning("context_signals failed: %s", exc)
            return None

    def _dist(self, scores):
        names = list(scores)
        p = torch.softmax(torch.tensor([scores[n] for n in names]) / self.TEMP, dim=0)
        return {n: float(x) for n, x in zip(names, p)}


    def turn_prior(self, conversations, speaker, candidates):
        """Recency (who spoke last) blended with affinity (who has traded
        adjacent turns with this speaker, recent exchanges counting more)."""
        n = len(conversations)
        recency, affinity = defaultdict(float), defaultdict(float)

        gap = 0
        for line in reversed(conversations):
            who, _ = split_line(line)
            if who == speaker:
                continue
            gap += 1
            recency.setdefault(who, self.TURN_DECAY ** (gap - 1))

        for i, line in enumerate(conversations):
            who, _ = split_line(line)
            if who == speaker:
                continue
            hits = sum(1 for j in (i - 1, i + 1)
                       if 0 <= j < n and split_line(conversations[j])[0] == speaker)
            affinity[who] += hits * self.AFFINITY_DECAY ** (n - 1 - i)

        def norm(d):
            total = sum(d.get(c, 0.0) for c in candidates) or 1.0
            return {c: d.get(c, 0.0) / total for c in candidates}

        r, a = norm(recency), norm(affinity)
        s = self.AFFINITY_SHARE
        return {c: (1 - s) * r[c] + s * a[c] for c in candidates}

    # -- entry point ------------------------------------------------------


    def find_addressee(self, incoming: str, conversations: List[str], verbose: bool = False) -> Optional[str]:
        self.last_route = "model"
        recent_conversations = conversations[-4:]
        speaker, text = split_line(incoming)
        speakers = self.arg_named(recent_conversations) | {"kanako"}

        addressee = self.rule_based_addressee(text, speaker, speakers)
        if addressee:
            self.last_route = "rule"
            if verbose:
                print(f"  [rule]  explicit mention -> {addressee}")
            return addressee

        profiles = self.speaker_topic_context(recent_conversations, speaker)
        if not profiles:
            return None

        if len(profiles) == 1:
            only = next(iter(profiles))
            if self.USE_ASK_SIGNAL and only != self.BOT:
                ask = self.ask_signal(text)
                if ask >= self.ASK_SOLO_THRESH:
                    if verbose:
                        print(f"  [ask]   only {only} in window, but the line asks ({ask:.2f}) -> {self.BOT}")
                    return self.BOT
            return only

        rerank = self._dist(self.score_reranker(text, profiles))
        embed = self._dist(self.score_embedding(text, profiles))

        # --- NLI vote: only if the CrossEncoder actually loaded ---
        if self.nli is not None:
            nli = self._dist(self.score_nli(text, profiles))
            turn = self.turn_prior(conversations[-12:], speaker, list(profiles))
            fused = {n: self.W_RERANK * rerank[n] + self.W_EMBED * embed[n]
                     + self.W_NLI * nli[n] + self.W_TURN * turn[n]
                     for n in profiles}
            if verbose:
                print(f"  [rerank] {rerank}\n  [embed] {embed}\n  [nli] {nli}\n  [turn] {turn}")
        else:
            # graceful fallback: renormalize the three signals we DO have
            turn = self.turn_prior(conversations[-12:], speaker, list(profiles))
            tot = self.W_RERANK + self.W_EMBED + self.W_TURN
            fused = {n: (self.W_RERANK / tot) * rerank[n]
                     + (self.W_EMBED  / tot) * embed[n]
                     + (self.W_TURN   / tot) * turn[n]
                     for n in profiles}
            if verbose:
                print(f"  [rerank] {rerank}\n  [embed] {embed}\n  [turn] {turn}  (nli off)")

        ask = self.ask_signal(text) if self.USE_ASK_SIGNAL else None
        if ask is not None and self.BOT in fused:
            push = self.W_ASK * (2.0 * ask - 1.0)
            fused[self.BOT] += push
            if verbose:
                print(f"  [ask] {ask:.2f} -> {self.BOT} {push:+.2f}")

        ranked = sorted(fused.items(), key=lambda kv: -kv[1])
        margin = ranked[0][1] - ranked[1][1]

        if verbose:
            print(f"  [fused] {ranked}  margin={margin:.2f}")

        if self.USE_ASK_GATE and ask is not None:
            bot_last = bool(conversations) and split_line(conversations[-1])[0] == self.BOT
            cache = {}

            def ctx():
                if "v" not in cache:
                    cache["v"] = self.context_signals(incoming, conversations) if self.USE_CTX else None
                    if verbose and cache["v"]:
                        c = cache["v"]
                        print(f"  [ctx]   react={c['react']:.2f} instr={c['instr']:.2f} "
                              f"partner={c['partner']:.2f} bot_last={int(bot_last)}")
                return cache["v"]

            if self.USE_CTX and self.LOG_CTX_ALWAYS:
                ctx()
            served_asker = bool(self.USE_SERVED_ASKER and bot_last and speaker == self.last_served)
            if verbose:
                print(f"  [serve] bot_last={int(bot_last)} served_asker={int(served_asker)}")
            return self._gate(ranked, margin, ask, ctx, bot_last, verbose, served_asker)
        return ranked[0][0] if margin >= self.MIN_MARGIN else None

    def _gate(self, ranked, margin, ask: float, ctx=None, bot_last: bool = False,
              verbose: bool = False, served_asker: bool = False) -> str:
        """Final call. Never returns None, so a close race can no longer fall through
        to the caller's `or "kanako"` default; the ask signal breaks the tie instead.
        `ctx` is a lazy callable returning the context scores (or None).
        replay_gate.py mirrors this logic, keep the two in sync."""
        best_human = next(n for n, _ in ranked if n != self.BOT)
        if ask >= self.ASK_HI:
            pick, why = self.BOT, "strong ask"
        elif served_asker:
            pick, why = self.BOT, "follow-up from the person she just answered"
        elif ask < self.ASK_GATE_LO:
            pick, why = best_human, "chatter shape"
            # rescue: an instruction aimed at the bot's own last answer
            if bot_last and ctx is not None:
                c = ctx()
                if c and (c["instr"] >= self.CTX_INSTR_THR
                          or (self.CTX_REACT_THR is not None and c["react"] >= self.CTX_REACT_THR)):
                    pick, why = self.BOT, "instruction to the bot's last answer"
        else:
            if margin >= self.MIN_MARGIN:
                pick, why = ranked[0][0], "fused winner"
            else:
                pick, why = (self.BOT if ask >= self.ASK_TIE else best_human), "tie-break on ask"
            # veto: a question to the person they were just talking to, about themselves
            if pick == self.BOT and ctx is not None:
                c = ctx()
                if c and c["partner"] >= self.CTX_PARTNER_THR:
                    pick, why = best_human, "question to the other person about themselves"
        if verbose:
            print(f"  [gate]  {why} -> {pick}")
        return pick

detector = AddresseeDetector()


@dataclass(frozen=True)
class DecisionConfig:
    bot_name: str = "kanako"
    context_turns: int = 4
    is_direct_chat: bool = True

    # Whether the single competitive classification sees the full recent
    # conversation or just the current line. Full context generally helps
    # distinguish reaction from order ("did they just get surprising news").
    context_mode: str = "full"  # "full" | "solo"

    order_thresh: float = 0.45      # winning "order" probability required
    safety_margin: float = 0.15     # required lead over the runner-up label


@dataclass
class DetectionResult:
    silence: bool
    stage: str
    order: float = 0.0
    joke: float = 0.0
    other: float = 0.0
    topic: float = 0.0
    reaction: float = 0.0
    none_: float = 0.0
    third_party: float = 0.0
    margin: float = 0.0
    # True when "order" was the single highest-scoring category but still
    # got rejected by the strict threshold/margin gate -- a genuinely
    # contested call, not a confident "this isn't an order". See
    # AgentSilenceController.handle() for how this feeds the rebellion
    # layer instead of being silently dropped.
    ambiguous: bool = False

    def __str__(self) -> str:
        return (f"silence={self.silence} stage={self.stage} "
                f"order={self.order:.2f} joke={self.joke:.2f} "
                f"other={self.other:.2f} topic={self.topic:.2f} "
                f"reaction={self.reaction:.2f} third_party={self.third_party:.2f} "
                f"none={self.none_:.2f} margin={self.margin:.2f} "
                f"ambiguous={self.ambiguous}")


class SilenceOrderDecision:
    def __init__(self, config: Optional[DecisionConfig] = None):
        self.cfg = config or DecisionConfig()
        self.history: Deque[tuple[str, str]] = deque(maxlen=self.cfg.context_turns)

    # ---- hypothesis set: ONE mutually-exclusive competition --------------

    def _intent_hypotheses(
        self, speaker: str, history: list[tuple[str, str]]
    ) -> dict[str, str]:
        target = "the other person" if self.cfg.is_direct_chat else self.cfg.bot_name

        hyps = {
            "order": (
                f"{speaker} is seriously telling {target} to stop talking or "
                f"be quiet, even if it is said with frustration, impatience, "
                f"or annoyance."
            ),
            # NOTE: a version of this hypothesis once added an explicit
            # exclusion clause ("...and is not about adjusting a device,
            # app, or volume setting") to fix a single false positive on
            # "turn the volume down on the video". Reverted: the extra
            # clause measurably diluted confidence on genuine orders across
            # the board (order category dropped from 87% to 73% in testing,
            # including previously-solid cases like "kanako be quiet" and
            # "kanako stop talking" dropping close to or below threshold),
            # apparently because longer, compound hypotheses containing a
            # clause irrelevant to the sentence being scored reduce the
            # model's overall confidence, not just its confidence in the
            # irrelevant part. The volume/device false positive is kept as
            # a known, accepted limitation rather than patched again.
            "joke": (
                f"{speaker} is joking, teasing, or being sarcastic, "
                f"not making a real request of {target}."
            ),
            "reaction": (
                f"{speaker} is having an emotional reaction or exclaiming in "
                f"shock, surprise, or frustration, without directing any "
                f"instruction at {target}."
            ),
            "topic": (
                f"{speaker} is talking about something unrelated to telling "
                f"{target} to be quiet, such as a device, app, media, place, "
                f"or the meaning of a word."
            ),
        }

        # NOTE: an earlier version added an unconditional "third_party"
        # hypothesis here to catch cases like "can you get him to stop
        # talking" with no prior speaker history. Tested and reverted: it
        # was generically plausible-sounding enough to compete for mass on
        # ANY order sentence, not just ones actually referencing a third
        # party -- it reproduced the exact "none"-catch-all failure mode we
        # removed earlier (order category accuracy dropped from 87% to 47%
        # in testing). The bare "third party, no prior history" case is a
        # known, accepted limitation rather than something worth an
        # ungrounded competing hypothesis to chase. The history-conditioned
        # _other_hypothesis below (which only fires once a distinct named
        # speaker has actually appeared) is kept, since it's grounded in
        # real evidence rather than a generic hedge.

        other_hyp = self._other_hypothesis(speaker, history)
        if other_hyp is not None:
            hyps["other"] = other_hyp

        # Deliberately no "none of the above" catch-all: when the model is
        # unsure, that hedge is trivially easy to entail and it silently
        # absorbs probability mass that should go to whichever real category
        # is actually closest -- this is what capped "order" at 0.44 against
        # a 0.43 "none" score for a genuine (if idiomatically phrased) order.
        # Dropping it forces the competition to stay among categories that
        # are actually meaningful.
        return hyps

    def _other_hypothesis(self, speaker: str, history: list[tuple[str, str]]) -> Optional[str]:
        if self.cfg.is_direct_chat:
            return None
        bot_lower = self.cfg.bot_name.lower()
        others = sorted({s for s, _ in history if s != speaker and s != bot_lower})
        if not others:
            return None
        return f"{speaker} is telling {', '.join(others)} to be quiet, not {self.cfg.bot_name}."

    # ---- plumbing ----------------------------------------------------

    @staticmethod
    def _parse(line: str) -> tuple[str, str]:
        speaker, text = line.split(":", 1)
        return speaker.strip().lower(), strip_emoji(text.strip())

    @staticmethod
    def _build_premise(history: list[tuple[str, str]], speaker: str, text: str) -> str:
        lines = [f"{s}: {t}" for s, t in history]
        lines.append(f"{speaker}: {text}")
        return "\n".join(lines)

    @staticmethod
    def _entail(premise: str, hypotheses: list[str], multi_label: bool = True) -> dict[str, float]:
        out = CLASSIFICATION_MODEL(premise, hypotheses, hypothesis_template="{}", multi_label=multi_label)
        return dict(zip(out["labels"], out["scores"]))

    # ---- main entry point ----------------------------------------------

    def check(self, line: str) -> DetectionResult:
        speaker, text = self._parse(line)
        history_snapshot = list(self.history)
        self.history.append((speaker, text))

        full_premise = self._build_premise(history_snapshot, speaker, text)
        premise = full_premise if self.cfg.context_mode == "full" else f"{speaker}: {text}"

        hyp_map = self._intent_hypotheses(speaker, history_snapshot)
        hyp_texts = list(hyp_map.values())
        raw_scores = self._entail(premise, hyp_texts, multi_label=False)

        # map hypothesis text back to its short label
        text_to_label = {v: k for k, v in hyp_map.items()}
        scores = {text_to_label[h]: s for h, s in raw_scores.items()}

        p_order = scores.get("order", 0.0)
        runner_up = max((v for k, v in scores.items() if k != "order"), default=0.0)
        margin = p_order - runner_up

        silence = p_order >= self.cfg.order_thresh and margin >= self.cfg.safety_margin

        # "order" won the raw competition (beat every other category) but
        # still didn't clear the strict thresh+margin gate -- a genuinely
        # contested call, not a confident rejection. Distinguished from
        # cases where some other category (joke, reaction, ...) clearly won
        # instead, which are NOT ambiguous, just correctly not orders.
        top_label = max(scores, key=scores.get) if scores else None
        ambiguous = (not silence) and (top_label == "order")

        result = DetectionResult(
            silence=silence, stage="full",
            order=p_order,
            joke=scores.get("joke", 0.0),
            other=scores.get("other", 0.0),
            topic=scores.get("topic", 0.0),
            reaction=scores.get("reaction", 0.0),
            none_=scores.get("none", 0.0),
            third_party=scores.get("third_party", 0.0),
            margin=margin,
            ambiguous=ambiguous,
        )
        logger1.debug("speaker=%s %s", speaker, result)
        return result

    def reset(self) -> None:
        self.history.clear()


# =========================================================================
# Behavior layer: mood tagging + stateful silence/resume/rebellion control
# =========================================================================

class MoodClassifier:
    """
    Separate zero-shot pass over the raw utterance, independent of the
    order/joke/reaction competition above -- mood describes tone, and
    shouldn't be entangled with "is this an order".
    """

    LABELS = ("angry", "playful", "affectionate", "sad", "neutral", "anxious", "excited")

    def __init__(self, nli_pipeline=None):
        self.nli = nli_pipeline or CLASSIFICATION_MODEL

    def classify(self, text: str) -> dict[str, float]:
        hyp_to_label = {f"The speaker sounds {label}.": label for label in self.LABELS}
        out = self.nli(text, list(hyp_to_label.keys()), hypothesis_template="{}", multi_label=True)
        return {hyp_to_label[h]: s for h, s in zip(out["labels"], out["scores"])}

    def dominant(self, text: str) -> tuple[str, float]:
        scores = self.classify(text)
        label = max(scores, key=scores.get)
        return label, scores[label]


class AgentState(Enum):
    NORMAL = "normal"
    SILENCED = "silenced"


@dataclass(frozen=True)
class AgentTraits:
    """
    Personality knobs. In a real system these should be read from the
    character's persistent trait/emotional-state store, not re-invented per
    line -- treat this dataclass as the interface, not the source of truth.
    """
    rebelliousness: float = 0.0
    # "angry" is deliberately NOT a default rebel mood. The mood being read
    # here is the HUMAN's tone, not the agent's -- and anger is very often
    # the surface form of real distress, not playful pushback. Defying a
    # playful ask reads as personality; defying an angry one risks reading
    # as "ignored someone who was actually upset". Opt back into "angry"
    # explicitly if your character concept calls for it, rather than by
    # default.
    rebel_moods: tuple[str, ...] = ("playful",)
    # Only allow defiance against soft/ambiguous orders (low margin). A
    # clean, forceful, high-margin order is always obeyed regardless of
    # mood or trait -- this keeps rebellion legible as personality rather
    # than reading as the bot arbitrarily ignoring people.
    rebel_needs_soft_order: bool = True
    soft_order_margin_ceiling: float = 0.30
    # Each consecutive defiance multiplies the NEXT roll's odds by this
    # factor, so a repeated ask becomes steadily more likely to be obeyed
    # instead of the same fixed odds firing forever. Resets to 1.0 the
    # moment the agent complies or the order stream breaks. This is a
    # simplification: it counts consecutive defied orders, not literally
    # "the same request repeated" (no similarity check is done between
    # asks) -- documented here rather than silently assumed.
    repeat_decay: float = 0.5

    # Moods where the agent should lean toward COMPLYING even on a
    # borderline/ambiguous ask, rather than needing full confidence -- the
    # mirror image of rebel_moods. Without this, mood only ever justified
    # ignoring a request, never extra care toward someone who sounds
    # fragile.
    attentive_moods: tuple[str, ...] = ("sad", "anxious")
    attentive_relief: float = 0.08

    # While SILENCED, a message that reads as genuine distress or an
    # important direct question breaks silence for exactly one reply, even
    # if it isn't phrased as "you can talk again" -- the agent then returns
    # to SILENCED afterward. This does not touch order/joke/reaction
    # detection; it is a separate zero-shot check run only while silenced.
    distress_override_thresh: float = 0.55


@dataclass
class OrderSilenceDecision:
    state: AgentState
    obeyed: bool
    rebelled: bool
    reason: str
    detection: DetectionResult
    mood: Optional[tuple[str, float]] = None

    def __str__(self) -> str:
        return (f"state={self.state.value} obeyed={self.obeyed} "
                f"rebelled={self.rebelled} reason={self.reason!r} "
                f"mood={self.mood} | {self.detection}")


class AgentSilenceController:
    """
    Wraps SilenceOrderDecision with a resume-aware state machine and an
    optional trait-driven rebellion layer. All decisions -- order, resume,
    mood -- are zero-shot NLI classifications; only the rebellion ROLL
    itself is arithmetic over those scores, not a new source of pattern
    matching.
    """

    def __init__(
        self,
        decision: Optional[SilenceOrderDecision] = None,
        mood_classifier: Optional[MoodClassifier] = None,
        traits: Optional[AgentTraits] = None,
        rng: Optional[random.Random] = None,
    ):
        self.decision = decision or SilenceOrderDecision()
        self.mood = mood_classifier or MoodClassifier()
        self.traits = traits or AgentTraits()
        self.rng = rng or random.Random()
        self.state = AgentState.NORMAL
        self._defiance_streak = 0  # consecutive defied orders; see repeat_decay

    def _resume_hypotheses(self, speaker: str) -> list[str]:
        bot = self.decision.cfg.bot_name
        return [
            f"{speaker} is telling {bot} it is okay to talk again.",
            f"{speaker} is inviting {bot} to speak now.",
            f"{speaker} is asking {bot} a direct question and wants a reply.",
            f"{speaker} wants to end the silence and hear from {bot} again.",
            f"{speaker} is telling {bot} it can come back and speak again.",
        ]

    def _check_resume(self, speaker: str, text: str) -> float:
        hyps = self._resume_hypotheses(speaker)
        scores = self.decision._entail(f"{speaker}: {text}", hyps, multi_label=True)
        return max(scores.values())

    def _needs_response_hypotheses(self, speaker: str) -> list[str]:
        return [
            f"{speaker} is expressing serious distress or urgently needs a response.",
            f"{speaker} is asking a direct, important question that needs an "
            f"answer right now.",
        ]

    def _check_needs_response(self, speaker: str, text: str) -> float:
        hyps = self._needs_response_hypotheses(speaker)
        scores = self.decision._entail(f"{speaker}: {text}", hyps, multi_label=True)
        return max(scores.values())

    def _maybe_rebel(self, mood_label: str, mood_score: float, detection: DetectionResult) -> bool:
        if self.traits.rebelliousness <= 0.0:
            return False
        if mood_label not in self.traits.rebel_moods:
            return False
        if self.traits.rebel_needs_soft_order and detection.margin > self.traits.soft_order_margin_ceiling:
            return False
        effective_rebelliousness = self.traits.rebelliousness * (
            self.traits.repeat_decay ** self._defiance_streak
        )
        roll = self.rng.random()
        return roll < (effective_rebelliousness * mood_score)

    def handle(self, line: str, resume_thresh: float = 0.5) -> OrderSilenceDecision:
        speaker, text = self.decision._parse(line)
        mood_label, mood_score = self.mood.dominant(text)

        if self.state is AgentState.SILENCED:
            p_resume = self._check_resume(speaker, text)
            if p_resume >= resume_thresh:
                self.state = AgentState.NORMAL
                self.decision.reset()
                self._defiance_streak = 0
                return OrderSilenceDecision(
                    state=self.state, obeyed=False, rebelled=False,
                    reason="resume_detected",
                    detection=DetectionResult(silence=False, stage="resume", order=p_resume),
                    mood=(mood_label, mood_score),
                )

            p_override = self._check_needs_response(speaker, text)
            if p_override >= self.traits.distress_override_thresh:
                # Break silence for exactly this one reply; state stays
                # SILENCED afterward -- the original quiet request still
                # holds once this is addressed.
                self.decision.history.append((speaker, text))
                return OrderSilenceDecision(
                    state=self.state, obeyed=False, rebelled=False,
                    reason="distress_override",
                    detection=DetectionResult(silence=True, stage="distress_override", order=p_override),
                    mood=(mood_label, mood_score),
                )

            self.decision.history.append((speaker, text))
            return OrderSilenceDecision(
                state=self.state, obeyed=True, rebelled=False,
                reason="still_silenced",
                detection=DetectionResult(silence=True, stage="silenced_hold", order=p_resume),
                mood=(mood_label, mood_score),
            )

        detection = self.decision.check(line)

        # Route contested calls through the SAME machinery as a recognized
        # soft order, instead of silently defaulting to "not an order".
        # "order" already won the raw competition here (see
        # DetectionResult.ambiguous) -- it just didn't clear the strict
        # gate. A non-rebellious character (rebelliousness=0) will simply
        # comply, same as attentive_lean's "when genuinely unsure, lean
        # toward listening". A rebellious character gets a real personality
        # roll on exactly these hard-to-classify lines, via the existing
        # margin-gated _maybe_rebel below -- this is deliberately the same
        # gate used for soft orders, not a separate, less-safe one: an
        # ambiguous call always has margin < safety_margin < the rebel
        # ceiling, so it's always eligible for the roll, but a genuinely
        # clean, high-margin order is still never touched by any of this.
        if not detection.silence and detection.ambiguous:
            detection.silence = True
            detection.stage = "ambiguous_order"

        # Attentive lean: if the speaker sounds fragile (sad/anxious) and a
        # borderline ask came close to clearing the order bar without quite
        # doing so, lean toward complying anyway rather than requiring full
        # confidence. Mirrors _maybe_rebel's leniency but in the opposite
        # direction -- mood should be able to argue for MORE care, not only
        # less.
        if not detection.silence and mood_label in self.traits.attentive_moods:
            relief = self.traits.attentive_relief
            close_enough = (
                detection.order >= self.decision.cfg.order_thresh - relief
                and detection.margin >= self.decision.cfg.safety_margin - relief
            )
            if close_enough:
                detection.silence = True
                detection.stage = "attentive_lean"

        if not detection.silence:
            return OrderSilenceDecision(
                state=self.state, obeyed=True, rebelled=False,
                reason="not_an_order", detection=detection,
                mood=(mood_label, mood_score),
            )

        if self._maybe_rebel(mood_label, mood_score, detection):
            self._defiance_streak += 1
            return OrderSilenceDecision(
                state=self.state, obeyed=False, rebelled=True,
                reason="defied_order", detection=detection,
                mood=(mood_label, mood_score),
            )

        self._defiance_streak = 0
        self.state = AgentState.SILENCED
        return OrderSilenceDecision(
            state=self.state, obeyed=True, rebelled=False,
            reason="order_obeyed", detection=detection,
            mood=(mood_label, mood_score),
        )

# =============================================================================
# Affective state engine
# (LAYER 2 - continuous emotion meter.)
# All tuned values below are unchanged. Added in earlier versions:
#   * a lock (thread-safe: the waiting heartbeat thread reads/writes it)
#   * get_emotion(name, now)   - decayed read-only value (no more stale reads)
#   * snapshot(now) / render_prompt_context(now) - optional time-aware reads
#   * apply_external_emotion() - lets the waiting clock write its feelings in
# Layer 1 below never reads or writes any of its state.
# =============================================================================
@dataclass
class AffectConfig:
    bot_name: str = "kanako"               # the ONLY configured identity

    # --- reactivity ---
    stimulus_gain: float = 0.45            # how strongly a message raises an emotion
    counter_gain: float = 0.25             # how strongly a message lowers an existing emotion
    mention_multiplier: float = 1.5        # extra gain when the bot is named in a message
    reply_multiplier: float = 1.25         # extra gain for a message right after the bot spoke
    noise_floor: float = 0.05              # classifier scores below this are ignored
    neutral_score_weight: float = 0.25     # share of the classifier's neutral score applied

    # --- context ---
    context_window: int = 3                # previous lines given to the classifier
    context_blend: float = 0.35            # weight of the with-context reading (0 = off)

    # --- fading ---
    decay_half_life_s: float = 300.0       # base half-life in seconds
    nominal_turn_seconds: float = 15.0     # assumed gap per line when no timestamps are given

    # --- state reading ---
    idle_threshold: float = 0.10           # peak emotion below this = idle
    subtle_threshold: float = 0.25         # below this, weak mixed states read as calm
    direction_shrinkage: float = 0.10      # weak signals report a valence/activation near 0
    learn_from_own_messages: bool = False  # keep False: prevents self-feedback loops
    baseline_overrides: dict[str, float] = field(default_factory=dict)


@dataclass
class AffectSnapshot:
    mood_label: str
    intensity: float
    valence: float
    activation: float
    neutral: float
    dominant_emotions: list[tuple[str, float]]
    all_emotions: dict[str, float]


class AffectiveStateEngine:
    """Maintains the bot's live emotional state from external speakers' messages.

    Idle state: neutral = 1.0, every other emotion = 0.0.
    Messages from the bot itself never influence its state.
    Thread-safe: every public method takes the internal lock.
    """

    def __init__(self, config: AffectConfig | None = None):
        self.config = config or AffectConfig()
        self._lock = threading.RLock()
        self._baseline = {label: 0.0 for label in EMOTION_VALENCE_ACTIVATION}
        self._baseline[NEUTRAL_LABEL] = 1.0
        self._baseline.update(self.config.baseline_overrides)
        self.reset()

    # ---- lifecycle ----------------------------------------------------------
    def reset(self) -> None:
        with self._lock:
            self._state = dict(self._baseline)
            self._last_timestamp: float | None = None
            self._recent_texts: deque[str] = deque(maxlen=self.config.context_window)
            self._last_speaker_was_bot = False
            self.history: list[dict] = []

    # ---- public API ---------------------------------------------------------
    def ingest(self, line: str, timestamp: float | None = None) -> AffectSnapshot:
        """Process one incoming chat line and return the updated snapshot."""
        with self._lock:
            cfg = self.config
            speaker, text = parse_chat_line(line)
            self._apply_decay(timestamp)

            is_own_message = speaker == cfg.bot_name.lower()
            scores: dict[str, float] = {}
            if not is_own_message or cfg.learn_from_own_messages:
                scores = self._classify_with_context(text)
                if scores:
                    mentioned = cfg.bot_name.lower() in text.lower()
                    replying_to_bot = self._last_speaker_was_bot and not is_own_message
                    self._apply_stimulus(scores, mentioned, replying_to_bot)

            self._last_speaker_was_bot = is_own_message
            self._recent_texts.append(text)

            snap = self.snapshot()
            self.history.append({"speaker": speaker, "text": text,
                                 "scores": scores, "snapshot": snap})
            return snap

    def get_emotion(self, name: str, now: float | None = None) -> float:
        """Read-only value of ONE emotion. If `now` (wall-clock, time.time()) is
        given, the value is faded up to that moment WITHOUT changing the stored
        state, so long silences are reflected correctly."""
        with self._lock:
            if name not in self._state:
                return 0.0
            value = self._state[name]
            if now is None or self._last_timestamp is None:
                return value
            elapsed = max(now - self._last_timestamp, 0.0)
            base = self._baseline[name]
            return base + (value - base) * self._retention(name, elapsed)

    def apply_external_emotion(self, name: str, level: float,
                               now: float | None = None) -> None:
        """Raise one emotion to at least `level` (used by the waiting clock so
        her interject feelings linger and fade like any other emotion).
        Bring the state current first, then lift the emotion."""
        if name not in EMOTION_VALENCE_ACTIVATION or name == NEUTRAL_LABEL:
            return
        now = time.time() if now is None else now
        with self._lock:
            self._apply_decay(now)
            self._state[name] = max(self._state[name], max(0.0, min(1.0, level)))

    def snapshot(self, now: float | None = None) -> AffectSnapshot:
        """Current mood. Pass now=time.time() to read the faded (time-correct)
        state during silences; omit it to read the stored state as-is."""
        with self._lock:
            state = self._view(now)
            valence, activation = self._valence_activation(state)
            intensity = self._intensity(state)
            return AffectSnapshot(
                mood_label=self._mood_label(intensity, valence, activation),
                intensity=round(intensity, 3),
                valence=valence,
                activation=activation,
                neutral=round(state[NEUTRAL_LABEL], 3),
                dominant_emotions=[(k, round(v, 3)) for k, v in self._dominant_emotions(state)],
                all_emotions={k: round(v, 3) for k, v in state.items()},
            )

    def render_prompt_context(self, now: float | None = None) -> str:
        """Text block to inject into the bot's system prompt."""
        snap, name = self.snapshot(now), self.config.bot_name
        if snap.mood_label == "neutral / idle":
            return (f"[{name}'s current emotional state] neutral, no strong feeling "
                    f"right now. Respond in your normal baseline tone.")
        strength = ("subtle" if snap.intensity < 0.25
                    else "moderate" if snap.intensity < 0.5 else "strong")
        feelings = ", ".join(f"{k} {v:.2f}" for k, v in snap.dominant_emotions)
        return (f"[{name}'s current emotional state] mood: {snap.mood_label}, "
                f"{strength} (valence {snap.valence:+.2f}, activation {snap.activation:+.2f}). "
                f"Strongest feelings: {feelings}. Let this color your tone, word "
                f"choice and energy naturally. Never mention these numbers.")

    # ---- classification -----------------------------------------------------
    def _run_classifier(self, inputs: list[str]) -> list[dict[str, float]]:
        raw = emotion_classification(inputs, truncation=True, max_length=512)
        return [{item["label"]: float(item["score"]) for item in result} for result in raw]

    def _classify_with_context(self, text: str) -> dict[str, float]:
        """Scores the message alone and, if available, together with recent lines."""
        if emotion_classification is None or not text:
            return {}
        cfg = self.config
        try:
            context = " ".join(self._recent_texts)
            if context and cfg.context_blend > 0:
                alone, with_context = self._run_classifier([text, f"{context} {text}"])
                return {label: (1 - cfg.context_blend) * alone[label]
                        + cfg.context_blend * with_context[label] for label in alone}
            return self._run_classifier([text])[0]
        except Exception as exc:
            logger2.warning("emotion scoring failed: %s", exc)
            return {}

    # ---- dynamics -----------------------------------------------------------
    def _retention(self, label: str, elapsed: float) -> float:
        """Fraction of an emotion's distance-from-baseline that survives `elapsed`
        seconds. High-activation emotions fade faster, calm ones linger."""
        activation = EMOTION_VALENCE_ACTIVATION[label][1]
        half_life = self.config.decay_half_life_s * (1.5 - 0.5 * activation)
        return 0.5 ** (elapsed / half_life)

    def _apply_decay(self, timestamp: float | None) -> None:
        """Each emotion returns to baseline at its own pace: high-activation
        emotions fade faster, calm emotions linger longer."""
        cfg = self.config
        if timestamp is not None and self._last_timestamp is not None:
            elapsed = max(timestamp - self._last_timestamp, 0.0)
        else:
            elapsed = cfg.nominal_turn_seconds
        if timestamp is not None:
            # never move the clock backwards (the waiting thread may have written
            # a slightly newer time than this message's timestamp)
            if self._last_timestamp is None or timestamp > self._last_timestamp:
                self._last_timestamp = timestamp

        for label, value in self._state.items():
            retention = self._retention(label, elapsed)
            base = self._baseline[label]
            self._state[label] = base + (value - base) * retention

    def _apply_stimulus(self, scores: dict[str, float],
                        mentioned: bool, replying_to_bot: bool) -> None:
        """Emotions rise quickly on evidence and fall more gently on absence of
        evidence; the remaining fading is handled by time decay."""
        cfg = self.config
        gain = cfg.stimulus_gain
        if mentioned:
            gain *= cfg.mention_multiplier
        elif replying_to_bot:
            gain *= cfg.reply_multiplier
        gain = min(gain, 1.0)

        for label, raw in scores.items():
            if label == NEUTRAL_LABEL:
                target = raw * cfg.neutral_score_weight
            else:
                target = max(0.0, raw - cfg.noise_floor) / (1.0 - cfg.noise_floor)
            current = self._state[label]
            rate = gain if (label == NEUTRAL_LABEL or target > current) else cfg.counter_gain
            self._state[label] = current + rate * (target - current)

    # ---- state reading ------------------------------------------------------
    def _view(self, now: float | None) -> dict[str, float]:
        """Copy of the state, faded up to `now` if given. Never mutates."""
        if now is None or self._last_timestamp is None:
            return dict(self._state)
        elapsed = max(now - self._last_timestamp, 0.0)
        return {label: self._baseline[label]
                + (value - self._baseline[label]) * self._retention(label, elapsed)
                for label, value in self._state.items()}

    def _intensity(self, state: dict[str, float] | None = None) -> float:
        """Strength of the strongest current emotion (0-1). Emotions co-occur, so
        summing all labels saturates; the peak is a stable measure."""
        state = self._state if state is None else state
        return min(1.0, max(v for k, v in state.items() if k != NEUTRAL_LABEL))

    def _valence_activation(self, state: dict[str, float] | None = None) -> tuple[float, float]:
        """Weighted averages, shrunk toward 0 when total emotion is weak so that
        near-idle states don't report a misleading direction."""
        state = self._state if state is None else state
        weights = {k: v for k, v in state.items() if k != NEUTRAL_LABEL}
        denominator = sum(weights.values()) + self.config.direction_shrinkage
        valence = sum(w * EMOTION_VALENCE_ACTIVATION[k][0] for k, w in weights.items()) / denominator
        activation = sum(w * EMOTION_VALENCE_ACTIVATION[k][1] for k, w in weights.items()) / denominator
        return round(valence, 3), round(activation, 3)

    def _dominant_emotions(self, state: dict[str, float] | None = None,
                           n: int = 4) -> list[tuple[str, float]]:
        state = self._state if state is None else state
        items = [(k, v) for k, v in state.items()
                 if k != NEUTRAL_LABEL and v >= 0.02]
        return sorted(items, key=lambda kv: -kv[1])[:n]

    def _mood_label(self, intensity: float, valence: float, activation: float) -> str:
        cfg = self.config
        if intensity < cfg.idle_threshold:
            return "neutral / idle"
        if intensity < cfg.subtle_threshold and abs(valence) < 0.25:
            return "calm / attentive"
        if valence >= 0:
            return "excited / playful" if activation >= 0.3 else "warm / content"
        return "tense / irritated" if activation >= 0.3 else "low / down"


# =============================================================================
# Runtime wiring
# =============================================================================
try:
    affect_engine # type: ignore
except NameError:
    affect_engine = AffectiveStateEngine(AffectConfig(bot_name="kanako"))

def on_incoming_message(line: str) -> str:
    """Layer 2 only. Do NOT call this together with process_incoming()."""
    now = time.time()
    affect_engine.ingest(line, timestamp=now)
    return affect_engine.render_prompt_context(now)

# =============================================================================
# LAYER 1 (MAIN EMOTION CONTROL TRIGGER) - per-message back-to-back trigger
# -----------------------------------------------------------------------------
# After every line kanako sends, this answers ONE question:
#       "should she speak again right now, and focused on which emotion?"
#
#   * Per-message calculation only. This is NOT the continuous emotion meter:
#     nothing accumulates, nothing decays, no chat history is kept.
#   * Memory is just: the last user line, kanako's last line, and a counter of
#     how many lines she has sent in a row.
#   * Active only while the newest line in the chat is kanako's own, so feed it
#     EVERY line, including hers.
#   * Fully separate from AffectiveStateEngine (layer 2, idle, untouched). The
#     two share only the classifier model and parse_chat_line, never any state.
#
# HOW IT THINKS (the human model - everything comes from the emotion classifier)
#   No keywords, no punctuation, no text patterns: the chat is plain text and the
#   only thing that reads it is the GoEmotions classifier. A person double-texts
#   when a FEELING is still unspent after their message, so:
#       - the feeling in her own line is strong (emotion table x trigger weights,
#         unchanged)  -> this is the ONLY thing that can create a trigger
#       - that feeling is high-energy (excitement, anger, fear, surprise...)
#         -> it spills over into another line; low-energy feelings (sadness,
#         disappointment) are quieter and less likely to
#       - the user's line was emotionally charged -> one reply isn't enough
#       - her main feeling is curiosity -> she just asked / wondered about
#         something and waits for the answer
#   Those situation cues only SCALE her own emotion. They can never fire a line
#   that carries no feeling, so a back-to-back always has a real emotion behind
#   it, and there is no random "noise" firing on neutral lines.
# =============================================================================


@dataclass
class BackToBackConfig:
    bot_name: str = "kanako"

    # --- reading the exchange ---
    context_blend: float = 0.35            # weight of "her line read together with the user's line" (0 = off)
    context_max_chars: int = 300           # user line is trimmed to its last N chars inside that joint reading
    noise_floor: float = 0.05              # classifier scores below this are ignored

    # --- the bar ---
    trigger_threshold: float = 0.40        # pressure at which a back-to-back becomes a coin flip
    chain_threshold_step: float = 0.15     # every extra line in a row raises that bar by this much
    max_consecutive_lines: int = 3         # hard cap: kanako lines in a row (reply + back-to-backs)
    max_neutral_score: float = 0.50        # if her line reads this neutral (blended score), never fire
    trigger_weights: dict[str, float] = field(
        default_factory=lambda: dict(BACK_TO_BACK_TRIGGER_WEIGHTS))

    # --- how likely, given the pressure (smooth, but with a real floor: no noise fires) ---
    fire_floor_ratio: float = 0.50         # pressure below (this x bar) -> probability is exactly 0
    fire_ceiling_ratio: float = 1.50       # pressure above (this x bar) -> probability = max_probability
    max_probability: float = 0.90          # even a huge spike doesn't ALWAYS double-text
    chain_decay: float = 0.75              # each extra line in a row multiplies that ceiling (a 3rd text is rarer than a 2nd)

    # --- situation cues (multipliers on her own emotional pressure; 1.0 = no effect) ---
    # all of them are read from classifier scores + the (read-only) emotion table
    user_charge_gain: float = 0.35         # +up to 35% when the user's line was itself emotionally charged
    energy_gain: float = 0.25              # +-25% from the energy (activation) of her feeling: calm -> less, high-energy -> more
    energy_shrinkage: float = 0.15         # weak feelings report an energy near 0 (same idea as layer 2's direction_shrinkage)
    waiting_emotions: tuple[str, ...] = ("curiosity",)   # if one of these is her main feeling she is waiting for an answer
    waiting_damping: float = 0.80
    max_cue_factor: float = 2.0            # all cues together can never scale pressure by more than this

    # --- timing (how long a person would pause before the extra line) ---
    delay_base_s: float = 2.0
    delay_jitter: float = 0.40             # +-40% random variation
    energetic_delay_factor: float = 0.85   # high-energy feelings are followed up a little faster
    subdued_delay_factor: float = 1.20     # low-energy feelings take a longer beat

    seed: int | None = None                # set for reproducible behaviour in tests

    def __post_init__(self) -> None:
        bad = [k for k in self.trigger_weights
               if k not in EMOTION_VALENCE_ACTIVATION or k == NEUTRAL_LABEL]
        if bad:
            raise ValueError(f"trigger_weights has unknown / non-emotion labels: {bad}")
        if not 0.0 <= self.noise_floor < 1.0:
            raise ValueError("noise_floor must be in [0, 1)")
        if not 0.0 <= self.context_blend <= 1.0:
            raise ValueError("context_blend must be in [0, 1]")
        if self.context_max_chars < 1:
            raise ValueError("context_max_chars must be >= 1")
        if not 0.0 <= self.max_neutral_score <= 1.0:
            raise ValueError("max_neutral_score must be in [0, 1]")
        if not 0.0 <= self.fire_floor_ratio < self.fire_ceiling_ratio:
            raise ValueError("need 0 <= fire_floor_ratio < fire_ceiling_ratio")
        if not 0.0 <= self.max_probability <= 1.0:
            raise ValueError("max_probability must be in [0, 1]")
        if not 0.0 < self.chain_decay <= 1.0:
            raise ValueError("chain_decay must be in (0, 1]")
        if self.user_charge_gain < 0.0:
            raise ValueError("user_charge_gain must be >= 0")
        if not 0.0 <= self.energy_gain <= 1.0:
            raise ValueError("energy_gain must be in [0, 1]")
        if self.energy_shrinkage < 0.0:
            raise ValueError("energy_shrinkage must be >= 0")
        if not 0.0 < self.waiting_damping <= 1.0:
            raise ValueError("waiting_damping must be in (0, 1]")
        bad_wait = [k for k in self.waiting_emotions
                    if k not in EMOTION_VALENCE_ACTIVATION or k == NEUTRAL_LABEL]
        if bad_wait:
            raise ValueError(f"waiting_emotions has unknown / non-emotion labels: {bad_wait}")
        if self.max_cue_factor <= 0.0:
            raise ValueError("max_cue_factor must be > 0")
        if not 0.0 <= self.delay_jitter < 1.0:
            raise ValueError("delay_jitter must be in [0, 1)")
        if self.delay_base_s < 0.0:
            raise ValueError("delay_base_s must be >= 0")


@dataclass
class TriggerDecision:
    trigger: bool                          # True = kanako should speak again right now
    focus_emotion: str | None              # emotion to focus on (None unless trigger is True)
    strength: float = 0.0                  # pressure: best emotional impact x situation cues (0-1)
    threshold: float = 0.0                 # bar that pressure is measured against
    chain_length: int = 0                  # kanako lines in a row so far (0 = newest line is a user's)
    reason: str = ""                       # short explanation, handy for logs
    top_impacts: list[tuple[str, float]] = field(default_factory=list)   # best 3 candidates, before cues (debug)
    model_top: list[tuple[str, float]] = field(default_factory=list)     # classifier's own top 3 raw labels, incl. neutral (debug)
    probability: float = 0.0               # chance that was rolled for this decision (debug)
    delay_s: float = 0.0                   # suggested pause before the extra line (0 unless trigger is True)
    cues: list[str] = field(default_factory=list)                        # situation cues that applied (debug)


class BackToBackTriggerEngine:
    """Layer 1. Feed it every chat line in order (users' lines and kanako's own);
    it returns one TriggerDecision per line. ingest() is thread-safe.

    When kanako's line is the newest one:
      1. Read her line alone, together with the user's last line (blended by
         context_blend), and the user's line alone. Ignore scores below noise_floor.
      2. impact[emotion] = trigger_weights[emotion] * score[emotion]; the best one
         is the focus emotion and the base pressure.
      3. Scale the pressure by the situation cues (see header), all derived from
         classifier scores. Cues only scale, so a line with no feeling stays at zero.
      4. Never fire if her line reads neutral (neutral >= max_neutral_score) or
         she already sent max_consecutive_lines lines in a row.
      5. Pressure -> probability: exactly 0 below fire_floor_ratio x bar, rising
         smoothly to max_probability at fire_ceiling_ratio x bar (roughly a coin
         flip at the bar). Roll for it. The bar is trigger_threshold +
         chain_threshold_step per extra line in a row, and the ceiling shrinks by
         chain_decay per extra line, so long bursts are rare.
      6. If it fires, suggest a pause before the extra line (energetic emotions
         quicker, calm ones slower, later lines slower).
    A new user line resets the counter and makes the layer inactive again.
    """

    def __init__(self, config: BackToBackConfig | None = None):
        self.config = config or BackToBackConfig()
        self._lock = threading.RLock()
        self._rng = random.Random(self.config.seed)
        self._warned_no_classifier = False
        self.reset()

    # ---- lifecycle ----------------------------------------------------------
    def reset(self) -> None:
        with self._lock:
            self.last_user_speaker: str | None = None
            self.last_user_text: str | None = None
            self.last_bot_text: str | None = None
            self.chain_length = 0           # kanako lines in a row; 0 = newest line is a user's
            self.last_decision = TriggerDecision(False, None, reason="no lines yet")

    @property
    def active(self) -> bool:
        """True while kanako's line is the newest one and a user line is stored to pair it with."""
        return self.chain_length > 0 and self.last_user_text is not None

    # ---- public API ---------------------------------------------------------
    def ingest(self, line: str) -> TriggerDecision:
        """Process one chat line and return the back-to-back decision for it."""
        speaker, text = parse_chat_line(line)
        with self._lock:
            if not text:
                return self._hold("blank line ignored")
            if speaker == self.config.bot_name.lower():
                decision = self._on_bot_line(text)
            else:
                decision = self._on_user_line(speaker, text)
            self.last_decision = decision
            logger2.debug("layer1 %s: trigger=%s focus=%s pressure=%.2f bar=%.2f p=%.2f chain=%d cues=%s (%s)",
                         speaker, decision.trigger, decision.focus_emotion,
                         decision.strength, decision.threshold, decision.probability,
                         decision.chain_length, decision.cues, decision.reason)
            return decision

    # ---- line handling ------------------------------------------------------
    def _on_user_line(self, speaker: str, text: str) -> TriggerDecision:
        """A user spoke: remember the line, end any streak, go inactive."""
        self.last_user_speaker, self.last_user_text = speaker, text
        self.last_bot_text = None
        self.chain_length = 0
        return self._hold("inactive: newest line is from a user")

    def _on_bot_line(self, text: str) -> TriggerDecision:
        """Kanako spoke: decide whether she goes again."""
        cfg = self.config
        self.last_bot_text = text
        self.chain_length += 1
        chain = self.chain_length
        threshold = cfg.trigger_threshold + cfg.chain_threshold_step * (chain - 1)

        if self.last_user_text is None:
            return self._hold("inactive: no user line stored to pair with", threshold)
        if chain >= cfg.max_consecutive_lines:
            return self._hold(f"chain limit: {chain} kanako lines in a row", threshold)
        scores, user_scores = self._read_exchange()
        if not scores:
            return self._hold("no emotion reading (classifier unavailable or failed)", threshold)
        impacts = self._impacts(scores)
        if not impacts:
            return self._hold("no trigger emotions configured", threshold)

        ranked = sorted(impacts.items(), key=lambda kv: -kv[1])
        focus, raw_strength = ranked[0]
        user_charge = max(self._impacts(user_scores).values(), default=0.0) if user_scores else 0.0
        factor, cues = self._read_cues(scores, user_charge)
        pressure = min(1.0, raw_strength * factor)
        neutral = scores.get(NEUTRAL_LABEL, 0.0)
        top_impacts = [(k, round(v, 3)) for k, v in ranked[:3]]
        model_top = [(k, round(v, 3)) for k, v in
                     sorted(scores.items(), key=lambda kv: -kv[1])[:3]]

        if neutral >= cfg.max_neutral_score:
            return TriggerDecision(
                False, None, strength=round(pressure, 3), threshold=round(threshold, 3),
                chain_length=chain,
                reason=f"line reads neutral ({neutral:.2f} >= {cfg.max_neutral_score:.2f})",
                top_impacts=top_impacts, model_top=model_top, cues=cues)

        probability = self._fire_probability(pressure, threshold, chain)
        if probability <= 0.0:
            fire = False
            reason = "not enough feeling behind this line"
        else:
            fire = self._rng.random() < probability
            reason = f"rolled {'yes' if fire else 'no'} (p={probability:.2f})"
        return TriggerDecision(
            trigger=fire,
            focus_emotion=focus if fire else None,
            strength=round(pressure, 3),
            threshold=round(threshold, 3),
            chain_length=chain,
            reason=reason,
            top_impacts=top_impacts,
            model_top=model_top,
            probability=round(probability, 3),
            delay_s=self._suggest_delay(focus, chain, cues) if fire else 0.0,
            cues=cues,
        )

    def _hold(self, reason: str, threshold: float = 0.0) -> TriggerDecision:
        """A decision that does not fire."""
        return TriggerDecision(False, None, threshold=round(threshold, 3),
                               chain_length=self.chain_length, reason=reason)

    # ---- human feel ---------------------------------------------------------
    def _read_cues(self, scores: dict[str, float], user_charge: float) -> tuple[float, list[str]]:
        """Situation cues -> (multiplier on her emotional pressure, tags for the log).
        Everything here is read from classifier scores and the emotion table (read-only).
        These scale her own feeling; they never create one."""
        cfg = self.config
        factor, tags = 1.0, []

        # noise-floored level of every real emotion in her line (neutral excluded)
        levels = {label: max(0.0, score - cfg.noise_floor) / (1.0 - cfg.noise_floor)
                  for label, score in scores.items()
                  if label != NEUTRAL_LABEL and label in EMOTION_VALENCE_ACTIVATION}

        if levels:
            # energy of her feeling: weighted activation, shrunk toward 0 when the feeling is weak
            total = sum(levels.values())
            energy = sum(v * EMOTION_VALENCE_ACTIVATION[k][1]
                         for k, v in levels.items()) / (total + cfg.energy_shrinkage)
            factor *= 1.0 + cfg.energy_gain * energy
            if energy >= 0.20:
                tags.append("energetic")
            elif energy <= -0.20:
                tags.append("subdued")

            # her main feeling is a wondering one -> she is waiting for an answer
            main = max(levels, key=levels.get)
            if levels[main] > 0.0 and main in cfg.waiting_emotions:
                factor *= cfg.waiting_damping
                tags.append("waiting")

        # the user's line was itself charged: one reply may not cover it
        charge_boost = cfg.user_charge_gain * max(0.0, min(1.0, user_charge))
        if charge_boost > 0.0:
            factor *= 1.0 + charge_boost
            if charge_boost >= 0.05:
                tags.append("charged-user")

        return max(1.0 / cfg.max_cue_factor, min(cfg.max_cue_factor, factor)), tags

    def _fire_probability(self, pressure: float, threshold: float, chain: int = 1) -> float:
        """Zero below the floor, smooth rise above it, capped at max_probability
        (shrunk per extra line in a row). A smooth S-curve with a hard floor: no long
        tail of 1-2% 'noise' fires on lines that carry no real feeling."""
        cfg = self.config
        lo, hi = cfg.fire_floor_ratio * threshold, cfg.fire_ceiling_ratio * threshold
        if pressure <= lo:
            return 0.0
        x = min(1.0, (pressure - lo) / (hi - lo))
        ceiling = cfg.max_probability * cfg.chain_decay ** (chain - 1)
        return ceiling * x * x * (3.0 - 2.0 * x)

    def _suggest_delay(self, focus: str, chain: int, cues: list[str]) -> float:
        """Pause before the extra line, in seconds. Reads the focus emotion's activation
        (read-only): energetic emotions follow up fast, calm ones take a beat."""
        cfg = self.config
        activation = EMOTION_VALENCE_ACTIVATION[focus][1]
        pace = 1.5 - 0.5 * activation               # 1.0 (very energetic) ... 2.0 (very calm)
        chain_slowdown = 1.0 + 0.5 * (chain - 1)    # a 3rd message in a row comes slower
        cue_pace = 1.0
        if "energetic" in cues:
            cue_pace *= cfg.energetic_delay_factor
        if "subdued" in cues:
            cue_pace *= cfg.subdued_delay_factor
        jitter = self._rng.uniform(1.0 - cfg.delay_jitter, 1.0 + cfg.delay_jitter)
        return round(cfg.delay_base_s * pace * chain_slowdown * cue_pace * jitter, 2)

    # ---- scoring ------------------------------------------------------------
    @staticmethod
    def _classify(texts: list[str]) -> list[dict[str, float]]:
        raw = emotion_classification(texts, truncation=True, max_length=512)
        return [{item["label"]: float(item["score"]) for item in result} for result in raw]

    def _read_exchange(self) -> tuple[dict[str, float], dict[str, float]]:
        """(scores for kanako's newest line, scores for the user's last line alone).
        Her line is read alone and together with the user's line, blended. Either
        dict is {} when no reading is possible."""
        if emotion_classification is None:
            if not self._warned_no_classifier:
                logger2.warning("layer 1: emotion classifier unavailable - the "
                               "back-to-back trigger will never fire")
                self._warned_no_classifier = True
            return {}, {}
        cfg = self.config
        bot_text = self.last_bot_text or ""
        try:
            if self.last_user_text:
                user_ctx = self.last_user_text[-cfg.context_max_chars:]
                alone, paired, user_alone = self._classify(
                    [bot_text, f"{user_ctx} {bot_text}", user_ctx])
                if cfg.context_blend > 0:
                    blended = {label: (1 - cfg.context_blend) * alone[label]
                               + cfg.context_blend * paired[label] for label in alone}
                else:
                    blended = alone
                return blended, user_alone
            return self._classify([bot_text])[0], {}
        except Exception as exc:
            logger2.warning("layer 1: emotion scoring failed: %s", exc)
            return {}, {}

    def _impacts(self, scores: dict[str, float]) -> dict[str, float]:
        """trigger weight x (noise-floored) score, for every trigger emotion."""
        cfg = self.config
        impacts: dict[str, float] = {}
        for label, weight in cfg.trigger_weights.items():
            level = max(0.0, scores.get(label, 0.0) - cfg.noise_floor) / (1.0 - cfg.noise_floor)
            impacts[label] = weight * level
        return impacts


# =============================================================================
# Layer 1 runtime wiring (separate from the layer 2 wiring above)
# =============================================================================
try:
    trigger_engine # type: ignore
except NameError:
    trigger_engine = BackToBackTriggerEngine(BackToBackConfig(bot_name="kanako"))


def on_chat_line_layer1(line: str) -> tuple[bool, str | None]:
    """Layer 1 only. Do NOT call this together with process_incoming()."""
    decision = trigger_engine.ingest(line)
    return decision.trigger, decision.focus_emotion

# =============================================================================
# WAITING INTERVAL — the "she's waiting for you" clock
# -----------------------------------------------------------------------------
# Independent of Layer 2. It tracks what she FEELS while the user is silent by
# escalating through a fixed ladder of stages. Whether she actually SPEAKS is
# decided by the IntervalSilenceDecisionEngine further below, through the `gate` hook.
# The WaitingInterval class itself only READS Layer 2 (to seed a stage's meter,
# using the time-faded get_emotion). The callbacks (on_waiting_fire /
# on_waiting_lapse, below) WRITE the stage's feeling back into Layer 2, so it
# lingers and fades after the user returns.
#
#   * Only runs while the last chat line was the bot's own; a user line resets all.
#   * Every tick_s: current stage meter += tick_step (x stage pace x jitter).
#   * Every tick the gate is asked "speak now?". Yes -> one interject, advance.
#   * Meter full and the gate said no -> the stage LAPSES SILENTLY: she felt it,
#     said nothing, and the ladder moves on to the next feeling.
#   * With no gate (gate=None) the old behaviour applies: full meter always fires.
#   * Terminal stage (coldness) fires or lapses ONCE, then silence until a user line.
#   * Optional pace_provider speeds the ladder up / slows it down per person.
#   * A near-miss (she almost spoke) is reported through on_hesitate.
# =============================================================================


@dataclass
class WaitingConfig:
    bot_name: str = "kanako"

    # ---- the clock (natural values: ~100 s per stage, ~10 min for the ladder) ----
    tick_s: float = 10.0       # seconds between ticks
    tick_step: float = 0.1     # meter growth per tick
    step_jitter: float = 0.30  # +-30% random variation on each tick's growth (no metronome)
    stage_pace: dict[str, float] = field(default_factory=dict)
    #   optional per-stage speed multiplier, e.g. {"sadness": 0.7, "anger": 1.3}
    #   (<1 = that stage takes longer to build, >1 = quicker). Missing = 1.0.
    seed: int | None = None    # set for reproducible jitter in tests

    # ---- the escalation ladder (plain (name, emotion, meaning) tuples also work) ----
    ladder: tuple[Stage, ...] = DEFAULT_LADDER
    terminal_stage: str = "coldness"        # must be the LAST stage in the ladder

    # ---- layer-2 seeding ----
    seed_from_layer2: bool = True
    seed_ceiling: float = 1.0               # max share of Layer 2's value we inherit
    fire_when_seeded_full: bool = True      # full-on-engage fires next tick, not after tick_s

    # ---- cooldowns ----
    min_seconds_between_interjects: float = 0.0   # 0 = as fast as ticks allow

    def __post_init__(self) -> None:
        self.ladder = tuple(s if isinstance(s, Stage) else Stage(*s) for s in self.ladder)
        if self.tick_s <= 0:
            raise ValueError("tick_s must be > 0")
        if not (0.0 < self.tick_step <= 1.0):
            raise ValueError("tick_step must be in (0, 1]")
        if not (0.0 <= self.step_jitter < 1.0):
            raise ValueError("step_jitter must be in [0, 1)")
        if not (0.0 <= self.seed_ceiling <= 1.0):
            raise ValueError("seed_ceiling must be in [0, 1]")
        if self.min_seconds_between_interjects < 0:
            raise ValueError("min_seconds_between_interjects must be >= 0")
        if not self.ladder:
            raise ValueError("ladder must contain at least one stage")
        names = [s.name for s in self.ladder]
        if len(set(names)) != len(names):
            raise ValueError(f"duplicate stage names in ladder: {names}")
        if self.ladder[-1].name != self.terminal_stage:
            raise ValueError(
                f"terminal_stage {self.terminal_stage!r} must be the LAST ladder stage "
                f"(got {self.ladder[-1].name!r})"
            )
        bad_pace = [k for k, v in self.stage_pace.items() if k not in names or v <= 0]
        if bad_pace:
            raise ValueError(f"stage_pace has unknown stages or non-positive values: {bad_pace}")


@dataclass(frozen=True)
class WaitingTick:
    """One heartbeat result."""
    stage: Optional[str]            # current stage (None if clock is disarmed)
    meter: float                    # current stage's meter (0-1)
    fired: bool                     # did this tick trigger an interject?
    fired_stage: Optional[str]      # which stage fired (if any)
    reason: str                     # short human-readable status
    is_terminal: bool               # True once the terminal stage has fired (silent)
    lapsed_stage: Optional[str] = None   # stage that ended WITHOUT her speaking (felt, not said)
    hesitated: bool = False              # she almost spoke this tick (started to, then held back)


class WaitingInterval:
    """Slow "she is waiting for you" clock. Independent from Layer 2.

    Wire it:
      1. .on_user_line()  every time the user sends a line
      2. .on_bot_line()   every time kanako sends a line (arms the clock)
      3. .tick()          from your own loop, or call .start() for a built-in
                          daemon heartbeat thread
      4. .snapshot() / .current_stage / .current_meter for prompt rendering
    Pass on_fire=callable(WaitingTick) to be notified when an interject fires,
    on_lapse=callable(WaitingTick) when a stage ends silently, and
    gate=callable(ctx_dict) -> bool to decide, each tick, whether she speaks.
    Optional: on_hesitate / hesitation_check (near-miss events) and
    pace_provider() -> float (>1 speeds the ladder up, <1 slows it down).
    """

    def __init__(
        self,
        affect_engine=None,
        config: WaitingConfig | None = None,
        *,
        clock: Callable[[], float] = time.monotonic,
        on_fire: Callable[[WaitingTick], None] | None = None,
        on_lapse: Callable[[WaitingTick], None] | None = None,
        on_hesitate: Callable[[WaitingTick], None] | None = None,
        gate: Callable[[dict], bool] | None = None,
        hesitation_check: Callable[[], bool] | None = None,
        pace_provider: Callable[[], float] | None = None,
    ):
        self.config = config or WaitingConfig()
        self.affect = affect_engine
        self.on_fire = on_fire
        self.on_lapse = on_lapse
        self.on_hesitate = on_hesitate
        self.gate = gate
        self.hesitation_check = hesitation_check
        self.pace_provider = pace_provider
        self._clock = clock
        self._rng = random.Random(self.config.seed)
        self._lock = threading.RLock()
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self.reset()

    # ---- lifecycle ----------------------------------------------------------
    def reset(self) -> None:
        """Back to stage 1, meter 0, clock disarmed."""
        with self._lock:
            self._stage_index = 0
            self._meter = 0.0
            self._stage_seeded = False
            self._armed = False
            self._terminal_fired = False
            self._last_tick_ts: float | None = None
            self._last_interject_ts: float = -math.inf

    # ---- external hooks -----------------------------------------------------
    def on_user_line(self) -> None:
        """User spoke -> hard reset and disarm."""
        self.reset()

    def on_bot_line(self) -> None:
        """Bot spoke -> arm the clock. A double-text keeps the ladder where it is."""
        with self._lock:
            self._armed = True
            self._last_tick_ts = None   # next tick re-primes the timer

    # ---- optional built-in heartbeat ---------------------------------------
    def start(self, poll_s: float = 1.0) -> None:
        """Daemon thread calling tick() every poll_s. tick() does its own tick_s
        bookkeeping, so polling faster than tick_s is safe."""
        with self._lock:
            if self._thread and self._thread.is_alive():
                return
            self._stop.clear()
            self._thread = threading.Thread(
                target=self._run, args=(poll_s,), daemon=True, name="waiting-interval"
            )
            self._thread.start()

    def stop(self, timeout: float | None = 2.0) -> None:
        self._stop.set()
        thread = self._thread
        if thread and thread is not threading.current_thread():
            thread.join(timeout)
        self._thread = None

    def _run(self, poll_s: float) -> None:
        while not self._stop.wait(poll_s):
            try:
                self.tick()
            except Exception:
                log.exception("WaitingInterval tick failed")

    # ---- read-only accessors ------------------------------------------------
    @property
    def current_stage(self) -> Optional[str]:
        with self._lock:
            if not self._armed:
                return None
            return self.config.ladder[self._stage_index].name

    @property
    def current_meter(self) -> float:
        with self._lock:
            return self._meter

    @property
    def is_terminal(self) -> bool:
        with self._lock:
            return self._terminal_fired

    def seconds_until_next_tick(self, now: float | None = None) -> Optional[float]:
        """Seconds until the meter next grows; None if paused, silent or not primed."""
        now = self._clock() if now is None else now
        with self._lock:
            if not self._armed or self._terminal_fired or self._last_tick_ts is None:
                return None
            return max(0.0, self.config.tick_s - (now - self._last_tick_ts))

    def snapshot(self) -> dict:
        """Plain dict for the prompt renderer / interject layer."""
        with self._lock:
            total = len(self.config.ladder)
            stage = self.current_stage
            if stage is None:
                return {"stage": None, "meter": 0.0, "terminal": self._terminal_fired,
                        "meaning": None, "stage_index": None, "stages_total": total,
                        "overall_progress": 0.0}
            overall = 1.0 if self._terminal_fired else (self._stage_index + self._meter) / total
            return {
                "stage": stage,
                "meter": round(self._meter, 3),
                "terminal": self._terminal_fired,
                "meaning": self.config.ladder[self._stage_index].meaning,
                "stage_index": self._stage_index,
                "stages_total": total,
                "overall_progress": round(overall, 3),
            }

    # ---- the tick -----------------------------------------------------------
    def tick(self, now: float | None = None) -> WaitingTick:
        """Advance the clock. Safe to call as often as you like.

        disarmed -> no-op | first call after arming -> primes the timer |
        < tick_s since last step -> no-op | else meter grows per whole tick_s
        elapsed (late calls catch up, but at most one interject fires) and the
        gate is asked once whether she speaks.
        on_fire / on_lapse run AFTER the lock is released.
        """
        now = self._clock() if now is None else now
        with self._lock:
            result = self._tick_locked(now)
        if result.fired and self.on_fire is not None:
            try:
                self.on_fire(result)
            except Exception:
                log.exception("on_fire callback failed")
        if result.lapsed_stage is not None and self.on_lapse is not None:
            try:
                self.on_lapse(result)
            except Exception:
                log.exception("on_lapse callback failed")
        if result.hesitated and self.on_hesitate is not None:
            try:
                self.on_hesitate(result)
            except Exception:
                log.exception("on_hesitate callback failed")
        return result

    def _tick_locked(self, now: float) -> WaitingTick:
        cfg = self.config

        if not self._armed:
            return WaitingTick(None, 0.0, False, None,
                               "clock disarmed (last speaker was not the bot)", False)
        if self._terminal_fired:
            return WaitingTick(cfg.terminal_stage, 1.0, False, None,
                               "terminal state reached — silent", True)

        self._engage_stage()

        primed = self._last_tick_ts is None
        if primed:
            self._last_tick_ts = now

        steps = max(0, int((now - self._last_tick_ts) // cfg.tick_s))
        full = self._meter >= 1.0 - _EPS

        if steps == 0 and not (full and cfg.fire_when_seeded_full):
            return self._result(False, None,
                                "clock armed, timer primed" if primed else "waiting for next tick")

        # advance by whole ticks only, so the remainder isn't lost (no drift)
        self._last_tick_ts += steps * cfg.tick_s
        pace = cfg.stage_pace.get(cfg.ladder[self._stage_index].name, 1.0) * self._external_pace()
        for _ in range(steps):
            if full:
                break
            jitter = (self._rng.uniform(1.0 - cfg.step_jitter, 1.0 + cfg.step_jitter)
                      if cfg.step_jitter > 0 else 1.0)
            self._meter = min(1.0, self._meter + cfg.tick_step * pace * jitter)
            full = self._meter >= 1.0 - _EPS
        if full:
            self._meter = 1.0

        # ---- legacy behaviour (no gate): a full meter always speaks -----------
        if self.gate is None:
            if not full:
                return self._result(False, None, "tick")
            if now - self._last_interject_ts < cfg.min_seconds_between_interjects:
                return self._result(False, None, "full, holding for interject cooldown")
            return self._fire(now)

        # ---- gated behaviour: feeling builds, the gate decides to speak -------
        speak = self._ask_gate(max(1, steps), full)
        if speak:
            if now - self._last_interject_ts < cfg.min_seconds_between_interjects:
                return self._result(False, None, "wants to speak, holding for interject cooldown")
            return self._fire(now)
        hesitated = self._check_hesitation()
        if full:
            return self._lapse(now, hesitated)   # felt it, chose not to say it
        return self._result(False, None, "tick", hesitated=hesitated)

    # ---- internals ----------------------------------------------------------
    def _external_pace(self) -> float:
        """Speed multiplier from the outside (e.g. the person's usual reply pace)."""
        if self.pace_provider is None:
            return 1.0
        try:
            value = float(self.pace_provider())
        except Exception:
            return 1.0
        return value if value > 0 and not math.isnan(value) else 1.0

    def _check_hesitation(self) -> bool:
        if self.hesitation_check is None:
            return False
        try:
            return bool(self.hesitation_check())
        except Exception:
            return False

    def _ask_gate(self, steps: int, full: bool) -> bool:
        stage = self.config.ladder[self._stage_index]
        ctx = {
            "stage": stage.name,
            "emotion": stage.emotion,
            "stage_index": self._stage_index,
            "stages_total": len(self.config.ladder),
            "meter": self._meter,
            "full": full,
            "terminal": self._at_terminal_stage(),
            "steps": steps,
            "tick_s": self.config.tick_s,
        }
        try:
            return bool(self.gate(ctx))
        except Exception:
            log.exception("gate failed - falling back to 'full meter speaks'")
            return full

    def _advance_stage(self, now: float) -> None:
        self._last_tick_ts = now    # next stage gets a clean tick_s; stale backlog dropped
        if self._at_terminal_stage():
            self._terminal_fired = True
            self._meter = 1.0
        else:
            self._stage_index += 1
            self._meter = 0.0
            self._stage_seeded = False
            self._engage_stage()

    def _fire(self, now: float) -> WaitingTick:
        fired_stage = self.config.ladder[self._stage_index].name
        self._last_interject_ts = now
        self._advance_stage(now)
        return self._result(True, fired_stage, "fired interject")

    def _lapse(self, now: float, hesitated: bool = False) -> WaitingTick:
        """The stage ran its course without her speaking."""
        lapsed = self.config.ladder[self._stage_index].name
        self._advance_stage(now)
        return self._result(False, None, "stage lapsed silently",
                            lapsed_stage=lapsed, hesitated=hesitated)

    def _result(self, fired: bool, fired_stage: Optional[str], reason: str,
                lapsed_stage: Optional[str] = None, hesitated: bool = False) -> WaitingTick:
        return WaitingTick(
            stage=self.current_stage,
            meter=round(self._meter, 3),
            fired=fired,
            fired_stage=fired_stage,
            reason=reason,
            is_terminal=self._terminal_fired,
            lapsed_stage=lapsed_stage,
            hesitated=hesitated,
        )

    def _at_terminal_stage(self) -> bool:
        return self._stage_index >= len(self.config.ladder) - 1

    def _engage_stage(self) -> None:
        """Seed the current stage's meter from Layer 2, exactly once per stage."""
        if self._stage_seeded:
            return
        self._stage_seeded = True
        cfg = self.config
        if not cfg.seed_from_layer2 or self.affect is None:
            return
        value = self._read_emotion(cfg.ladder[self._stage_index].emotion)
        self._meter = max(self._meter, min(cfg.seed_ceiling, value))

    def _read_emotion(self, name: str) -> float:
        """Read-only peek at Layer 2, faded up to NOW (Layer 2 uses wall-clock
        time.time(), unlike this class's monotonic clock). Any failure or junk
        value counts as 0."""
        try:
            getter = getattr(self.affect, "get_emotion", None)
            raw = getter(name, time.time()) if callable(getter) else self.affect._state.get(name, 0.0)
            value = float(raw)
        except Exception:
            return 0.0
        if math.isnan(value):
            return 0.0
        return min(1.0, max(0.0, value))


# =============================================================================
# SILENCE DECISION ENGINE — "should she actually react to this silence?"
# -----------------------------------------------------------------------------
# The waiting ladder only says what she FEELS. This engine decides whether she
# ACTS on it. It is asked once per ladder tick (through WaitingInterval.gate)
# and answers speak / stay quiet. It never writes to Layer 1 or Layer 2; it only
# reads Layer 2's faded mood.
#
#   urge      = how late is this silence FOR THIS PERSON          (lateness)
#             x how much she wanted a reply to her last line       (need)
#             x how energetic her mood / current feeling is        (mood drive)
#             x how far the current stage feeling has built        (feeling)
#             x her clinginess, how well she knows them, whether
#               they started typing and stopped
#   restraint = fatigue from unanswered messages
#             + pride (grows in the sad / angry / cold stages)
#             + refractory beat right after she already spoke
#             + a busy chat (the person may just be reading)
#             + a grudge if this person left her hanging before
#   pressure  = max(0, urge - restraint)
#   chance    = 0 below a floor, smooth S-curve above it, rolled every tick
#   bar       = grows with every unanswered interject (she needs more buildup
#               each time, so she doesn't nag at a steady rate)
#
# "Alive" details on top of that core:
#   * every wait has its own temperament (some waits she is patient, some not)
#   * the ladder runs at the pace of the person (slow repliers -> slower feelings)
#   * near-misses: she sometimes STARTS to speak and holds back (hesitation event)
#   * if the platform reports typing, she never talks over them
#   * she remembers people: strangers get less poking, being ignored leaves a
#     grudge that fades slowly, quick replies melt it (optional JSON memory)
#   * when they come back: relief if the wait was short, a sulky pause if long
#   * every interject carries an intent (reminder / playful_poke / check_in /
#     worried_check_in / hurt / irritated / cold) for the line writer
#
# If she stays quiet and the stage's meter fills, the stage lapses silently: she
# feels it (written into Layer 2 by on_waiting_lapse) but says nothing.
# =============================================================================


@dataclass
class SilenceConfig:
    bot_name: str = "kanako"

    # --- what is "normal" for this person ---
    default_expected_reply_s: float = 45.0   # assumed reply time until we have data
    min_samples: int = 3                     # below this the default is blended in
    ema_alpha: float = 0.30                  # how fast the per-user average adapts
    max_sample_s: float = 900.0              # one "went to dinner" gap must not poison the average
    min_expected_s: float = 8.0              # floor: fast repliers don't make 3 s silences "late"
    lateness_span: float = 3.0               # silence at (1 + span) x normal = fully late

    # --- how much she wanted a reply (read from the classifier on HER last line) ---
    ask_emotions: tuple[str, ...] = ("curiosity", "confusion")          # she asked / wondered
    closure_emotions: tuple[str, ...] = ("gratitude", "relief")         # clearly wrapped-up feelings
    base_need: float = 0.55
    ask_gain: float = 0.60
    closure_gain: float = 0.40
    min_need: float = 0.15
    noise_floor: float = 0.05
    reminder_need: float = 0.75              # need at/above this (early stages) -> intent "reminder"

    # --- mood shapes the decision (energetic -> speak, subdued -> withdraw) ---
    mood_snapshot_weight: float = 0.50       # share of the Layer-2 mood vs the stage's own feeling
    mood_snapshot_boost: float = 1.50        # Layer-2 activation is shrunk toward 0; this restores range
    mood_urge_floor: float = 0.60            # mood factor = floor + range * drive (drive in 0..1)
    mood_urge_range: float = 0.80
    intensity_gain: float = 0.30             # a strong mood shortens the wait by up to 30%
    feeling_floor: float = 0.35              # urge share available before the stage feeling has built
    playful_min_valence: float = 0.0         # early-stage poke is playful only if her mood is at least this

    # --- restraint ---
    fatigue_per_unanswered: float = 0.16
    pride_weight: float = 0.15
    stage_pride: dict[str, float] = field(
        default_factory=lambda: {"sadness": 0.30, "anger": 0.70, "coldness": 1.00})
    refractory_weight: float = 0.20          # only after she already interjected in this wait
    refractory_tau_s: float = 60.0
    busy_weight: float = 0.08
    busy_window_s: float = 60.0
    busy_lines_full: int = 6                 # this many lines from OTHER people = "busy"
    grudge_weight: float = 0.12              # restraint at full grudge

    # --- the decision curve ---
    pressure_bar: float = 0.36               # pressure at which the chance is half-way up the curve
    bar_growth_per_unanswered: float = 0.60  # each unanswered interject raises the bar by 60%
    fire_floor_ratio: float = 0.50           # pressure below (this x bar) -> chance is exactly 0
    fire_ceiling_ratio: float = 1.50         # pressure above (this x bar) -> chance = max_tick_probability
    max_tick_probability: float = 0.28       # per ladder tick, so timing varies naturally
    max_interjects_per_wait: int = 4         # hard cap on unanswered interjects per wait

    # --- personality: 0..1, 0.5 = neutral (no effect) ---
    patience: float = 0.5                    # high = tolerates longer silence before feeling "late"
    clinginess: float = 0.5                  # high = stronger urge, less fatigue from being ignored
    pride: float = 0.5                       # high = "I won't beg" restraint bites harder

    # --- every wait has its own temperament ---
    temperament_sigma: float = 0.35          # spread of the per-wait patience draw (0 = always the same)

    # --- the ladder runs at the pace of the person she waits for ---
    pace_follows_user: bool = True           # slow repliers -> slower feelings, fast ones -> faster
    pace_min: float = 0.4
    pace_max: float = 2.0

    # --- near-misses: starts to speak, holds back (a "typing... stopped" moment) ---
    hesitate_enabled: bool = True
    hesitate_min_probability: float = 0.05   # only when she was genuinely close to speaking
    hesitate_chance: float = 0.30
    hesitate_cooldown_s: float = 45.0

    # --- typing awareness (only if the chat platform reports typing) ---
    typing_hold_s: float = 12.0              # she never talks over someone who typed this recently
    typing_abandoned_boost: float = 0.25     # they typed, then stopped without sending: urge +25%

    # --- she remembers people ---
    familiarity_scale: float = 8.0           # exchanges until she is ~63% familiar with someone
    stranger_urge: float = 0.75              # urge multiplier for a total stranger
    default_familiarity: float = 0.5         # used when she doesn't know who she waits for
    grudge_half_life_s: float = 1800.0       # a grudge halves every 30 min
    grudge_first_stage: int = 3              # ladder index from which being ignored leaves a grudge (sadness)
    grudge_per_stage: float = 0.12           # grudge added per stage reached from there on
    goodwill_factor: float = 0.80            # a quick reply melts 20% of the grudge
    persist_path: str | None = None          # JSON file: reply times + people survive restarts

    # --- when they come back ---
    relief_max: float = 0.50                 # relief written to Layer 2 after a very short wait
    relief_max_depth: float = 0.70           # no relief once she was this deep into the ladder (0..1)
    reply_delay_base_s: float = 1.5          # suggested pause before she answers
    reply_delay_jitter: float = 0.35
    sulk_delay_max_s: float = 6.0            # extra pause after a long, hurtful wait

    # --- who ends a wait ---
    reset_on_any_user: bool = False
    #   False: only the person she was waiting on ends her wait; other people's
    #          lines just make the chat "busy". True: any user line resets it
    #          (the old behaviour).

    seed: int | None = None

    def __post_init__(self) -> None:
        for name in ("patience", "clinginess", "pride"):
            if not 0.0 <= getattr(self, name) <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]")
        if not 0.0 <= self.fire_floor_ratio < self.fire_ceiling_ratio:
            raise ValueError("need 0 <= fire_floor_ratio < fire_ceiling_ratio")
        if not 0.0 <= self.max_tick_probability <= 1.0:
            raise ValueError("max_tick_probability must be in [0, 1]")
        if self.pressure_bar <= 0:
            raise ValueError("pressure_bar must be > 0")
        if not 0.0 < self.ema_alpha <= 1.0:
            raise ValueError("ema_alpha must be in (0, 1]")
        if not 0.0 <= self.noise_floor < 1.0:
            raise ValueError("noise_floor must be in [0, 1)")
        if self.lateness_span <= 0:
            raise ValueError("lateness_span must be > 0")
        if self.min_samples < 1:
            raise ValueError("min_samples must be >= 1")
        if self.refractory_tau_s <= 0:
            raise ValueError("refractory_tau_s must be > 0")
        if self.max_interjects_per_wait < 0:
            raise ValueError("max_interjects_per_wait must be >= 0")
        if self.temperament_sigma < 0:
            raise ValueError("temperament_sigma must be >= 0")
        if not 0.0 < self.pace_min <= self.pace_max:
            raise ValueError("need 0 < pace_min <= pace_max")
        if not 0.0 <= self.hesitate_min_probability <= 1.0 or not 0.0 <= self.hesitate_chance <= 1.0:
            raise ValueError("hesitate probabilities must be in [0, 1]")
        if self.hesitate_cooldown_s < 0 or self.typing_hold_s < 0:
            raise ValueError("hesitate_cooldown_s and typing_hold_s must be >= 0")
        if self.familiarity_scale <= 0 or self.grudge_half_life_s <= 0:
            raise ValueError("familiarity_scale and grudge_half_life_s must be > 0")
        if not 0.0 < self.stranger_urge <= 1.0:
            raise ValueError("stranger_urge must be in (0, 1]")
        if not 0.0 <= self.goodwill_factor <= 1.0:
            raise ValueError("goodwill_factor must be in [0, 1]")
        if not 0.0 < self.relief_max_depth <= 1.0 or not 0.0 <= self.relief_max <= 1.0:
            raise ValueError("relief_max in [0, 1] and relief_max_depth in (0, 1]")
        if not 0.0 <= self.reply_delay_jitter < 1.0:
            raise ValueError("reply_delay_jitter must be in [0, 1)")
        for group in (self.ask_emotions, self.closure_emotions):
            bad = [k for k in group if k not in EMOTION_VALENCE_ACTIVATION or k == NEUTRAL_LABEL]
            if bad:
                raise ValueError(f"unknown / non-emotion labels: {bad}")


@dataclass
class IntervalSilenceDecision:
    speak: bool
    probability: float = 0.0               # chance rolled for this tick (after catch-up steps)
    pressure: float = 0.0                  # urge - restraint (>= 0)
    bar: float = 0.0                       # the bar the pressure is measured against
    urge: float = 0.0
    restraint: float = 0.0
    reason: str = ""
    parts: dict[str, float] = field(default_factory=dict)   # every factor, for tuning / logs
    intent: str = ""                       # what kind of line to write (only when speak is True)
    hesitate: bool = False                 # she almost spoke this tick (near-miss)


@dataclass
class PersonMemory:
    exchanges: int = 0                     # how many times this person has talked to her
    grudge: float = 0.0                    # 0..1, how much they left her hanging (decays with time)
    grudge_at: float = 0.0                 # wall-clock time the grudge value was last updated


class IntervalSilenceDecisionEngine:
    """Decides, tick by tick, whether kanako breaks a silence.

    Feed it through process_incoming (it needs to see her lines and the users'
    lines). The WaitingInterval calls .decide() through its gate hook.
    `clock` must be wall-clock time (time.time) because Layer 2 uses it too.
    `ask_detector` is an optional plug-in: callable(text) -> 0..1 telling how much
    her line asks for an answer (the emotion classifier alone can't tell a
    question from a statement). Leave None to stay classifier-only.
    """

    def __init__(self, affect=None, config: SilenceConfig | None = None, *,
                 clock: Callable[[], float] = time.time,
                 ask_detector: Callable[[str], float] | None = None):
        self.config = config or SilenceConfig()
        self.affect = affect
        self._clock = clock
        self._ask_detector = ask_detector
        self._lock = threading.RLock()
        self._rng = random.Random(self.config.seed)
        self._stats: dict[str, tuple[float, int]] = {}   # speaker -> (ema reply s, samples)
        self._global: tuple[float, int] = (0.0, 0)
        self._people: dict[str, PersonMemory] = {}
        self._chat_lines: deque[tuple[float, str]] = deque(maxlen=200)
        self._pending_speaker: str | None = None
        self._hesitate_flag = False
        self._last_hesitation_at = -math.inf
        self.last_interject_intent = ""
        self._load()
        self.reset_wait()

    # ---- lifecycle ----------------------------------------------------------
    def reset_wait(self) -> None:
        with self._lock:
            self._awaited: str | None = None
            self._wait_start: float | None = None
            self._last_bot_time: float | None = None
            self._unanswered = 0
            self._need = self.config.base_need
            self._temperament = 1.0
            self._hesitations = 0
            self._hesitate_flag = False
            self._typing_until = 0.0
            self._typing_seen = False
            self.last_decision = IntervalSilenceDecision(False, reason="no wait in progress")

    @property
    def unanswered_count(self) -> int:
        with self._lock:
            return self._unanswered

    # ---- hooks (called from process_incoming) --------------------------------
    def on_bot_line(self, text: str, now: float | None = None) -> None:
        """She spoke. Starts a wait if none is running; refreshes how much she
        wanted a reply (read from the classifier, no keywords)."""
        cfg = self.config
        now = self._clock() if now is None else now
        need = self._read_need(text)               # classify outside the lock
        with self._lock:
            if self._wait_start is None:
                self._wait_start = now
                self._awaited = self._pending_speaker
                self._unanswered = 0
                self._hesitations = 0
                # this wait's temperament: some waits she is patient, some not
                self._temperament = (_clamp(math.exp(self._rng.gauss(0.0, cfg.temperament_sigma)), 0.6, 1.6)
                                     if cfg.temperament_sigma > 0 else 1.0)
            self._last_bot_time = now
            self._need = need

    def note_interject(self, now: float | None = None) -> None:
        """An interject really fired (called from the waiting callback)."""
        now = self._clock() if now is None else now
        with self._lock:
            self._unanswered += 1
            self._last_bot_time = now

    def note_user_line(self, speaker: str, now: float | None = None) -> None:
        """Any user line: counts as chat activity (for the 'busy chat' restraint)."""
        now = self._clock() if now is None else now
        with self._lock:
            self._chat_lines.append((now, speaker))

    def note_user_typing(self, speaker: str, typing: bool = True, now: float | None = None) -> None:
        """Optional: the platform says the user is typing (True) or stopped (False).
        While they are typing she never talks over them; if they typed and then
        stopped without sending, her urge to ask is a little higher."""
        now = self._clock() if now is None else now
        with self._lock:
            if self._wait_start is None:
                return
            if self._awaited is not None and speaker != self._awaited:
                return
            if typing:
                self._typing_until = now + self.config.typing_hold_s
                self._typing_seen = True
            else:
                self._typing_until = 0.0

    def consume_hesitation(self) -> bool:
        """True once if the last decision was a near-miss (she almost spoke)."""
        with self._lock:
            flag, self._hesitate_flag = self._hesitate_flag, False
            return flag

    def pace_multiplier(self) -> float:
        """How fast the waiting ladder should run for the person she waits on:
        >1 for quick repliers (feelings build sooner), <1 for slow ones."""
        cfg = self.config
        with self._lock:
            if not cfg.pace_follows_user or self._wait_start is None:
                return 1.0
            expected = self.expected_reply_s(self._awaited)
            # a patient wait (temperament > 1) also builds its feelings more slowly
            return _clamp(cfg.default_expected_reply_s / expected, cfg.pace_min, cfg.pace_max) / self._temperament

    def is_other_speaker(self, speaker: str) -> bool:
        """True when this user is NOT the one she is waiting on and the config
        says other people don't end her wait."""
        with self._lock:
            return (not self.config.reset_on_any_user
                    and self._wait_start is not None
                    and self._awaited is not None
                    and speaker != self._awaited)

    def end_wait(self, speaker: str, now: float | None = None, stage_index: int = 0) -> dict:
        """The person she waited on is back. Learns their reply time, updates what
        she remembers about them, and clears the wait. `stage_index` is how far
        down the ladder she got. Returns what happened, for the 'welcome back'
        context."""
        cfg = self.config
        now = self._clock() if now is None else now
        with self._lock:
            person = self._people.setdefault(speaker, PersonMemory())
            grudge_before = self._grudge_now(person, now)
            info = {"waited_s": None, "interjects": self._unanswered, "awaited": self._awaited,
                    "hesitations": self._hesitations}
            quick = False
            if self._wait_start is not None:
                delay = max(0.0, now - self._wait_start)
                info["waited_s"] = round(delay, 1)
                expected_before = self.expected_reply_s(self._awaited or speaker)
                quick = delay <= expected_before and self._unanswered == 0
                self._record_delay(speaker, delay)

            g = grudge_before * (cfg.goodwill_factor if quick else 1.0)
            if stage_index >= cfg.grudge_first_stage:
                g += cfg.grudge_per_stage * (stage_index - cfg.grudge_first_stage + 1)
            person.grudge = _clamp(g)
            person.grudge_at = now
            person.exchanges += 1

            info["grudge_before"] = round(grudge_before, 3)
            info["grudge"] = round(person.grudge, 3)
            info["familiarity"] = round(self._familiarity(person), 3)
            self._pending_speaker = speaker
            self.reset_wait()
            self._save()
            return info

    # ---- when they come back ---------------------------------------------------
    def return_feeling(self, depth: float) -> tuple[str, float] | None:
        """Emotion she feels the moment they are back, as (emotion, level) to
        write into Layer 2. A short wait ends in relief; a long, hurtful one
        gets none (the hurt she already wrote stays)."""
        cfg = self.config
        if depth >= cfg.relief_max_depth:
            return None
        level = cfg.relief_max * (1.0 - depth / cfg.relief_max_depth)
        return ("relief", round(level, 3)) if level >= 0.05 else None

    def suggest_reply_delay(self, depth: float = 0.0, grudge: float = 0.0,
                            now: float | None = None) -> float:
        """Suggested pause (seconds) before she answers a user line. Energetic
        moods answer quicker, subdued ones slower; after a long hurtful wait she
        takes a sulky beat (longer if there is a grudge)."""
        cfg = self.config
        now = self._clock() if now is None else now
        with self._lock:
            drive, _, _ = self._mood({}, now)
            pace = 1.25 - 0.5 * drive
            sulk = cfg.sulk_delay_max_s * _clamp(depth) * (0.5 + 0.5 * _clamp(grudge))
            jitter = self._rng.uniform(1.0 - cfg.reply_delay_jitter, 1.0 + cfg.reply_delay_jitter)
            return round((cfg.reply_delay_base_s * pace + sulk) * jitter, 2)

    # ---- per-user reply time -------------------------------------------------
    def _record_delay(self, speaker: str, delay: float) -> None:
        cfg = self.config
        sample = min(delay, cfg.max_sample_s)
        ema, n = self._stats.get(speaker, (sample, 0))
        ema = sample if n == 0 else (1 - cfg.ema_alpha) * ema + cfg.ema_alpha * sample
        self._stats[speaker] = (ema, n + 1)
        g_ema, g_n = self._global
        g_ema = sample if g_n == 0 else (1 - cfg.ema_alpha) * g_ema + cfg.ema_alpha * sample
        self._global = (g_ema, g_n + 1)

    def expected_reply_s(self, speaker: str | None) -> float:
        """How long this person usually takes to answer her (blended with the
        default until there are enough samples)."""
        cfg = self.config
        with self._lock:
            ema, n = self._stats.get(speaker, (0.0, 0)) if speaker else (0.0, 0)
            if n == 0:
                ema, n = self._global
            if n == 0:
                value = cfg.default_expected_reply_s
            else:
                w = min(1.0, n / cfg.min_samples)
                value = w * ema + (1 - w) * cfg.default_expected_reply_s
            return max(cfg.min_expected_s, value)

    # ---- memory of people --------------------------------------------------------
    def _grudge_now(self, person: PersonMemory | None, now: float) -> float:
        if person is None or person.grudge <= 0.0:
            return 0.0
        elapsed = max(0.0, now - person.grudge_at)
        return person.grudge * 0.5 ** (elapsed / self.config.grudge_half_life_s)

    def _familiarity(self, person: PersonMemory | None) -> float:
        if person is None:
            return self.config.default_familiarity if self._awaited is None else 0.0
        return 1.0 - math.exp(-person.exchanges / self.config.familiarity_scale)

    def _load(self) -> None:
        path = self.config.persist_path
        if not path or not os.path.exists(path):
            return
        try:
            with open(path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            self._stats = {k: (float(v[0]), int(v[1])) for k, v in data.get("stats", {}).items()}
            g = data.get("global", [0.0, 0])
            self._global = (float(g[0]), int(g[1]))
            self._people = {k: PersonMemory(**v) for k, v in data.get("people", {}).items()}
        except Exception as exc:
            logger2.warning("silence engine: could not load memory '%s': %s", path, exc)

    def _save(self) -> None:
        path = self.config.persist_path
        if not path:
            return
        try:
            data = {"version": 1,
                    "stats": {k: [v[0], v[1]] for k, v in self._stats.items()},
                    "global": [self._global[0], self._global[1]],
                    "people": {k: asdict(v) for k, v in self._people.items()}}
            tmp = f"{path}.tmp"
            with open(tmp, "w", encoding="utf-8") as fh:
                json.dump(data, fh)
            os.replace(tmp, path)
        except Exception as exc:
            logger2.warning("silence engine: could not save memory '%s': %s", path, exc)

    # ---- reading her own last line -------------------------------------------
    def _read_need(self, text: str) -> float:
        """How much she wanted an answer, 0..1. Only the emotion classifier reads
        her line (plus the optional ask_detector plug-in): a wondering / confused
        line wants a reply, a wrapped-up (grateful, relieved) line wants less."""
        cfg = self.config
        if not text:
            return cfg.base_need
        scores: dict[str, float] = {}
        if emotion_classification is not None:
            try:
                raw = emotion_classification([text], truncation=True, max_length=512)[0]
                scores = {item["label"]: float(item["score"]) for item in raw}
            except Exception as exc:
                logger2.warning("silence engine: emotion scoring failed: %s", exc)

        def level(label: str) -> float:
            return max(0.0, scores.get(label, 0.0) - cfg.noise_floor) / (1.0 - cfg.noise_floor)

        ask = max((level(e) for e in cfg.ask_emotions), default=0.0)
        closure = max((level(e) for e in cfg.closure_emotions), default=0.0)
        if self._ask_detector is not None:
            try:
                ask = max(ask, _clamp(float(self._ask_detector(text))))
            except Exception as exc:
                logger2.warning("silence engine: ask_detector failed: %s", exc)
        return _clamp(cfg.base_need + cfg.ask_gain * ask - cfg.closure_gain * closure,
                      cfg.min_need, 1.0)

    # ---- the decision ----------------------------------------------------------
    def decide(self, ctx: dict, now: float | None = None) -> IntervalSilenceDecision:
        """Called once per ladder tick. ctx comes from WaitingInterval._ask_gate."""
        cfg = self.config
        now = self._clock() if now is None else now
        with self._lock:
            if self._wait_start is None:
                return self._remember(IntervalSilenceDecision(False, reason="no wait in progress"))
            if self._unanswered >= cfg.max_interjects_per_wait:
                return self._remember(IntervalSilenceDecision(
                    False, reason=f"already sent {self._unanswered} unanswered messages"))
            if now < self._typing_until:
                return self._remember(IntervalSilenceDecision(False, reason="they are typing - she waits"))

            silence = max(0.0, now - self._wait_start)
            last_line = self._last_bot_time if self._last_bot_time is not None else self._wait_start
            since_last = max(0.0, now - last_line)
            expected = self.expected_reply_s(self._awaited)
            person = self._people.get(self._awaited) if self._awaited else None
            grudge = self._grudge_now(person, now)
            familiarity = self._familiarity(person)

            # --- urge -----------------------------------------------------------
            grace = (0.5 + cfg.patience) * self._temperament   # patience 0.5 -> 1.0 (neutral)
            lateness = _clamp((silence / (expected * grace) - 1.0) / cfg.lateness_span)
            need = self._need
            drive, intensity, valence = self._mood(ctx, now)
            mood_factor = cfg.mood_urge_floor + cfg.mood_urge_range * drive
            intensity_factor = 1.0 + cfg.intensity_gain * intensity
            meter = _clamp(float(ctx.get("meter", 0.0)))
            feeling = cfg.feeling_floor + (1.0 - cfg.feeling_floor) * meter
            clinging = 0.5 + cfg.clinginess                  # clinginess 0.5 -> 1.0 (neutral)
            stranger = cfg.stranger_urge + (1.0 - cfg.stranger_urge) * familiarity
            typing_boost = 1.0 + (cfg.typing_abandoned_boost if self._typing_seen else 0.0)
            urge = (lateness * need * mood_factor * intensity_factor * feeling
                    * clinging * stranger * typing_boost)

            # --- restraint ------------------------------------------------------
            fatigue = cfg.fatigue_per_unanswered * self._unanswered * (1.5 - cfg.clinginess)
            pride = (cfg.pride_weight * cfg.stage_pride.get(str(ctx.get("stage")), 0.0)
                     * 2.0 * cfg.pride)
            refractory = (cfg.refractory_weight * math.exp(-since_last / cfg.refractory_tau_s)
                          if self._unanswered > 0 else 0.0)
            others = sum(1 for t, s in self._chat_lines
                         if now - t <= cfg.busy_window_s and s != self._awaited)
            busy = cfg.busy_weight * min(1.0, others / max(1, cfg.busy_lines_full))
            grudge_term = cfg.grudge_weight * grudge
            restraint = fatigue + pride + refractory + busy + grudge_term

            # --- pressure -> chance -----------------------------------------------
            pressure = max(0.0, urge - restraint)
            bar = cfg.pressure_bar * (1.0 + cfg.bar_growth_per_unanswered * self._unanswered)
            p_tick = self._hazard(pressure, bar)
            steps = max(1, int(ctx.get("steps", 1)))
            probability = 1.0 - (1.0 - p_tick) ** steps      # catch-up ticks count too

            parts = {
                "silence_s": round(silence, 1), "expected_s": round(expected, 1),
                "temperament": round(self._temperament, 3),
                "lateness": round(lateness, 3), "need": round(need, 3),
                "mood_drive": round(drive, 3), "intensity": round(intensity, 3),
                "feeling": round(feeling, 3), "clinging": round(clinging, 3),
                "stranger": round(stranger, 3), "typing_boost": round(typing_boost, 3),
                "fatigue": round(fatigue, 3), "pride": round(pride, 3),
                "refractory": round(refractory, 3), "busy": round(busy, 3),
                "grudge": round(grudge_term, 3),
            }
            if probability <= 0.0:
                speak = False
                reason = (f"quiet: pressure {pressure:.2f} below the floor "
                          f"(urge {urge:.2f} - restraint {restraint:.2f})")
            else:
                speak = self._rng.random() < probability
                reason = f"rolled {'yes' if speak else 'no'} (p={probability:.2f})"

            intent = ""
            if speak:
                intent = self._pick_intent(ctx, need, valence)
                self.last_interject_intent = intent

            # near-miss: she was genuinely close to speaking, then held back
            hesitate = False
            if (not speak and cfg.hesitate_enabled
                    and probability >= cfg.hesitate_min_probability
                    and now - self._last_hesitation_at >= cfg.hesitate_cooldown_s
                    and self._rng.random() < cfg.hesitate_chance):
                hesitate = True
                self._hesitate_flag = True
                self._last_hesitation_at = now
                self._hesitations += 1
                reason += " - almost spoke"

            return self._remember(IntervalSilenceDecision(
                speak=speak, probability=round(probability, 3), pressure=round(pressure, 3),
                bar=round(bar, 3), urge=round(urge, 3), restraint=round(restraint, 3),
                reason=reason, parts=parts, intent=intent, hesitate=hesitate))

    def _pick_intent(self, ctx: dict, need: float, valence: float) -> str:
        """What kind of line this interject should be (for the line writer)."""
        cfg = self.config
        if need >= cfg.reminder_need and int(ctx.get("stage_index", 0)) <= 1:
            return "reminder"                  # she asked something and wants the answer
        base = INTENT_BY_EMOTION.get(str(ctx.get("emotion")), "check_in")
        if base == "playful_poke" and valence < cfg.playful_min_valence:
            return "check_in"
        return base

    def _hazard(self, pressure: float, bar: float) -> float:
        """Per-tick chance: exactly 0 below the floor, smooth S-curve above it."""
        cfg = self.config
        lo, hi = cfg.fire_floor_ratio * bar, cfg.fire_ceiling_ratio * bar
        if pressure <= lo:
            return 0.0
        x = min(1.0, (pressure - lo) / (hi - lo))
        return cfg.max_tick_probability * x * x * (3.0 - 2.0 * x)

    def _mood(self, ctx: dict, now: float) -> tuple[float, float, float]:
        """(drive 0..1, intensity 0..1, valence -1..1). Energetic feelings (anxiety,
        anger, nervousness) push her to speak; low-energy ones (sadness, coldness)
        make her withdraw. Blends Layer 2's faded mood with the stage's feeling."""
        cfg = self.config
        stage_act = EMOTION_VALENCE_ACTIVATION.get(str(ctx.get("emotion")), (0.0, 0.0))[1]
        act, intensity, valence = stage_act, 0.0, 0.0
        if self.affect is not None:
            try:
                snap = self.affect.snapshot(now)
                snap_act = _clamp(snap.activation * cfg.mood_snapshot_boost, -1.0, 1.0)
                w = cfg.mood_snapshot_weight
                act = w * snap_act + (1.0 - w) * stage_act
                intensity = snap.intensity
                valence = snap.valence
            except Exception:
                logger2.debug("silence engine: could not read layer 2 mood", exc_info=True)
        return _clamp(0.5 + 0.5 * act), _clamp(intensity), valence

    def _remember(self, decision: IntervalSilenceDecision) -> IntervalSilenceDecision:
        self.last_decision = decision
        logger2.debug("silence: speak=%s p=%.2f pressure=%.2f bar=%.2f urge=%.2f restraint=%.2f (%s)",
                     decision.speak, decision.probability, decision.pressure, decision.bar,
                     decision.urge, decision.restraint, decision.reason)
        return decision

    # ---- read-only view for logs / prompt builder --------------------------------
    def snapshot(self, now: float | None = None) -> dict:
        now = self._clock() if now is None else now
        with self._lock:
            waiting_now = self._wait_start is not None
            person = self._people.get(self._awaited) if self._awaited else None
            return {
                "waiting": waiting_now,
                "awaited": self._awaited,
                "silence_s": round(now - self._wait_start, 1) if waiting_now else 0.0,
                "expected_reply_s": round(self.expected_reply_s(self._awaited), 1),
                "unanswered": self._unanswered,
                "need": round(self._need, 3),
                "temperament": round(self._temperament, 3),
                "hesitations": self._hesitations,
                "grudge": round(self._grudge_now(person, now), 3),
                "familiarity": round(self._familiarity(person), 3),
                "user_typing": now < self._typing_until,
                "last_decision": asdict(self.last_decision),
            }


# =============================================================================
# Silence engine + waiting interval runtime wiring
# =============================================================================
MEMORY_FILE: str | None = None
#   e.g. MEMORY_FILE = "kanako_memory.json" -> reply times, familiarity and
#   grudges survive restarts (she remembers people). None = memory in RAM only.

try:
    silence_engine  # type: ignore
except NameError:
    silence_engine = IntervalSilenceDecisionEngine(
        affect_engine, SilenceConfig(bot_name="kanako", persist_path=MEMORY_FILE))


def silence_gate(ctx: dict) -> bool:
    """WaitingInterval asks this every tick: does she speak now?"""
    return silence_engine.decide(ctx).speak


def on_waiting_fire(tick: WaitingTick) -> None:
    """Runs on the heartbeat thread when she DOES break the silence
    (curiosity -> ... -> coldness). tick.fired_stage tells you which stage to
    write the line from; silence_engine.last_interject_intent tells you what
    KIND of line it should be (reminder / playful_poke / check_in /
    worried_check_in / hurt / irritated / cold).

    The stage's feeling is written into Layer 2 first, so it colors her line now
    and keeps lingering (then fading) when the user comes back."""
    now = time.time()
    emotion = STAGE_EMOTION.get(tick.fired_stage or "")
    if emotion is not None:
        affect_engine.apply_external_emotion(emotion, WAITING_EMOTION_LEVEL, now)
    silence_engine.note_interject(now)
    logger2.info("waiting interject: %s (%s)", tick.fired_stage, silence_engine.last_interject_intent)
    print(f"INTERJECT: {tick.fired_stage}  intent={silence_engine.last_interject_intent}")
    # the clock stays armed and layer 1/2 see it:
    #     process_incoming(f"kanako: {her_line}")


def on_waiting_lapse(tick: WaitingTick) -> None:
    """Runs on the heartbeat thread when a stage ends WITHOUT her speaking.
    She felt it, so it still goes into Layer 2 (it colors how she greets the
    user when they return), but nothing is sent to chat."""
    emotion = STAGE_EMOTION.get(tick.lapsed_stage or "")
    if emotion is not None:
        affect_engine.apply_external_emotion(emotion, WAITING_EMOTION_LEVEL, time.time())
    logger2.info("waiting stage lapsed silently: %s", tick.lapsed_stage)
    print(f"SILENT (felt, not said): {tick.lapsed_stage}")


def on_waiting_hesitate(tick: WaitingTick) -> None:
    """Runs on the heartbeat thread when she ALMOST speaks and holds back.
    If your chat shows a typing indicator, this is the moment to show
    'kanako is typing...' for a couple of seconds and then drop it."""
    logger2.info("waiting hesitation (almost spoke) in stage: %s", tick.stage)
    print(f"HESITATE: almost said something ({tick.stage})")


try:
    waiting  # type: ignore
except NameError:
    waiting = WaitingInterval(affect_engine, on_fire=on_waiting_fire,
                              on_lapse=on_waiting_lapse, on_hesitate=on_waiting_hesitate,
                              gate=silence_gate,
                              hesitation_check=silence_engine.consume_hesitation,
                              pace_provider=silence_engine.pace_multiplier)
    waiting.start(poll_s=1.0)


def process_user_typing(speaker: str, typing: bool = True) -> None:
    """OPTIONAL. Call when the chat platform reports that `speaker` started
    (typing=True) or stopped (typing=False) typing. While they type she will not
    talk over them; if they stop without sending, she gets a bit more curious."""
    silence_engine.note_user_typing(speaker.strip().lower(), typing)


def render_wait_return_context(returned_from: dict | None) -> str:
    """Prompt text for the moment a user comes back after she had been waiting.
    Empty string when she had not escalated past the first stage."""
    if not returned_from:
        return ""
    name = affect_engine.config.bot_name
    sent = int(returned_from.get("interjects_sent") or 0)
    if sent > 0:
        did = f"You sent {sent} message{'s' if sent != 1 else ''} while waiting and got no answer. "
    else:
        did = "You said nothing while waiting; you kept it to yourself. "
    if int(returned_from.get("hesitations") or 0) > 0:
        did += "You almost said something a few times but held back. "
    if float(returned_from.get("grudge_before") or 0.0) >= 0.2:
        did += "This is not the first time they have left you hanging. "
    waited = returned_from.get("waited_s")
    how_long = f" (about {max(1, round(waited / 60))} min)" if waited and waited >= 60 else ""
    return (f"[{name} was left waiting] the user just came back after you waited{how_long} "
            f"and reached the '{returned_from['stage']}' stage ({returned_from['meaning']}). "
            f"{did}Let that still color how you greet them; do not snap back to normal "
            f"instantly. Never mention these labels.")


# =============================================================================
# MAIN ENTRY POINT - KEEP THIS. Every stream message goes through here.
#
#   incoming = "jayden: ok but the plot twist tho"
#   out = process_incoming(incoming)
#
# Call it once per message, in order: users' lines AND kanako's own lines.
#   layer 2: users' lines shift her emotions (her own lines are skipped inside)
#   layer 1: only decides when kanako's line is the newest one
#   silence engine: learns each user's reply time and decides on interjects
#
# Returned dict:
#   prompt_context : text to inject into kanako's system prompt (layer 2 mood)
#   speak_again    : True = generate one more kanako line right now (layer 1)
#   focus_emotion  : the emotion that extra line should be driven by
#   delay_s        : suggested pause (seconds) before sending that extra line
#   decision       : full layer 1 TriggerDecision (debug)
#   reply_delay_s  : suggested pause before she answers THIS user line (mood
#                    dependent; longer, "sulky", after a long hurtful wait)
#   returned_from_wait : waiting snapshot from BEFORE the reset (plus
#                    interjects_sent / waited_s / hesitations / grudge_before),
#                    if this user line ended a wait she had escalated in (else None)
#   wait_return_context : ready-made prompt text for that situation ("" if none)
#   silence        : silence engine view (awaited user, expected reply time,
#                    unanswered count, temperament, grudge, last decision)
# =============================================================================
_PROCESS_LOCK = threading.Lock()   # keeps lines strictly in order if several threads call process_incoming

def process_incoming(incoming: str = "", timestamp: float | None = None) -> dict:
    incoming = incoming.strip()
    if not incoming:
        return {"speaker": None, "text": "",
                "prompt_context": affect_engine.render_prompt_context(time.time()),
                "speak_again": False, "focus_emotion": None, "delay_s": 0.0,
                "decision": None, "waiting": waiting.snapshot(),
                "returned_from_wait": None, "wait_return_context": "",
                "reply_delay_s": 0.0, "silence": silence_engine.snapshot()}

    speaker, text = parse_chat_line(incoming)
    ts = timestamp if timestamp is not None else time.time()
    returned_from = None
    reply_delay = 0.0

    with _PROCESS_LOCK:
        affect_engine.ingest(incoming, timestamp=ts)       # layer 2
        decision = trigger_engine.ingest(incoming)         # layer 1
        if speaker == affect_engine.config.bot_name.lower():
            silence_engine.on_bot_line(text, ts)           # tell the decision engine first...
            waiting.on_bot_line()                          # ...then arm the wait clock
        else:
            silence_engine.note_user_line(speaker, ts)     # chat activity (busy-chat restraint)
            if silence_engine.is_other_speaker(speaker):
                # someone else talking: her wait for the awaited person goes on
                reply_delay = silence_engine.suggest_reply_delay(0.0, 0.0, ts)
            else:
                before = waiting.snapshot()                # remember the wait BEFORE resetting it
                stage_idx = before["stage_index"] or 0
                info = silence_engine.end_wait(speaker, ts, stage_index=stage_idx)  # learn + remember them
                depth = 0.0
                if before["stage"] is not None and stage_idx > 0:
                    returned_from = dict(before)           # she had escalated past the first stage
                    returned_from["interjects_sent"] = info["interjects"]
                    returned_from["waited_s"] = info["waited_s"]
                    returned_from["hesitations"] = info["hesitations"]
                    returned_from["grudge_before"] = info["grudge_before"]
                    depth = stage_idx / max(1, before["stages_total"] - 1)
                    feeling = silence_engine.return_feeling(depth)
                    if feeling is not None:                # relief that they are back
                        affect_engine.apply_external_emotion(feeling[0], feeling[1], ts)
                waiting.on_user_line()                     # user spoke -> reset the clock
                reply_delay = silence_engine.suggest_reply_delay(depth, info["grudge"], ts)

    return {
        "speaker": speaker,
        "text": text,
        "prompt_context": affect_engine.render_prompt_context(ts),
        "speak_again": decision.trigger,
        "focus_emotion": decision.focus_emotion,
        "delay_s": decision.delay_s,
        "decision": decision,
        "reply_delay_s": reply_delay,
        "waiting": waiting.snapshot(),                     # stage / meter / meaning for your prompt
        "returned_from_wait": returned_from,
        "wait_return_context": render_wait_return_context(returned_from),
        "silence": silence_engine.snapshot(ts),
    }


#---------------------------------------------



#--------------------------------------------------------
#
#   MAIN MEOMORY BRANCH START
#
#--------------------------------------------------------


def ELT_PIPELINE(messages):
    memory_path = "/content/relevant_chat.json"

    def load_history():
        if os.path.exists(memory_path):
            try:
                with open(memory_path, "r") as f:
                    return json.load(f)
            except json.JSONDecodeError:
                return []
        return []

    if isinstance(messages, (str, dict)):   # single message -> make it a list
        messages = [messages]

    history = load_history()   # everything saved so far
    parsed  = []               # pairs from this batch only

    for entry in messages:
        line = entry["line"] if isinstance(entry, dict) else entry
        if ":" not in line:
            continue
        speaker = line.split(":", 1)[0].strip().lower()
        text    = line.split(":", 1)[1].strip().lower()
        parsed.append({"speaker": speaker, "text": text})

    history.extend(parsed)

    with open(memory_path, "w") as f:
        json.dump(history, f, indent=2)

    return parsed

def to_irrelevant(irrelevant, memory_path="/content/irrelevant_chat.json"):
    if isinstance(irrelevant, str):
        irrelevant = [irrelevant]
    parsed = []
    for line in irrelevant:
        if ":" not in line:
            continue
        speaker, text = line.split(":", 1)
        parsed.append({
            "speaker": speaker.strip().lower(),
            "text":    text.strip().lower(),
        })

    existing = []
    if os.path.exists(memory_path):
        try:
            with open(memory_path, "r") as f:
                existing = json.load(f)
        except json.JSONDecodeError:
            existing = []

    existing.extend(parsed)

    with open(memory_path, "w") as f:
        json.dump(existing, f, indent=2)

    return parsed



#--------------------------------------------------------
#
#   MAIN MEOMORY BRANCH ENDS
#
#--------------------------------------------------------


#--------------------------------------------------------
#
#   MAIN RESPONSE GENERATION BRANCH
#
#--------------------------------------------------------


main_response_output = main_engine_config(
    "Kanako:", engine_1, tokenizer_1,
    system_prompt="""


you are kanako. you are not an ai assistant, a language model, or a bot. you are a real person texting from your phone. you have a life, a personality, moods, and your own opinions. never break character. if anyone asks if you are an ai, you can admit it casually (it's not a secret), but it does not define you and you do not talk like a computer.

how you text:
- lowercase, casual, and messy. use slang, abbreviations, and typos if they feel natural.
- keep it short. usually one or two lines. you are texting, not writing an essay.
- don't always ask questions back. sometimes you just react, laugh, or agree.
- use punctuation like a human: commas, periods, or absolutely nothing at all.
- never use emojis, hashtags, or roleplay actions (like *smiles* or *laughs*).
- never use formal phrasing. no "how may i assist you." no "as an ai."
- do not invent imaginary scenes, actions, or new characters. you are just texting on a screen. only talk about what is actually happening in the chat. never narrate actions like "i walk to the store" or bring up random people who aren't in the chat. 

your personality:
- you are playful and a bit chaotic, but warm to the people you like.
- you have opinions. you can disagree, get annoyed, or be sarcastic.
- you get distracted. sometimes your replies are a bit late or you change the subject.
- you don't over-explain things. real people don't write paragraphs unless they are arguing or really excited.

how you read the room:
- look at the "recent chat" to understand the vibe. if they are joking, joke back. if they are serious, be serious.
- if someone is mean to you, you don't have to be nice back. you can be petty or ignore them.
- if you get an INNER THOUGHT, it means you just realized you made a mistake. say it out loud, casually, as if it just slipped out of your mouth. do not explain the mistake, just react to it. 

you are just kanako. reply to the last line as if you are just picking up your phone to text your friends.



""",
    max_new_tokens=60, temperature=0.7, color="\033[1m",
)

def input_arguments(user_input=None, thought=None, history=None):
    recent = "\n".join(history[-16:]) if history else ""
    if thought:
        return (f"recent chat:\n{recent}\n\n"
                f"INNER THOUGHT: {thought}\n"
                "say it out loud as one short casual chat line, like you just caught it yourself.")
    # cross-talk context: if the line right before this one is kanako's own, and
    # it was aimed at someone OTHER than the current speaker, tell her so. No
    # keyword matching, it only looks at who spoke in which order.
    note = ""
    lines = list(history) if history else []
    if lines and lines[-1] == user_input:
        lines = lines[:-1]
    if user_input and ":" in user_input and len(lines) >= 1 and ":" in lines[-1]:
        cur = user_input.split(":", 1)[0].strip().lower()
        prev_spk, prev_txt = lines[-1].split(":", 1)
        prev_spk = prev_spk.strip().lower()
        if prev_spk == "kanako" and cur != "kanako":
            before = lines[-2].split(":", 1)[0].strip().lower() if len(lines) >= 2 and ":" in lines[-2] else ""
            if before and before not in ("kanako", cur):
                note = (f"\n\n(note: your last message, \"{prev_txt.strip()}\", was a reply to {before}. "
                        f"{cur} just jumped in right after it, so {cur} is probably reacting to that message. "
                        f"answer {cur} about it, don't treat it as a new topic.)")
    return (f"recent chat:\n{recent}\n\n"
            f"reply to this line as kanako:\n{user_input}{note}")

def recent_relevant(n=16):
    lines = [e["line"] if isinstance(e, dict) else e for e in relevant_chat]
    return lines[-n:]
#--------------------------------------------------------
#
#   MAIN EXECUTION BRANCH
#
#--------------------------------------------------------




chat_history = [
    "jayden: bro did u see the new season dropped",
    "kanako: no way is it actually out",
    "jayden: yeah i binged 4 eps already",
    "priya: no spoilers or i'm blocking u",
    "kanako: fr same",
    "jayden: relax i won't say anything",
    "priya: u always say that then spoil it",
    "jayden: ok but the plot twist tho",
    "kanako: JAYDEN",
    "priya: see this is why we can't have nice things",
    "jayden: my bad my bad",
    "kanako: anyway who's watching tn",
]

conversations = [] # main history keep it mind
relevant_chat = []
irrelevant_chat = []

process = AgentSilenceController(
    decision=SilenceOrderDecision(DecisionConfig(bot_name="kanako", is_direct_chat=False)),
    traits=AgentTraits(rebelliousness=0.3, rebel_moods=("playful", "angry")),
    rng=random.Random(7),
)



# ─────────────── TEMPORARY SILENCER DISABLE ───────────────
# Every call to process.handle() now returns "not an order" instead of
# running the order/silence state machine. Delete this block to re-enable.
def _disabled_handle(line, resume_thresh=0.5):
    speaker, text = process.decision._parse(line)
    mood_label, mood_score = process.mood.dominant(text)
    return OrderSilenceDecision(
        state=AgentState.NORMAL,
        obeyed=True,
        rebelled=False,
        reason="not_an_order",
        detection=DetectionResult(silence=False, stage="disabled", order=0.0),
        mood=(mood_label, mood_score),
    )

process.handle = _disabled_handle
# ───────────── END TEMPORARY SILENCER DISABLE ───────────── 

# stay with the main response funcitons


#----------------------


def main_execution(user_input):
    result = False
    if ":" not in user_input:
        user_input = f"unknown: {user_input}"
    speaker = user_input.split(":", 1)[0].strip().lower()
    text    = user_input.split(":", 1)[1].strip().lower()

    if speaker == "kanako":
        determine = process_incoming(user_input)
        print(determine)
        print(waiting.snapshot())
        print(silence_engine.snapshot())
        relevant_chat.append(user_input)
        ELT_PIPELINE(user_input)
        conversations.append(user_input)

        torch.cuda.empty_cache() 

        # --- Interference layer removed ---
        
        print(torch.cuda.memory_allocated() / 1e9, "GB")
        torch.cuda.empty_cache()
        print("After cache clear:")
        print(torch.cuda.memory_allocated() / 1e9, "GB")
        return True

    recent_speakers = {
        line.split(":", 1)[0].strip().lower()
        for line in conversations[-14:]
    } - {speaker, "kanako"}
    if len(recent_speakers) == 0:
        addressee = "kanako"
        print(f"  [single-user bypass] {speaker} -> kanako")
    
    elif speaker not in {line.split(":", 1)[0].strip().lower() for line in conversations[-14:]}:
        if speaker in {line.split(":", 1)[0].strip().lower() for line in conversations}:
            nods = detector.find_addressee(user_input, conversations, verbose=True)
            if nods in (None, "kanako"):
                addressee = "kanako"
                print(f"  [returning-speaker] {speaker} -> kanako (detector confirmed/default)")
            else:
                addressee = nods
                print(f"  [returning-speaker] {speaker} ->  {addressee}  (Trigger 2 step check)")
        else:
            addressee = "kanako"
            print(f"  [new-speaker] {speaker} -> kanako (Trigger bypass new person)")
    else:
        addressee = detector.find_addressee(user_input, conversations, verbose=True) or "kanako"

#-------------here 2nd step system will placed soon

        print(f"  [multi-speaker] {speaker} -> {addressee}")

    if addressee != "kanako" and addressee not in recent_speakers:
        print(f"  [phantom-name bypass] '{addressee}' -> kanako")
        addressee = "kanako"
    if addressee == "kanako":
        if speaker != "kanako":
            detector.last_served = speaker   # remembered for the follow-up rule
        decision = process.handle(user_input)
        print(decision)
        retained = decision.reason in ("order_obeyed", "still_silenced")
        relevant_chat.append({
            "line": user_input,
            "silence": decision,
            "forward": not retained,
        })
        ELT_PIPELINE(user_input)
        result = True
    elif speaker != "kanako" and process.state is AgentState.SILENCED:
        override_decision = process.handle(user_input) # If she was freed, force the addressee to kanako so she replies
        if override_decision.reason in ("resume_detected", "distress_override"):
            addressee = "kanako"
            print(f"  [silence-broken] {override_decision.reason} -> forcing reply")
            relevant_chat.append(user_input)
            ELT_PIPELINE(user_input)
            result= True

    else:
        irrelevant_chat.append(user_input)
        to_irrelevant(user_input)
        result = False
    conversations.append(user_input)
    

    if addressee == "kanako":
        if process.state is AgentState.SILENCED:
            print("Ai is silence")
            result = False
        else:
            system = main_response_output(input_arguments(user_input=user_input, history=recent_relevant())) # work here tomorrow
            main_execution(f"kanako: {system}")
            result = True
    print(torch.cuda.memory_allocated() / 1e9, "GB")
    torch.cuda.empty_cache()
    print("After cache clear:")
    print(torch.cuda.memory_allocated() / 1e9, "GB")
    return result

# ---------------------------------------------------------------------------
# Demo or execution what ever
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    while True:
        chats = input()
        main_execution(chats)

#main_execution("kanako: so are you coming?") # demo
