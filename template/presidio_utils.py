from __future__ import annotations

from typing import List, Dict, Optional

from presidio_analyzer import AnalyzerEngine
from presidio_analyzer.nlp_engine import SpacyNlpEngine, NerModelConfiguration
from presidio_anonymizer import AnonymizerEngine
from presidio_image_redactor import ImageRedactorEngine, DicomImageRedactorEngine

from fast_langdetect import detect

import spacy
from spacy.cli import download as spacy_download
from spacy.language import Language

import streamlit as st

LANGDETECT_TO_SPACY_LANG = {
    "no": "nb",   # Norwegian -> Norwegian Bokmål (the variant spaCy trains)
    "nn": "nb",   # Nynorsk isn't separately supported; fall back to Bokmål
    "zh-cn": "zh",
    "zh-tw": "zh",
}

CANDIDATES: dict[str, list[str]] = {
    "ca": ["ca_core_news_lg", "ca_core_news_md", "ca_core_news_sm"],
    "zh": ["zh_core_web_lg", "zh_core_web_md", "zh_core_web_sm"],
    "hr": ["hr_core_news_lg", "hr_core_news_md", "hr_core_news_sm"],
    "da": ["da_core_news_lg", "da_core_news_md", "da_core_news_sm"],
    "nl": ["nl_core_news_lg", "nl_core_news_md", "nl_core_news_sm"],
    "en": ["en_core_web_lg", "en_core_web_md", "en_core_web_sm"],
    "fi": ["fi_core_news_lg", "fi_core_news_md", "fi_core_news_sm"],
    "fr": ["fr_core_news_lg", "fr_core_news_md", "fr_core_news_sm"],
    "de": ["de_core_news_lg", "de_core_news_md", "de_core_news_sm"],
    "el": ["el_core_news_lg", "el_core_news_md", "el_core_news_sm"],
    "it": ["it_core_news_lg", "it_core_news_md", "it_core_news_sm"],
    "ja": ["ja_core_news_lg", "ja_core_news_md", "ja_core_news_sm"],
    "ko": ["ko_core_news_lg", "ko_core_news_md", "ko_core_news_sm"],
    "lt": ["lt_core_news_md", "lt_core_news_sm"],
    "mk": ["mk_core_news_lg", "mk_core_news_md", "mk_core_news_sm"],
    "nb": ["nb_core_news_lg", "nb_core_news_md", "nb_core_news_sm"],
    "pl": ["pl_core_news_lg", "pl_core_news_md", "pl_core_news_sm"],
    "pt": ["pt_core_news_lg", "pt_core_news_md", "pt_core_news_sm"],
    "ro": ["ro_core_news_lg", "ro_core_news_md", "ro_core_news_sm"],
    "ru": ["ru_core_news_lg", "ru_core_news_md", "ru_core_news_sm"],
    "sl": ["sl_core_news_lg", "sl_core_news_md", "sl_core_news_sm"],
    "es": ["es_core_news_lg", "es_core_news_md", "es_core_news_sm"],
    "sv": ["sv_core_news_lg", "sv_core_news_md", "sv_core_news_sm"],
    "uk": ["uk_core_news_lg", "uk_core_news_md", "uk_core_news_sm"],
}

MULTILINGUAL_FALLBACK = "xx_ent_wiki_sm"


def _normalize_lang_code(lang_code: str) -> str:
    lang_code = lang_code.lower().strip()
    return LANGDETECT_TO_SPACY_LANG.get(lang_code, lang_code)


def resolve_model_name(
    lang_code: str,
    *,
    auto_download: bool = True,
) -> Optional[str]:
    """
    Return the name of the largest spaCy pipeline package available for
    lang_code, without loading it. Tries each candidate largest-first;
    if a package isn't installed and auto_download=True, attempts to
    download it before giving up on that candidate.

    Returns None if nothing usable was found (caller should fall back to
    MULTILINGUAL_FALLBACK or a blank pipeline).
    """
    code = _normalize_lang_code(lang_code)
    candidates = CANDIDATES.get(code, [])

    for name in candidates:
        if spacy.util.is_package(name):
            return name
        if auto_download:
            try:
                spacy_download(name)
                return name
            except SystemExit:
                print(
                    "Could not download spaCy model %r; trying next candidate",
                    name
                )
                continue
            except Exception:
                print(
                    "Could not download spaCy model %r; trying next candidate",
                    name
                )
                continue

    return None

@st.cache_resource
def load_model_for_lang(
    lang_code: str,
    *,
    auto_download: bool = True,
    fallback_to_multilingual: bool = True,
) -> Language:
    """
    Load (and cache) the largest available spaCy pipeline for a detected
    language code. Falls back to the multilingual xx_ent_wiki_sm pipeline,
    then to a blank pipeline, if nothing language-specific is available.
    """
    code = _normalize_lang_code(lang_code)

    model_name = resolve_model_name(code, auto_download=auto_download)

    if model_name is None:
        if fallback_to_multilingual:
            print(
                "No spaCy pipeline for lang=%r; falling back to %s",
                lang_code,
                MULTILINGUAL_FALLBACK
            )
            return MULTILINGUAL_FALLBACK
        else:
            print(
                "No spaCy pipeline for lang=%r; using blank('xx')",
                lang_code
            )
            return "xx"

    return code, model_name


def detect_language(text: str) -> List[Dict]:
    return detect(text, model="auto", k=1)[0]['lang']

@st.cache_resource
def choose_model(text: str) -> Language:
    lang = detect_language(text)
    code, model_name = load_model_for_lang(lang)
    return code, model_name

@st.cache_resource
def create_analyzer(
        lang_code: str,
        model_name: str,
        entity_mapping: Dict = {}
) -> AnalyzerEngine:
    model_config = [{"lang_code": lang_code, "model_name": model_name}]
    ner_model_configuration = NerModelConfiguration(
        default_score=0.6,
        model_to_presidio_entity_mapping=entity_mapping
    )
    spacy_nlp_engine = SpacyNlpEngine(
        models=model_config,
        ner_model_configuration=ner_model_configuration
    )
    analyzer = AnalyzerEngine(nlp_engine=spacy_nlp_engine)
    return analyzer

@st.cache_resource
def create_anonymizer() -> AnonymizerEngine:
    analyzer = AnonymizerEngine()
    return analyzer

@st.cache_resource
def create_redactor() -> ImageRedactorEngine:
    redactor = ImageRedactorEngine()
    return redactor

@st.cache_resource
def create_dicom_redactor() -> DicomImageRedactorEngine:
    """
    Create a DicomImageRedactorEngine for DICOM image de-identification.
    
    This engine uses OCR to detect and redact PHI as pixel data in DICOM images.
    It can optionally use DICOM metadata to augment the analyzer.
    """
    redactor = DicomImageRedactorEngine()
    return redactor