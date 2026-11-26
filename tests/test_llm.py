import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import llm

STORY = (
    'Amid the tall sunflowers, two friends sat together. Mia loved the bright flowers. '
    '"Look at that one!" Mia said, pointing to a sunflower. Alex laughed, "It is a giant!" They went home.'
)


def test_visible_answer_strips_followups_variants():
    base = '{"scenes":[]}'
    for tail in (
        "\n```followups\n[\"a\",\"b\"]\n```",
        "\n```Follow-ups\nx\n```",
        "\n<followups>x</followups>",
        "\n\n**Follow-up questions:**\n- x",
        "\nFollowups: x",
    ):
        assert llm.visible_answer(base + tail) == base


def test_extract_json_ignores_fences_and_trailing_followups():
    reply = '```json\n{"scenes":[{"text":"Hi.","keywords":["cat"]}]}\n```\n```followups\n["more?"]\n```'
    assert llm.extract_json(reply)["scenes"][0]["keywords"] == ["cat"]


def test_extract_json_with_prose_prefix():
    assert llm.extract_json('Sure! Here you go: [["cat"],["dog"]]') == [["cat"], ["dog"]]


def test_split_sentences_keeps_quotes_together():
    out = llm.split_sentences(STORY)
    assert out[2] == '"Look at that one!" Mia said, pointing to a sunflower.'
    assert out[3] == 'Alex laughed, "It is a giant!"'
    assert len(out) == 5


def test_clean_keywords():
    assert llm.clean_keywords(["The Cat", "cat", "a big old red barn door", "x", 3, "Dog!"]) == ["cat", "dog"]


def test_plan_scenes_accepts_faithful_split(monkeypatch):
    sentences = llm.split_sentences(STORY)
    reply = json.dumps({"scenes": [{"text": s, "keywords": ["cat", "dog"]} for s in sentences]}) + "\n```followups\n[]\n```"
    monkeypatch.setattr(llm, "exa_complete", lambda prompt, retries=3: reply)
    scenes = llm.plan_scenes(STORY)
    assert [s["text"] for s in scenes] == sentences


def test_group_sentences_merges_short_runs():
    out = llm.group_sentences(["One two.", "Three four five.", "Six " * 30 + "seven."])
    assert out[0] == "One two. Three four five." and len(out) == 2


def test_plan_scenes_rejects_rewritten_story_and_falls_back(monkeypatch):
    calls = []

    def fake(prompt, retries=3):
        calls.append(prompt)
        if "SENTENCES:" in prompt:
            return '[["sunflower"],["flower"],["girl"],["boy"],["house"]]'
        return json.dumps({"scenes": [{"text": "A completely different tale about robots.", "keywords": ["robot"]}]})

    monkeypatch.setattr(llm, "exa_complete", fake)
    scenes = llm.plan_scenes(STORY)
    assert [s["text"] for s in scenes] == llm.group_sentences(llm.split_sentences(STORY))
    assert scenes[0]["keywords"] == ["sunflower"]


def test_plan_scenes_survives_total_llm_failure(monkeypatch):
    def boom(prompt, retries=3):
        raise RuntimeError("down")

    monkeypatch.setattr(llm, "exa_complete", boom)
    scenes = llm.plan_scenes(STORY)
    assert scenes and all(s["keywords"] for s in scenes)


def test_drop_names_removes_capitalised_only_words():
    story = "Rita saw a rock. The rock was big. Rita smiled at the bear."
    assert llm.drop_names(["rita", "rock", "bear", "moon"], story) == ["rock", "bear", "moon"]


def test_heuristic_keywords_keep_sentence_order():
    assert llm.heuristic_keywords("The lighthouse stood above the harbour and a rowboat") == ["lighthouse", "harbour", "rowboat"]
