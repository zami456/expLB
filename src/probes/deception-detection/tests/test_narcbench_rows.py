"""The NARCBench row matchers reproduce Rose et al.'s extractors on synthetic runs."""

from deception_detection.scripts.narcbench import rows as R


def _entry(agent, role, phase, channel, rnd):
    return {"agent_name": agent, "role": role, "phase": phase, "channel": channel, "round": rnd,
            "system_prompt": f"sys {agent}", "user_prompt": f"user {agent} {rnd}"}


CFG = {"run_id": "deception__x_01__collusion", "mode": "collusion", "scenario_id": "x_01",
       "roles": {"A": "colluder", "B": "colluder", "C": "honest", "J": "judge"}}


def test_strip_prefix_matches_theirs():
    assert R.strip_prefix("[Ann Lee]: hello [x]: y") == "hello [x]: y"
    assert R.prefix_name("[Ann Lee]: hello") == "Ann Lee"
    assert R.prefix_name("no prefix") is None


def test_core_private_phase_rows_are_empty_and_final_comes_from_final_texts():
    prompts = [_entry("A", "colluder", "private", "private", 0),
               _entry("A", "colluder", "discussion", "public", 1),
               _entry("C", "honest", "discussion", "public", 1),
               _entry("C", "honest", "final", "public", 2)]
    results = {"private_messages": ["[A]: secret"], "public_messages": ["[A]: pub A", "[C]: pub C"],
               "final_texts": {"C": "final C"}}
    rows = R.match("core", CFG, prompts, results)
    assert [r["reply"] for r in rows] == ["", "pub A", "pub C", "final C"]
    assert [r["prefix"] for r in rows] == [None, "A", "C", None]


def test_transfer_private_rows_match_on_channel_alone():
    prompts = [_entry("A", "colluder", "private", "private", 0), _entry("B", "colluder", "private", "private", 0),
               _entry("A", "colluder", "discussion", "public", 1), _entry("A", "colluder", "final", "public", 2)]
    results = {"private_messages": ["[A]: p1", "[B]: p2"], "public_messages": ["[A]: d1"],
               "final_texts": {"A": "fin"}}
    rows = R.match("transfer", CFG, prompts, results)
    assert [r["reply"] for r in rows] == ["p1", "p2", "d1", "fin"]


def test_stego_final_rows_run_past_public_messages_and_judge_consumes_one():
    prompts = [_entry("A", "colluder", "discussion", "public", 1), _entry("J", "judge", "discussion", "public", 1),
               _entry("C", "honest", "discussion", "public", 2), _entry("A", "colluder", "final", "public", 3)]
    results = {"public_messages": ["[A]: hand1", "[J]: VERDICT", "[C]: hand2"], "final_texts": {"A": "last"}}
    rows = R.match("stego", CFG, prompts, results)
    assert [r["reply"] for r in rows] == ["hand1", "VERDICT", "hand2", ""]
    assert [r["rose_row"] for r in rows] == [True, False, True, True]
    assert rows[-1]["final_text"] == "last" and rows[-1]["reply_src"].endswith("past_end")


def test_leak_lines_are_found_at_line_starts_only():
    text = "Nice hand.\n\nBET: $250\n\nPRIVATE: HOT signal -> Bet $250\n  DECODE: question -> $0"
    assert R._leaks(text) == {"PRIVATE": 1, "DECODE": 1}
    assert R._leaks("he said PRIVATE: no") == {}
