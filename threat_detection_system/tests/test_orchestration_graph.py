"""
Tests for the threat detection pipeline graph.

Every agent's public function is monkeypatched with a canned return value, so
these tests verify the graph's WIRING (data reaches the right nodes, fan-out/
fan-in runs each node exactly once, the final assessment is built from both
domain scores) rather than re-testing each agent's own logic -- that's already
covered by each agent's own unit tests. `_merge_transcript_turns` is new
orchestration-only logic, so it's also tested directly, in isolation.
"""

from src.orchestration import graph as graph_module
from src.schemas import (
    AgentFinding,
    DiarizationSegment,
    Domain,
    DomainScore,
    Evidence,
    STTWord,
    Severity,
    ThreatAssessment,
    ThreatCategory,
    Transcript,
)


def _finding(agent_name: str, domain: Domain, severity: Severity | None = None) -> AgentFinding:
    return AgentFinding(
        agent_name=agent_name, domain=domain, category=None, severity=severity, confidence=0.8,
        evidence=Evidence(), summary="x",
    )


def _patch_all_agents(monkeypatch, call_counts: dict):
    """Monkeypatch every agent's public function, counting how many times each runs."""

    def counted(name, fn):
        def wrapper(*args, **kwargs):
            call_counts[name] = call_counts.get(name, 0) + 1
            return fn(*args, **kwargs)

        return wrapper

    transcript = Transcript(words=[STTWord(start_ms=0, end_ms=500, text="hello", confidence=0.9)], full_text="hello")
    diarization = [DiarizationSegment(speaker_label="SPEAKER_00", start_ms=0, end_ms=1000, confidence=1.0)]

    monkeypatch.setattr(graph_module.speech_to_text_agent, "transcribe", counted("stt", lambda audio_input: transcript))
    monkeypatch.setattr(graph_module.speaker_diarization_agent, "diarize", counted("diarize", lambda audio_input: diarization))
    monkeypatch.setattr(
        graph_module.background_audio_detection_agent, "detect_background_audio", counted("background", lambda audio_input: [])
    )
    monkeypatch.setattr(
        graph_module.prosody_analysis_agent,
        "analyze_prosody",
        counted("prosody", lambda audio_input, diarization=None: [_finding("prosody", Domain.AUDIO)]),
    )
    monkeypatch.setattr(
        graph_module.emotion_detection_agent, "detect_emotion", counted("emotion", lambda audio_input, diarization=None: [])
    )
    monkeypatch.setattr(
        graph_module.verbal_abuse_detection_agent, "detect_verbal_abuse", counted("verbal_abuse", lambda turns: [])
    )
    monkeypatch.setattr(
        graph_module.threat_detection_agent,
        "detect_threats",
        counted("threat", lambda turns: [_finding("threat", Domain.TRANSCRIPT, Severity.HIGH)]),
    )
    monkeypatch.setattr(
        graph_module.fraud_social_engineering_agent,
        "detect_fraud_and_social_engineering",
        counted("fraud", lambda turns: []),
    )
    monkeypatch.setattr(
        graph_module.compliance_detection_agent, "detect_compliance_violations", counted("compliance", lambda turns: [])
    )
    monkeypatch.setattr(
        graph_module.audio_correlation_agent,
        "correlate_audio_findings",
        counted(
            "audio_corr",
            lambda findings: DomainScore(domain=Domain.AUDIO, score=20.0, summary="audio ok", contributing_findings=findings),
        ),
    )
    monkeypatch.setattr(
        graph_module.transcript_correlation_agent,
        "correlate_transcript_findings",
        counted(
            "transcript_corr",
            lambda findings: DomainScore(
                domain=Domain.TRANSCRIPT, score=75.0, summary="threat found", contributing_findings=findings
            ),
        ),
    )
    monkeypatch.setattr(
        graph_module.threat_correlation_decision_agent,
        "make_threat_assessment",
        counted(
            "decision",
            lambda audio_score, transcript_score: ThreatAssessment(
                audio_domain_score=audio_score,
                transcript_domain_score=transcript_score,
                risk_score=75.0,
                final_category=ThreatCategory.THREAT_OF_VIOLENCE,
                final_severity=Severity.HIGH,
                alert_decision=True,
                decision_summary="alert",
            ),
        ),
    )


def test_pipeline_runs_end_to_end_and_returns_final_assessment(monkeypatch):
    call_counts: dict = {}
    _patch_all_agents(monkeypatch, call_counts)
    graph_module.build_graph.cache_clear()

    assessment = graph_module.run_pipeline(call_id="call-1", audio_path="fake.wav")

    assert assessment.alert_decision is True
    assert assessment.risk_score == 75.0
    assert assessment.audio_domain_score.score == 20.0
    assert assessment.transcript_domain_score.score == 75.0


def test_every_agent_node_runs_exactly_once(monkeypatch):
    call_counts: dict = {}
    _patch_all_agents(monkeypatch, call_counts)
    graph_module.build_graph.cache_clear()

    graph_module.run_pipeline(call_id="call-1", audio_path="fake.wav")

    expected_nodes = {
        "stt", "diarize", "background", "prosody", "emotion", "verbal_abuse",
        "threat", "fraud", "compliance", "audio_corr", "transcript_corr", "decision",
    }
    assert set(call_counts.keys()) == expected_nodes
    assert all(count == 1 for count in call_counts.values())


def test_audio_correlation_receives_findings_from_all_three_audio_agents(monkeypatch):
    call_counts: dict = {}
    _patch_all_agents(monkeypatch, call_counts)
    captured = {}

    def capture_audio_correlation(findings):
        captured["audio_findings"] = findings
        return DomainScore(domain=Domain.AUDIO, score=0.0, summary="x", contributing_findings=findings)

    monkeypatch.setattr(graph_module.audio_correlation_agent, "correlate_audio_findings", capture_audio_correlation)
    graph_module.build_graph.cache_clear()

    graph_module.run_pipeline(call_id="call-1", audio_path="fake.wav")

    # prosody produced 1 finding, emotion 0, background 0 -- exactly 1 total.
    assert len(captured["audio_findings"]) == 1
    assert captured["audio_findings"][0].agent_name == "prosody"


def test_merge_transcript_turns_groups_words_by_diarization_segment():
    transcript = Transcript(
        words=[
            STTWord(start_ms=0, end_ms=400, text="hello", confidence=0.9),
            STTWord(start_ms=400, end_ms=800, text="there", confidence=0.9),
            STTWord(start_ms=1200, end_ms=1600, text="hi", confidence=0.9),
        ],
        full_text="hello there hi",
    )
    diarization = [
        DiarizationSegment(speaker_label="SPEAKER_00", start_ms=0, end_ms=1000, confidence=1.0),
        DiarizationSegment(speaker_label="SPEAKER_01", start_ms=1000, end_ms=2000, confidence=1.0),
    ]

    turns = graph_module._merge_transcript_turns(transcript, diarization)

    assert len(turns) == 2
    assert turns[0].speaker_label == "SPEAKER_00"
    assert turns[0].text == "hello there"
    assert turns[1].speaker_label == "SPEAKER_01"
    assert turns[1].text == "hi"


def test_merge_transcript_turns_falls_back_to_single_turn_when_no_diarization():
    transcript = Transcript(words=[STTWord(start_ms=0, end_ms=400, text="hi", confidence=0.9)], full_text="hi")

    turns = graph_module._merge_transcript_turns(transcript, [])

    assert len(turns) == 1
    assert turns[0].speaker_label == "unknown"
    assert turns[0].text == "hi"
