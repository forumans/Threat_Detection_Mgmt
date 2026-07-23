"""
Tests for the synthetic dataset generation pipeline graph.

Every agent's public function is monkeypatched with a canned return value, so
these tests verify the graph's WIRING (data reaches the right nodes, fan-out/
fan-in runs each node exactly once, generate_dataset assembles a full batch
correctly) rather than re-testing each agent's own logic -- that's already
covered by each agent's own unit tests.
"""

from src.orchestration import graph as graph_module
from src.schemas import (
    AudioFile,
    CallMetadata,
    Category,
    ChannelQuality,
    Configuration,
    Conversation,
    ConversationTurn,
    ConversationTurnPlan,
    DatasetManifest,
    Persona,
    SampleRef,
    Scenario,
)


def _configuration(sample_count: int = 2) -> Configuration:
    return Configuration(
        config_id="cfg-1", category_distribution={"benign": 1.0}, sample_count=sample_count, locales=["en-US"], seed=1
    )


def _scenario() -> Scenario:
    return Scenario(
        scenario_id="scn-1", category=Category.BENIGN, severity=None, setting="x", locale="en-US",
        channel_quality=ChannelQuality.CLEAN, num_speakers=2,
    )


def _personas() -> list[Persona]:
    return [
        Persona(persona_id="p-caller", role="caller", voice_traits={"name": "Alex"}, emotional_baseline="calm"),
        Persona(persona_id="p-agent", role="agent", voice_traits={"name": "Sam"}, emotional_baseline="calm"),
    ]


def _conversation() -> Conversation:
    return Conversation(
        conversation_id="conv-1",
        scenario_id="scn-1",
        turns=[
            ConversationTurnPlan(turn_index=0, speaker_persona_id="p-caller", intended_content="hi", intended_emotion="neutral")
        ],
        injected_threat_turn_indices=[],
    )


def _transcript() -> list[ConversationTurn]:
    return [ConversationTurn(turn_index=0, speaker_persona_id="p-caller", text="hi", intended_emotion="neutral")]


def _metadata() -> CallMetadata:
    return CallMetadata(
        call_id="conv-1", scenario_id="scn-1", start_timestamp="2024-01-01T00:00:00", duration_ms=1000,
        channel_info={}, participant_ids=["a", "b"], locale="en-US",
    )


def _audio_file() -> AudioFile:
    return AudioFile(sample_id="conv-1", path="fake.wav", duration_ms=1000, sample_rate_hz=16000, channel_quality=ChannelQuality.CLEAN)


def _patch_all_agents(monkeypatch, call_counts: dict):
    """Monkeypatch every agent's public function, counting how many times each runs."""

    def counted(name, fn):
        def wrapper(*args, **kwargs):
            call_counts[name] = call_counts.get(name, 0) + 1
            return fn(*args, **kwargs)

        return wrapper

    monkeypatch.setattr(
        graph_module.scenario_generator_agent, "generate_scenario",
        counted("scenario", lambda configuration, sample_index=0: _scenario()),
    )
    monkeypatch.setattr(
        graph_module.persona_generator_agent, "generate_personas", counted("personas", lambda scenario: _personas())
    )
    monkeypatch.setattr(
        graph_module.conversation_generator_agent, "generate_conversation",
        counted("conversation", lambda scenario, personas: _conversation()),
    )
    monkeypatch.setattr(
        graph_module.transcript_generator_agent, "generate_transcript",
        counted("transcript", lambda conversation, personas: _transcript()),
    )
    monkeypatch.setattr(
        graph_module.translation_engine_agent, "translate_transcript",
        counted("translation", lambda turns, source_locale, target_locale: turns),
    )
    monkeypatch.setattr(
        graph_module.tts_engine_agent, "synthesize_turns", counted("tts", lambda turns, personas: ["fake-clip"])
    )
    monkeypatch.setattr(
        graph_module.ground_truth_generator_agent, "generate_ground_truth",
        counted("ground_truth", lambda conversation, scenario: []),
    )
    monkeypatch.setattr(
        graph_module.metadata_generator_agent, "generate_metadata",
        counted("metadata", lambda conversation, scenario: _metadata()),
    )
    monkeypatch.setattr(
        graph_module.audio_gen, "generate_audio",
        counted(
            "audio",
            lambda clips, scenario, sample_id, ground_truth_labels, output_path: (_audio_file(), ground_truth_labels),
        ),
    )
    monkeypatch.setattr(
        graph_module.dataset_exporter_agent, "export_sample",
        counted(
            "export",
            lambda sample_id, audio_file, transcript, ground_truth, metadata, output_dir: SampleRef(
                sample_id=sample_id, audio_path="a", transcript_path="t", ground_truth_path="g", metadata_path="m"
            ),
        ),
    )


def test_per_sample_pipeline_runs_end_to_end(monkeypatch, tmp_path):
    call_counts: dict = {}
    _patch_all_agents(monkeypatch, call_counts)
    graph_module.build_graph.cache_clear()

    sample_ref, scenario = graph_module.generate_sample(_configuration(), sample_index=0, output_dir=tmp_path)

    assert sample_ref.sample_id == "conv-1"
    assert scenario.scenario_id == "scn-1"


def test_every_node_in_per_sample_graph_runs_exactly_once(monkeypatch, tmp_path):
    call_counts: dict = {}
    _patch_all_agents(monkeypatch, call_counts)
    graph_module.build_graph.cache_clear()

    graph_module.generate_sample(_configuration(), sample_index=0, output_dir=tmp_path)

    expected_nodes = {
        "scenario", "personas", "conversation", "transcript", "translation",
        "tts", "ground_truth", "metadata", "audio", "export",
    }
    assert set(call_counts.keys()) == expected_nodes
    assert all(count == 1 for count in call_counts.values())


def test_audio_node_receives_both_tts_clips_and_ground_truth(monkeypatch, tmp_path):
    call_counts: dict = {}
    _patch_all_agents(monkeypatch, call_counts)
    captured = {}

    def capture_audio(clips, scenario, sample_id, ground_truth_labels, output_path):
        captured["clips"] = clips
        captured["ground_truth_labels"] = ground_truth_labels
        return _audio_file(), ground_truth_labels

    monkeypatch.setattr(graph_module.audio_gen, "generate_audio", capture_audio)
    graph_module.build_graph.cache_clear()

    graph_module.generate_sample(_configuration(), sample_index=0, output_dir=tmp_path)

    assert captured["clips"] == ["fake-clip"]
    assert captured["ground_truth_labels"] == []


def test_generate_dataset_runs_one_sample_per_configured_count(monkeypatch, tmp_path):
    call_counts: dict = {}
    _patch_all_agents(monkeypatch, call_counts)
    graph_module.build_graph.cache_clear()
    manifest_calls = {}

    def fake_write_manifest(dataset_version, sample_refs, scenarios, output_dir):
        manifest_calls["sample_refs"] = sample_refs
        manifest_calls["scenarios"] = scenarios
        return DatasetManifest(
            dataset_version=dataset_version, generated_at="2024-01-01T00:00:00", samples=sample_refs, coverage_report={}
        )

    monkeypatch.setattr(graph_module.dataset_exporter_agent, "write_manifest", fake_write_manifest)
    monkeypatch.setattr(
        graph_module.configuration_agent, "build_configuration", lambda *args, **kwargs: _configuration(sample_count=3)
    )

    graph_module.generate_dataset(
        category_distribution={"benign": 1.0}, sample_count=3, locales=["en-US"], dataset_version="v1", output_dir=tmp_path
    )

    assert len(manifest_calls["sample_refs"]) == 3
    assert len(manifest_calls["scenarios"]) == 3
    assert call_counts["scenario"] == 3  # generate_scenario ran once per requested sample
