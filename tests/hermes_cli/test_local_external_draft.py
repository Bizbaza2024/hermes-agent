"""An external DFlash draft on a recurrent model: replaces MTP when it fits, on one slot.

Speculative decoding keeps one copy of a recurrent model's state per slot per draft position, so a
depth-7 draft on the router's four slots holds 32 copies (4.7 GiB on Qwen3.8 27B, enough for
llama.cpp's fit to push weights to host memory on Nemotron 3.5 Bolt). One slot holds 8. The draft
is optional: a missing file or an older engine leaves the model on its own MTP heads.
"""

from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import pytest

from hermes_cli.local_runtime import bootstrap, catalog, presets
from hermes_cli.local_runtime.catalog import AssetFile, CatalogEntry, QuantVariant
from hermes_cli.local_runtime.estimator import HardwareBudget
from tests.hermes_cli.test_local_download_jobs import client, poll

GIB = 1 << 30
MIB = 1 << 20
NATIVE = 262144
BUILD = QuantVariant(quant="Q4_K_M", files=(AssetFile("Hybrid-Q4_K_M.gguf", 16 * GIB),))
ENTRY = CatalogEntry(
    id="hybrid", display_name="Hybrid", description="", repo="org/Hybrid-GGUF", variants=(BUILD,),
    n_ctx_train=NATIVE, full_layers=16, recurrent_layers=48, per_layer_f16=4096, mtp=True,
    n_vocab=248320, mtp_draft_depth=2,
    draft=AssetFile("Hybrid-DFlash.gguf", GIB, repo="lab/Hybrid-DFlash-GGUF", revision="0123abc"),
    draft_spec_type="draft-dflash", draft_depth=7, draft_min_engine="b11370")
ROOMY = HardwareBudget(usable_vram_bytes=int(128 * GIB * 0.8), total_device_bytes=128 * GIB,
                       ram_available_bytes=0, uma=True)


@pytest.fixture
def model(tmp_path, monkeypatch):
    """A staged hybrid model; returns (gguf path, draft path, setter for the active entry)."""
    active = {"entry": ENTRY, "state": 150 * MIB}
    monkeypatch.setattr(catalog, "entry_for_model",
                        lambda model_id: active["entry"] if model_id == BUILD.model_id else None)
    monkeypatch.setattr(presets, "read_gguf_header", lambda p: SimpleNamespace(sampling_defaults={}))
    monkeypatch.setattr(presets, "profile_from_gguf", lambda h: replace(
        active["entry"].profile(BUILD), kv_scale=1.0, recurrent_state_bytes=active["state"]))
    draft = bootstrap.assets_dir() / ENTRY.draft.local_name
    draft.parent.mkdir(parents=True, exist_ok=True)
    draft.touch()

    def use(entry=None, *, state=None):
        active["entry"] = entry or active["entry"]
        active["state"] = active["state"] if state is None else state
    return tmp_path / f"{BUILD.model_id}.gguf", draft, use


def _discrete(gib: float) -> HardwareBudget:
    return HardwareBudget(usable_vram_bytes=int(gib * GIB), total_device_bytes=int(gib * GIB),
                          ram_available_bytes=64 * GIB)


def _drafts(preset) -> bool:
    return preset.keys.get("spec-type") == "draft-dflash"


def test_the_draft_replaces_mtp_on_one_slot_when_it_fits(model):
    gguf, draft, _ = model

    preset = presets.preset_for_model(gguf, ROOMY, set(), engine_tag="b11671")

    assert preset.window == NATIVE and not preset.spilled
    assert preset.keys["spec-type"] == "draft-dflash"
    assert preset.keys["spec-draft-n-max"] == "7"
    assert preset.keys["model-draft"] == str(draft)
    assert preset.keys["parallel"] == "1"


def test_a_missing_draft_or_an_older_engine_keeps_mtp(model):
    gguf, draft, _ = model

    for tag in (None, "b11000"):
        preset = presets.preset_for_model(gguf, ROOMY, set(), engine_tag=tag)
        assert preset.keys["spec-type"] == "draft-mtp" and "parallel" not in preset.keys
    draft.unlink()
    preset = presets.preset_for_model(gguf, ROOMY, set(), engine_tag="b11671")
    assert preset.keys["spec-type"] == "draft-mtp" and "model-draft" not in preset.keys


def test_the_draft_may_shrink_the_window_to_half_native_and_no_further(model):
    """Each budget either keeps the MTP preset exactly, or drafts resident at no less than the
    smaller of the MTP window and half the native window. Growth past it drops the draft."""
    gguf, _, _ = model
    shrank = declined_with_room = 0
    for tenths in range(160, 400, 2):
        budget = _discrete(tenths / 10)
        mtp = presets.preset_for_model(gguf, budget, set())
        preset = presets.preset_for_model(gguf, budget, set(), engine_tag="b11671")
        if mtp.refusal:
            continue
        if _drafts(preset):
            assert not preset.spilled
            assert preset.window >= min(mtp.window, NATIVE // 2)
            assert preset.keys["parallel"] == "1"
            shrank += preset.window < mtp.window
        else:
            assert preset == mtp
            declined_with_room += not mtp.spilled
    assert shrank and declined_with_room


def test_a_grown_window_wins_over_the_draft(model):
    """Growth restored the native window on a roomy card, and other programs now hold part of it:
    whatever window still fits goes to the model as it was, not to the draft."""
    gguf, _, _ = model
    live = next(b for b in (_discrete(t / 10) for t in range(160, 400, 2))
                if _drafts(p := presets.preset_for_model(gguf, b, set(), engine_tag="b11671"))
                and p.window < presets.preset_for_model(gguf, b, set()).window)
    mtp = presets.preset_for_model(gguf, ROOMY, set(), requested_window=NATIVE, live=live)

    grown = presets.preset_for_model(gguf, ROOMY, set(), requested_window=NATIVE, live=live,
                                     engine_tag="b11671")

    assert grown == mtp and grown.keys["spec-type"] == "draft-mtp"


def test_each_draft_position_prices_a_copy_of_the_recurrent_state(model):
    """The same card that fits the draft beside 150 MiB of state per copy declines it at 1 GiB
    per copy: eight copies are 7 GiB more."""
    gguf, _, use = model
    budget = _discrete(32)
    assert _drafts(presets.preset_for_model(gguf, budget, set(), engine_tag="b11671"))

    use(state=GIB)

    assert not _drafts(presets.preset_for_model(gguf, budget, set(), engine_tag="b11671"))


def test_a_spilled_model_never_takes_the_draft(model):
    gguf, _, _ = model

    preset = presets.preset_for_model(gguf, _discrete(12), set(), engine_tag="b11671")

    assert preset.spilled and not _drafts(preset)


def test_a_model_without_mtp_drafts_without_backend_sampling(model):
    """Bolt's shape: no MTP heads, so nothing to replace and no backend sampling to keep."""
    gguf, _, use = model
    use(replace(ENTRY, mtp=False, draft_min_engine="b11671"))

    preset = presets.preset_for_model(gguf, ROOMY, set(), engine_tag="b11671")

    assert _drafts(preset) and preset.keys["parallel"] == "1"
    assert "backend-sampling" not in preset.keys


def test_the_residency_price_includes_the_draft_once_it_is_on_disk(model):
    gguf, draft, _ = model
    with_draft = presets._launch_footprint(gguf, ROOMY)
    draft.unlink()

    assert with_draft >= presets._launch_footprint(gguf, ROOMY) + ENTRY.draft.size_bytes


def _staged(monkeypatch, *entries):
    from hermes_cli.web_routers import local_models as lm

    monkeypatch.setattr(lm.catalog, "CATALOG", entries)
    monkeypatch.setattr(lm.bootstrap, "staged_model_ids", lambda: [e.variants[0].model_id for e in entries])
    return lm


def test_installed_models_missing_their_draft_get_it_from_the_engine_update(monkeypatch):
    """A Spark that downloaded the 27B before its draft joined the catalog: the update the app
    already offers fetches the draft. Models not on disk, drafts already on disk, and drafts the
    engine can't load are left alone."""
    unstaged = replace(ENTRY, id="other", variants=(replace(BUILD, files=(AssetFile("Other.gguf", GIB),)),))
    newer = replace(ENTRY, id="newer", variants=(replace(BUILD, files=(AssetFile("Newer.gguf", GIB),)),),
                    draft=AssetFile("Newer-DFlash.gguf", GIB), draft_min_engine="b99999")
    lm = _staged(monkeypatch, ENTRY, newer)
    monkeypatch.setattr(lm.catalog, "CATALOG", (ENTRY, newer, unstaged))
    dest = lm.bootstrap.assets_dir() / ENTRY.draft.local_name

    plan = lm._missing_drafts("b11671")

    assert plan == [("https://huggingface.co/lab/Hybrid-DFlash-GGUF/resolve/0123abc/Hybrid-DFlash.gguf",
                     dest, GIB)]
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.touch()
    assert lm._missing_drafts("b11671") == []


def test_the_update_job_downloads_missing_drafts_then_restarts_the_server(client, monkeypatch):
    lm = _staged(monkeypatch, ENTRY)
    engine = SimpleNamespace(tag="b11671", backend="cpu")
    monkeypatch.setattr(lm, "_runtime_target", lambda backend=None: ("b11671", "cpu"))
    monkeypatch.setattr(lm.binaries, "installed_engine", lambda backend: engine)
    monkeypatch.setattr(lm, "_install_engine_job", lambda job, backend: engine)
    fetched, events = [], []
    monkeypatch.setattr(lm, "_download_job", lambda job, plan: fetched.extend(plan))
    monkeypatch.setattr(lm.bootstrap, "get_supervisor", lambda: object())
    monkeypatch.setattr(lm.bootstrap, "refresh_local_runtime", lambda: events.append("restart"))

    job_id = client.post("/api/local-models/runtime/install", json={"backend": "cpu"}).json()["job_id"]

    assert poll(client, job_id, until=("done", "error"))["status"] == "done"
    assert [dest.name for _, dest, _ in fetched] == [ENTRY.draft.local_name]
    assert events == ["restart"]


def test_an_installed_model_without_its_draft_still_reads_as_downloaded(monkeypatch):
    lm = _staged(monkeypatch, ENTRY)
    model = lm.bootstrap.models_dir() / BUILD.files[0].local_name
    model.parent.mkdir(parents=True, exist_ok=True)
    model.touch()

    row = lm._catalog_row(ENTRY, ROOMY, None, None, {BUILD.model_id})

    assert row["downloaded"] and row["size_bytes"] == ENTRY.download_bytes(BUILD)
