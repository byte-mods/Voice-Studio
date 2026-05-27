from oas_core.db import init_db
from oas_core.db.models import Modality, ModelStage
from oas_core.registry import (
    create_model,
    list_models,
    list_versions,
    publish_version,
    set_stage,
)


def test_create_and_publish(project_id) -> None:
    init_db()
    mid = create_model(project_id, "whisper-en", "Whisper EN", Modality.ASR, family="whisper")
    assert mid
    vid = publish_version(
        mid, "0.1.0", "file:///tmp/x.safetensors", size_bytes=42, metrics={"wer": 0.1}
    )
    assert vid
    versions = list_versions(mid)
    assert len(versions) == 1
    assert versions[0].stage == ModelStage.DEV
    set_stage(vid, ModelStage.STAGING)
    assert list_versions(mid)[0].stage == ModelStage.STAGING
    assert len(list_models(project_id, modality=Modality.ASR)) == 1
