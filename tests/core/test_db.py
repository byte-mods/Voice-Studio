from oas_core.db import (
    Dataset,
    DatasetVersion,
    Project,
    init_db,
    session_scope,
)
from oas_core.db.models import Modality


def test_create_project_and_dataset() -> None:
    init_db()
    with session_scope() as s:
        p = Project(slug="p1", name="P1")
        s.add(p)
        s.flush()
        d = Dataset(project_id=p.id, slug="d1", name="D1", modality=Modality.ASR)
        s.add(d)
        s.flush()
        v = DatasetVersion(
            dataset_id=d.id,
            version="0.1.0",
            manifest_uri="file:///tmp/m",
            num_samples=10,
        )
        s.add(v)
        s.flush()
        assert v.id
        assert d.versions[0].version == "0.1.0"
