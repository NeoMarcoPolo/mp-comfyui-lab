import os

import pytest

from mp_comfyui_lab.executors.base import SearchRequest
from mp_comfyui_lab.executors.runninghub import RunningHubExecutor


pytestmark = [
    pytest.mark.runninghub_readonly,
    pytest.mark.skipif(
        os.environ.get("MP_COMFYUI_LAB_LIVE_DISCOVERY") != "1",
        reason="set MP_COMFYUI_LAB_LIVE_DISCOVERY=1 for public discovery",
    ),
]


def test_public_runninghub_search_and_detail_envelopes() -> None:
    executor = RunningHubExecutor()

    page = executor.search(
        SearchRequest(
            query="qwen",
            sort="REPUTATION",
            page=1,
            size=1,
        )
    )

    assert page.page == 1
    assert page.size >= 1
    assert page.total >= len(page.records) >= 1
    candidate = page.records[0]
    assert candidate.executor == "runninghub"
    assert candidate.workflow_id
    assert candidate.canonical_url.endswith(f"/post/{candidate.workflow_id}")

    detail = executor.detail(
        candidate.workflow_id,
        canonical_url=candidate.canonical_url,
    )

    assert detail.candidate.workflow_id == candidate.workflow_id
    assert detail.candidate.executor == "runninghub"
    assert detail.candidate.canonical_url == candidate.canonical_url
