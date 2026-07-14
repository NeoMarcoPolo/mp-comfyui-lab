import json
import hashlib
import os
from pathlib import Path
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier, Event
from time import perf_counter, sleep

import pytest

from mp_comfyui_lab import security as security_module
from mp_comfyui_lab.experiments import store as store_module
from mp_comfyui_lab.experiments.store import ExperimentStore
from mp_comfyui_lab.security import (
    SafetyError,
    safe_filename,
    sanitize_embedded_urls,
    sanitize_for_persistence,
    sanitize_untrusted_diagnostic,
)


EXPECTED_LAYOUT = {
    "sources",
    "evidence",
    "plans",
    "variants",
    "runs",
    "outputs",
    "export",
}
NORMALIZED_WINDOWS_DEVICE_NAMES = (
    "NUL .json",
    "CON .txt",
    "PRN .txt",
    "AUX .json",
    "COM1 .txt",
    "COM¹.json",
    "COM².txt",
    "COM³",
    "LPT¹.json",
    "LPT².txt",
    "LPT³",
)


def test_store_creates_complete_layout_and_redacts_secrets(tmp_path):
    store = ExperimentStore(tmp_path)
    root = store.create("qwen-angle-001")
    store.write_json(
        "qwen-angle-001",
        "runs/run-1.json",
        {"apiKey": "secret", "taskId": "42"},
    )
    payload = json.loads((root / "runs/run-1.json").read_text(encoding="utf-8"))
    assert payload == {"apiKey": "[REDACTED]", "taskId": "42"}
    assert EXPECTED_LAYOUT <= {
        path.name for path in root.iterdir() if path.is_dir()
    }


def test_new_artifact_writes_never_clobber_an_existing_destination(tmp_path):
    store = ExperimentStore(tmp_path)
    store.create("no-clobber-001")
    store.write_text_new("no-clobber-001", "plans/one.json", "first")

    with pytest.raises(SafetyError, match="refusing to overwrite"):
        store.write_text_new("no-clobber-001", "plans/one.json", "second")

    assert (tmp_path / "no-clobber-001" / "plans" / "one.json").read_text(encoding="utf-8") == "first"


def test_store_root_is_resolved(tmp_path):
    store = ExperimentStore(tmp_path / "parent" / ".." / "runs")

    assert store.root("abc") == (tmp_path / "runs" / "abc").resolve()


def test_create_unique_allocates_deterministic_suffixes_without_overwrite(tmp_path):
    store = ExperimentStore(tmp_path)

    allocated = [store.create_unique("collision-001") for _ in range(3)]

    assert allocated == ["collision-001", "collision-001-2", "collision-001-3"]
    assert all((tmp_path / experiment_id).is_dir() for experiment_id in allocated)


def test_create_unique_keeps_windows_safe_length_when_base_id_is_maximum(tmp_path):
    store = ExperimentStore(tmp_path)
    base_id = "a" * 96

    first = store.create_unique(base_id)
    second = store.create_unique(base_id)

    assert first == base_id
    assert second.endswith("-2")
    assert len(second) == 96
    assert store.root(second).is_dir()


def test_create_unique_retries_concurrent_directory_races(tmp_path):
    store = ExperimentStore(tmp_path)
    workers = 8
    barrier = Barrier(workers)

    def allocate() -> str:
        barrier.wait()
        return store.create_unique("parallel-001")

    with ThreadPoolExecutor(max_workers=workers) as pool:
        allocated = list(pool.map(lambda _index: allocate(), range(workers)))

    assert set(allocated) == {
        "parallel-001",
        *(f"parallel-001-{index}" for index in range(2, workers + 1)),
    }
    assert all(
        EXPECTED_LAYOUT <= {path.name for path in (tmp_path / item).iterdir()}
        for item in allocated
    )


@pytest.mark.parametrize(
    "experiment_id",
    [
        "ab",
        "Uppercase",
        "-starts-with-dash",
        "contains/slash",
        "contains space",
        "abc.",
        "con",
        "aux",
        "nul",
        "com1",
        "lpt9",
        "a" * 97,
    ],
)
def test_store_rejects_invalid_experiment_ids(tmp_path, experiment_id):
    store = ExperimentStore(tmp_path)

    with pytest.raises(SafetyError, match="invalid experiment ID"):
        store.root(experiment_id)


def test_sanitize_for_persistence_redacts_nested_secrets_without_mutation():
    value = {
        "Authorization": "Bearer secret",
        "nested": [
            {"refresh_TOKEN": "refresh-secret"},
            ({"Cookie": "session-secret"},),
        ],
    }

    sanitized = sanitize_for_persistence(value)

    assert sanitized == {
        "Authorization": "[REDACTED]",
        "nested": [
            {"refresh_TOKEN": "[REDACTED]"},
            ({"Cookie": "[REDACTED]"},),
        ],
    }
    assert value["Authorization"] == "Bearer secret"
    assert value["nested"][0]["refresh_TOKEN"] == "refresh-secret"
    assert value["nested"][1][0]["Cookie"] == "session-secret"


def test_graph_persistence_sanitizer_redacts_credentials_and_keeps_material_query():
    graph = {
        "1": {
            "class_type": "RemoteModel",
            "inputs": {
                "password": "password-secret",
                "auth_token": "auth-secret",
                "apiKey": "api-secret",
                "prompt": "A secret garden with a password reset sign",
                "aws_url": (
                    "https://models.example/base?revision=1&format=fp16&"
                    "X-Amz-Credential=aws-secret&X-Amz-Signature=aws-signature"
                ),
                "azure_url": (
                    "https://account.blob.core.windows.net/models/base"
                    "?revision=1&sv=2025-01-05&sp=r&se=2030-01-01&sig=azure-secret"
                ),
            },
        }
    }

    sanitized = security_module.sanitize_graph_for_persistence(graph)

    inputs = sanitized["1"]["inputs"]
    assert inputs["password"] == "[REDACTED]"
    assert inputs["auth_token"] == "[REDACTED]"
    assert inputs["apiKey"] == "[REDACTED]"
    assert inputs["prompt"] == "A secret garden with a password reset sign"
    assert inputs["aws_url"] == (
        "https://models.example/base?revision=1&format=fp16"
    )
    assert inputs["azure_url"] == (
        "https://account.blob.core.windows.net/models/base?revision=1"
    )
    assert graph["1"]["inputs"]["password"] == "password-secret"
    assert "aws-secret" in graph["1"]["inputs"]["aws_url"]


def test_graph_persistence_sanitizer_removes_url_authority_and_fragment_secrets():
    graph = {
        "direct": (
            "https://alice:password-secret@[2001:db8::1]:8443/model"
            "?revision=1&access_token=query-secret"
            "#section=weights&password=fragment-secret&format=fp16"
        ),
        "embedded": (
            "load https://bob:authority-secret@models.example/base"
            "?revision=1#tab=files&auth_token=fragment-token now"
        ),
        "malformed_material": "https://models.example:notaport/base?revision=1",
        "malformed_secret": (
            "https://models.example:notaport/base?revision=1&access_token=hidden"
        ),
        "encoded_fragment": (
            "https://models.example/base?revision=1"
            "#access_token%3Dencoded-secret%26section%3Dweights"
        ),
    }

    sanitized = security_module.sanitize_graph_for_persistence(graph)

    assert sanitized == {
        "direct": (
            "https://[2001:db8::1]:8443/model?revision=1"
        ),
        "embedded": (
            "load https://models.example/base?revision=1 now"
        ),
        "malformed_material": "https://models.example:notaport/base?revision=1",
        "malformed_secret": "https://models.example:notaport/base?revision=1",
        "encoded_fragment": "https://models.example/base?revision=1",
    }
    serialized = json.dumps(sanitized, sort_keys=True)
    for secret in (
        "alice",
        "password-secret",
        "query-secret",
        "fragment-secret",
        "bob",
        "authority-secret",
        "fragment-token",
        "hidden",
        "encoded-secret",
        "access_token",
        "auth_token",
    ):
        assert secret not in serialized
    assert security_module.sanitize_graph_for_persistence(sanitized) == sanitized


@pytest.mark.parametrize(
    "fragment",
    [
        "section=weights&access_token=secret",
        "section=weights;AcCeSsToKeN=secret",
        "section=weights,apiKey:secret",
        "section=weights auth_token secret",
        "section=weights;clientSecret secret",
        "authorization:Basic secret",
        "Bearer secret",
        "section%3Dweights%3BAccessToken%3Dencoded-secret",
        "section%3Dweights%26access%5Ftoken%3Dencoded-underscore-secret",
    ],
)
def test_graph_fragment_drops_entire_value_for_any_secret_marker(fragment):
    url = f"https://models.example/base?revision=1#{fragment}"

    sanitized = security_module.sanitize_graph_for_persistence(url)

    assert sanitized == "https://models.example/base?revision=1"
    assert security_module.sanitize_graph_for_persistence(sanitized) == sanitized


@pytest.mark.parametrize(
    "fragment",
    [
        "clean-anchor",
        "section=weights;revision=1",
        "section%3Dweights%3Brevision%3D1",
        "password-reset",
    ],
)
def test_graph_fragment_preserves_clean_material_value_byte_for_byte(fragment):
    url = f"https://models.example/base?revision=1#{fragment}"

    sanitized = security_module.sanitize_graph_for_persistence(url)

    assert sanitized == url
    assert security_module.sanitize_graph_for_persistence(sanitized) == sanitized


@pytest.mark.parametrize(
    "fragment",
    [
        "access_token%253Ddouble-secret",
        "access_token%25253Dtriple-secret",
        "access%255Ftoken%253Dencoded-key-secret",
        "section%253Dweights%2526access_token%253Dmixed-secret",
        "section%3Dweights%26access%255Ftoken%253Dmixed-layers",
        "clean%252Danchor",
    ],
)
def test_graph_fragment_drops_nested_percent_encoding(fragment):
    url = f"https://models.example/base?revision=1#{fragment}"

    sanitized = security_module.sanitize_graph_for_persistence(url)

    assert sanitized == "https://models.example/base?revision=1"
    assert security_module.sanitize_graph_for_persistence(sanitized) == sanitized


def test_graph_fragment_decoding_is_bounded_fail_closed():
    fragment = "clean-" + ("a" * 70_000)
    url = f"https://models.example/base?revision=1#{fragment}"

    assert security_module.sanitize_graph_for_persistence(url) == (
        "https://models.example/base?revision=1"
    )


def test_sanitize_for_persistence_strips_any_http_url_query():
    value = {
        "outputUrl": (
            "https://cdn.example.test/frame.png?view=public&size=large#preview"
        )
    }

    assert sanitize_for_persistence(value) == {
        "outputUrl": "https://cdn.example.test/frame.png#preview"
    }


def test_sanitize_for_persistence_strips_azure_sas_query():
    value = {
        "blobUrl": (
            "https://account.blob.core.windows.net/container/frame.png"
            "?sv=2025-01-05&sp=r&se=2030-01-01T00:00:00Z&sig=secret"
            "#preview"
        )
    }

    assert sanitize_for_persistence(value) == {
        "blobUrl": (
            "https://account.blob.core.windows.net/container/frame.png#preview"
        )
    }


def test_sanitize_for_persistence_strips_url_query_with_blank_value():
    value = {"outputUrl": "https://cdn.example.test/frame.png?TOKEN="}

    assert sanitize_for_persistence(value) == {
        "outputUrl": "https://cdn.example.test/frame.png"
    }


def test_sanitize_for_persistence_strips_signed_urls_in_collection():
    value = {"outputUrls": ["https://cdn/x?token=secret"]}

    assert sanitize_for_persistence(value) == {"outputUrls": ["https://cdn/x"]}


def test_sanitize_for_persistence_propagates_url_context_without_mutation():
    value = {
        "previewUrls": [
            {
                "items": (
                    "https://cdn/x?signature=secret",
                    ["https://cdn/stable?view=public"],
                )
            }
        ]
    }

    sanitized = sanitize_for_persistence(value)

    assert sanitized == {
        "previewUrls": [
            {
                "items": (
                    "https://cdn/x",
                    ["https://cdn/stable"],
                )
            }
        ]
    }
    assert value["previewUrls"][0]["items"][0].endswith("?signature=secret")
    assert value["previewUrls"][0]["items"][1][0].endswith("?view=public")


def test_sanitize_for_persistence_preserves_url_without_query():
    value = {"source_url": "https://example.test/workflows/42#source"}

    assert sanitize_for_persistence(value) == value


def test_sanitize_for_persistence_leaves_non_url_strings_unchanged():
    value = {
        "source_url": "not a URL?sig=plain-text",
        "outputUrls": [
            "plain text?view=public",
            "ftp://files.example.test/frame.png?token=not-http",
        ],
    }

    assert sanitize_for_persistence(value) == value


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (
            "https://cdn.example/model.safetensors?X-Amz-Signature=aws-secret",
            "https://cdn.example/model.safetensors",
        ),
        (
            "model=https://cdn.example/model.safetensors?sig=secret; format=fp16",
            "model=https://cdn.example/model.safetensors; format=fp16",
        ),
        (
            "[model](https://cdn.example/model.safetensors?sv=1&sig=secret#card).",
            "[model](https://cdn.example/model.safetensors#card).",
        ),
        (
            "first=https://one.example/a?token=secret, "
            "second=https://two.example/b?sv=1&sig=secret!",
            "first=https://one.example/a, second=https://two.example/b!",
        ),
        (
            "first=https://one.example/a?sig=one,"
            "second=https://two.example/b?sig=two!",
            "first=https://one.example/a,second=https://two.example/b!",
        ),
        (
            "one=https://one.example/a?sig=one|"
            "[two](https://two.example/b?sig=two#preview).",
            "one=https://one.example/a|"
            "[two](https://two.example/b#preview).",
        ),
        (
            "https://one.example/a?sig=one "
            "https://two.example/b?sig=two",
            "https://one.example/a https://two.example/b",
        ),
        (
            "clean=https://one.example/a, "
            "signed=https://two.example/b?sig=two; "
            "fragment=https://three.example/c#card.",
            "clean=https://one.example/a, signed=https://two.example/b; "
            "fragment=https://three.example/c#card.",
        ),
        (
            "first=https://one.example/a?sig=one;"
            "second=https://two.example/b?sig=two",
            "first=https://one.example/a;second=https://two.example/b",
        ),
        (
            "redirect=https://one.example/path"
            "?next=https://two.example/callback?sig=two",
            "redirect=https://one.example/path",
        ),
        (
            "redirect=https://one.example/path"
            "?mode=public&next=https://two.example/callback?sig=two",
            "redirect=https://one.example/path",
        ),
        (
            "query-free [model](https://cdn.example/model.safetensors#card), "
            "plain text",
            "query-free [model](https://cdn.example/model.safetensors#card), "
            "plain text",
        ),
        (
            "plain-model_lora_v1.safetensors",
            "plain-model_lora_v1.safetensors",
        ),
    ],
)
def test_sanitize_embedded_urls_preserves_surrounding_text_and_punctuation(
    value,
    expected,
):
    assert sanitize_embedded_urls(value) == expected


def test_sanitize_embedded_urls_is_idempotent_for_adjacent_descriptors():
    value = (
        "first=https://one.example/a?sig=one,"
        "second=https://two.example/b?sig=two|"
        "[third](https://three.example/c?sig=three#card)."
    )

    sanitized = sanitize_embedded_urls(value)

    assert sanitize_embedded_urls(sanitized) == sanitized


def test_sanitize_for_persistence_strips_urls_embedded_in_plain_text():
    value = {
        "descriptor": (
            "model=https://cdn.example/model.safetensors?sig=secret; format=fp16"
        )
    }

    assert sanitize_for_persistence(value) == {
        "descriptor": "model=https://cdn.example/model.safetensors; format=fp16"
    }


def test_sanitize_for_persistence_redacts_canonical_secret_key_aliases():
    value = {
        "Access-Token": "access-secret",
        "refresh_token": "refresh-secret",
        "clientSecret": "client-secret",
        "signedUrl": "https://cdn.example/model?sig=signed-secret",
        "SiGnAtUrE": "signature-secret",
        "credential": "credential-secret",
        "sig": "sig-secret",
        "api-key": "dash-secret",
        "api_key": "underscore-secret",
        "apikey": "compact-secret",
        "X-API-KEY": "header-dash-secret",
        "x_api_key": "header-underscore-secret",
        "X-Amz-Credential": "aws-credential-secret",
        "x_amz_signature": "aws-signature-secret",
        "Authorization": "Basic authorization-secret",
        "Cookies": "cookie-secret",
        "TOKEN": "token-secret",
        "signatureVersion": "v4",
        "client_secret_hint": "public-name",
    }

    sanitized = sanitize_for_persistence(value)

    for key in value.keys() - {"signatureVersion", "client_secret_hint"}:
        assert sanitized[key] == "[REDACTED]"
    assert sanitized["signatureVersion"] == "v4"
    assert sanitized["client_secret_hint"] == "public-name"


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (
            "Authorization: Bearer bearer-secret",
            "Authorization: [REDACTED]",
        ),
        (
            "Authorization: Basic base64-secret; retry later",
            "Authorization: [REDACTED]; retry later",
        ),
        (
            "authorization=Token token-secret, retry later",
            "authorization=[REDACTED], retry later",
        ),
        (
            "AUTHORIZATION: Custom custom-secret material | retry later",
            "AUTHORIZATION: [REDACTED] | retry later",
        ),
        (
            "Authorization: [REDACTED]",
            "Authorization: [REDACTED]",
        ),
        (
            "Authorization: Bearer [REDACTED]",
            "Authorization: [REDACTED]",
        ),
        (
            "upstream rejected Bearer abc.def-123",
            "upstream rejected Bearer [REDACTED]",
        ),
        (
            'Bearer "bearer-secret"; Basic \'basic-secret\'; Token token-secret',
            'Bearer "[REDACTED]"; Basic \'[REDACTED]\'; Token [REDACTED]',
        ),
        (
            "cookie='session-secret'; token=token-secret; provider unavailable",
            "cookie='[REDACTED]'; token=[REDACTED]; provider unavailable",
        ),
        (
            "SiGnAtUrE='signature-secret'; CREDENTIAL=credential-secret; "
            'X-API-KEY="header-secret"; client_secret=client-secret; retry later',
            "SiGnAtUrE='[REDACTED]'; CREDENTIAL=[REDACTED]; "
            'X-API-KEY="[REDACTED]"; client_secret=[REDACTED]; retry later',
        ),
        (
            "xAmzCredential=aws-credential-secret; "
            "X_AMZ_SIGNATURE='aws-signature-secret'; "
            "Access-Token=access-secret; RefreshToken=refresh-secret",
            "xAmzCredential=[REDACTED]; X_AMZ_SIGNATURE='[REDACTED]'; "
            "Access-Token=[REDACTED]; RefreshToken=[REDACTED]",
        ),
        (
            "asset=https://cdn.example/failure?sv=1&sig=query-secret, retry later",
            "asset=https://cdn.example/failure, retry later",
        ),
        (
            '{"message":"provider unavailable","apiKey":"json-secret",'
            '"nested":{"token":"nested-secret"},'
            '"url":"https://cdn.example/failure?sig=query-secret"}',
            '{"apiKey":"[REDACTED]","message":"provider unavailable",'
            '"nested":{"token":"[REDACTED]"},'
            '"url":"https://cdn.example/failure"}',
        ),
    ],
)
def test_sanitize_untrusted_diagnostic_redacts_credentials(value, expected):
    assert sanitize_untrusted_diagnostic(value) == expected


def test_sanitize_untrusted_diagnostic_preserves_redacted_aws_authorization():
    raw = (
        "AWS4-HMAC-SHA256 Credential=aws-credential-secret, "
        "SignedHeaders=host;x-amz-date, Signature=aws-signature-secret"
    )
    expected = (
        "AWS4-HMAC-SHA256 Credential=[REDACTED], "
        "SignedHeaders=host;x-amz-date, Signature=[REDACTED]"
    )

    sanitized = sanitize_untrusted_diagnostic(raw)

    assert sanitized == expected
    assert sanitize_untrusted_diagnostic(sanitized) == expected
    assert "aws-credential-secret" not in sanitized
    assert "aws-signature-secret" not in sanitized


def test_sanitize_untrusted_diagnostic_redacts_exact_json_aliases_stably():
    value = {
        "accessToken": "access-secret",
        "Refresh-Token": "refresh-secret",
        "clientSecret": "client-secret",
        "signedUrl": "https://cdn.example/model?sig=signed-secret",
        "SiGnAtUrE": "signature-secret",
        "credential": "credential-secret",
        "sig": "sig-secret",
        "api-key": "dash-secret",
        "api_key": "underscore-secret",
        "apikey": "compact-secret",
        "X-API-KEY": "header-dash-secret",
        "x_api_key": "header-underscore-secret",
        "X-Amz-Credential": "aws-credential-secret",
        "x_amz_signature": "aws-signature-secret",
        "Authorization": "Token authorization-secret",
        "Cookies": "cookie-secret",
        "TOKEN": "token-secret",
        "signatureVersion": "v4",
        "message": "provider unavailable",
    }
    raw = json.dumps(value)

    sanitized = sanitize_untrusted_diagnostic(raw)

    expected = {
        key: (
            original
            if key in {"signatureVersion", "message"}
            else "[REDACTED]"
        )
        for key, original in value.items()
    }
    assert json.loads(sanitized) == expected
    assert sanitized == json.dumps(
        expected,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )


@pytest.mark.parametrize(
    "name",
    [
        "",
        ".",
        "..",
        "../secret.json",
        "nested/file.json",
        "nested\\file.json",
        "bad<name.json",
        "bad>name.json",
        "bad:name.json",
        'bad"name.json',
        "bad|name.json",
        "bad?name.json",
        "bad*name.json",
        "trailing.",
        "trailing ",
        "CON",
        "CON.txt",
        "bad\x00name",
        "bad\x85name",
    ],
)
def test_safe_filename_rejects_unsafe_names(name):
    with pytest.raises(SafetyError):
        safe_filename(name)


def test_safe_filename_preserves_safe_unicode_name():
    assert safe_filename("comparison-日本語.json") == "comparison-日本語.json"


@pytest.mark.parametrize("name", NORMALIZED_WINDOWS_DEVICE_NAMES)
def test_safe_filename_rejects_normalized_windows_device_names(name):
    with pytest.raises(SafetyError):
        safe_filename(name)


@pytest.mark.parametrize("name", NORMALIZED_WINDOWS_DEVICE_NAMES)
def test_store_rejects_normalized_windows_device_name_before_write(
    tmp_path,
    monkeypatch,
    name,
):
    store = ExperimentStore(tmp_path)
    root = store.create("safe-001")
    destination = root / "runs" / name
    monkeypatch.setattr(
        store,
        "_atomic_write",
        lambda *args: pytest.fail("unsafe name reached atomic write"),
    )

    with pytest.raises(SafetyError):
        store.write_text("safe-001", Path("runs") / name, "unsafe")

    assert destination.name not in {path.name for path in destination.parent.iterdir()}
    assert not list(root.rglob("*.tmp"))


@pytest.mark.parametrize(
    "component",
    [
        "bad<name.json",
        "bad>name.json",
        "bad:name.json",
        'bad"name.json',
        "bad|name.json",
        "bad?name.json",
        "bad*name.json",
        "trailing.",
        "trailing ",
        "CON.txt",
    ],
)
def test_store_rejects_windows_unsafe_artifact_components(
    tmp_path,
    monkeypatch,
    component,
):
    store = ExperimentStore(tmp_path)
    store.create("safe-001")
    monkeypatch.setattr(
        Path,
        "read_text",
        lambda *args, **kwargs: pytest.fail("unsafe path reached the filesystem"),
    )

    with pytest.raises(SafetyError, match="unsafe artifact path component"):
        store.read_json("safe-001", Path("runs") / component)


def test_store_validates_every_artifact_path_component(tmp_path, monkeypatch):
    store = ExperimentStore(tmp_path)
    store.create("safe-001")
    monkeypatch.setattr(
        Path,
        "read_text",
        lambda *args, **kwargs: pytest.fail("unsafe path reached the filesystem"),
    )

    with pytest.raises(SafetyError, match="unsafe artifact path component"):
        store.read_json("safe-001", "plans/CON/result.json")


def test_write_json_persists_unicode_sorted_and_with_trailing_newline(tmp_path):
    store = ExperimentStore(tmp_path)
    store.create("unicode-001")

    destination = store.write_json(
        "unicode-001",
        "runs/result.json",
        {"z": "café", "a": "日本語"},
    )

    assert destination.read_text(encoding="utf-8") == (
        '{\n  "a": "日本語",\n  "z": "café"\n}\n'
    )
    assert not list(destination.parent.glob("*.tmp"))


def test_append_jsonl_is_sanitized_bounded_and_concurrency_safe(tmp_path):
    store = ExperimentStore(tmp_path)
    store.create("journal-001")

    def append(index: int) -> None:
        store.append_jsonl(
            "journal-001",
            "attempts.jsonl",
            {
                "order": index,
                "message": f"attempt {index} apiKey=inline-secret",
                "auth_token": "mapping-secret",
            },
        )

    with ThreadPoolExecutor(max_workers=12) as pool:
        list(pool.map(append, range(200)))

    destination = tmp_path / "journal-001" / "attempts.jsonl"
    content = destination.read_text(encoding="utf-8")
    records = store.read_jsonl("journal-001", "attempts.jsonl")
    assert content.endswith("\n")
    assert len(records) == 200
    assert {item["order"] for item in records} == set(range(200))
    assert {item["auth_token"] for item in records} == {"[REDACTED]"}
    assert "mapping-secret" not in content


def test_append_jsonl_loops_on_short_writes_without_concatenation(
    tmp_path,
    monkeypatch,
):
    store = ExperimentStore(tmp_path)
    store.create("journal-002")
    destination = store.append_jsonl(
        "journal-002",
        "attempts.jsonl",
        {"order": 1, "status": "success"},
    )
    original_write = store_module.os.write
    write_calls = 0

    def short_write(descriptor, value):
        nonlocal write_calls
        write_calls += 1
        written = original_write(descriptor, value[:5])
        assert 0 < written <= 5
        return written

    monkeypatch.setattr(store_module.os, "write", short_write)

    store.append_jsonl(
        "journal-002",
        "attempts.jsonl",
        {"order": 2, "status": "failed"},
    )

    assert write_calls > 1
    assert store.read_jsonl("journal-002", "attempts.jsonl") == (
        {"order": 1, "status": "success"},
        {"order": 2, "status": "failed"},
    )
    assert destination.read_bytes().count(b"\n") == 2


def test_two_store_instances_serialize_overlapping_short_writes(
    tmp_path,
    monkeypatch,
):
    first_store = ExperimentStore(tmp_path)
    second_store = ExperimentStore(tmp_path)
    first_store.create("journal-shared-001")
    destination = first_store.append_jsonl(
        "journal-shared-001",
        "attempts.jsonl",
        {"order": 0, "payload": "warm"},
    )
    original_write = store_module.os.write
    first_chunk = Event()

    def slow_short_write(descriptor, value):
        first_chunk.set()
        sleep(0.0005)
        return original_write(descriptor, value[:3])

    monkeypatch.setattr(store_module.os, "write", slow_short_write)

    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(
            first_store.append_jsonl,
            "journal-shared-001",
            "attempts.jsonl",
            {"order": 1, "payload": "a" * 256},
        )
        assert first_chunk.wait(timeout=5)
        second = pool.submit(
            second_store.append_jsonl,
            "journal-shared-001",
            "attempts.jsonl",
            {"order": 2, "payload": "b" * 256},
        )
        first.result(timeout=15)
        second.result(timeout=15)

    records = ExperimentStore(tmp_path).read_jsonl(
        "journal-shared-001",
        "attempts.jsonl",
    )
    assert [record["order"] for record in records] == [0, 1, 2]
    assert records[1]["payload"] == "a" * 256
    assert records[2]["payload"] == "b" * 256
    assert destination.read_bytes().count(b"\n") == 3


def test_many_store_instances_coordinate_read_repair_and_append(tmp_path):
    stores = [ExperimentStore(tmp_path) for _ in range(24)]
    stores[0].create("journal-shared-002")
    destination = stores[0].append_jsonl(
        "journal-shared-002",
        "attempts.jsonl",
        {"order": 0},
    )
    with destination.open("ab", buffering=0) as journal:
        journal.write(b'{"order":999,"status":"in-flight"')
        os.fsync(journal.fileno())

    barrier = Barrier(len(stores))

    def access(index):
        barrier.wait()
        if index < 8:
            return stores[index].read_jsonl(
                "journal-shared-002",
                "attempts.jsonl",
            )
        stores[index].append_jsonl(
            "journal-shared-002",
            "attempts.jsonl",
            {"order": index},
        )
        return ()

    with ThreadPoolExecutor(max_workers=len(stores)) as pool:
        futures = [pool.submit(access, index) for index in range(len(stores))]
        for future in futures:
            future.result(timeout=20)

    records = ExperimentStore(tmp_path).read_jsonl(
        "journal-shared-002",
        "attempts.jsonl",
    )
    assert records[0] == {"order": 0}
    assert len(records) == 17
    assert {record["order"] for record in records} == {0, *range(8, 24)}


def test_subprocess_jsonl_writers_share_advisory_lock(tmp_path):
    store = ExperimentStore(tmp_path)
    store.create("journal-process-001")
    store.append_jsonl(
        "journal-process-001",
        "attempts.jsonl",
        {"writer": -1, "sequence": -1},
    )
    start_marker = tmp_path / "start-process-writers"
    child_script = r"""
import os
from pathlib import Path
import sys
import time

from mp_comfyui_lab.experiments import store as store_module
from mp_comfyui_lab.experiments.store import ExperimentStore

root, marker, writer = sys.argv[1:]
while not Path(marker).exists():
    time.sleep(0.005)

original_write = store_module.os.write
def slow_short_write(descriptor, value):
    time.sleep(0.0005)
    return original_write(descriptor, value[:5])

store_module.os.write = slow_short_write
store = ExperimentStore(Path(root))
for sequence in range(4):
    store.append_jsonl(
        "journal-process-001",
        "attempts.jsonl",
        {"writer": int(writer), "sequence": sequence, "payload": "x" * 96},
    )
"""
    processes = [
        subprocess.Popen(
            [
                sys.executable,
                "-c",
                child_script,
                str(tmp_path),
                str(start_marker),
                str(writer),
            ],
            cwd=Path.cwd(),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        for writer in range(4)
    ]
    start_marker.write_text("start\n", encoding="utf-8")

    for process in processes:
        stdout, stderr = process.communicate(timeout=30)
        assert process.returncode == 0, f"stdout={stdout}\nstderr={stderr}"

    records = ExperimentStore(tmp_path).read_jsonl(
        "journal-process-001",
        "attempts.jsonl",
    )
    assert records[0] == {"sequence": -1, "writer": -1}
    process_records = records[1:]
    assert len(process_records) == 16
    assert {
        (record["writer"], record["sequence"])
        for record in process_records
    } == {(writer, sequence) for writer in range(4) for sequence in range(4)}
    for writer in range(4):
        assert [
            record["sequence"]
            for record in process_records
            if record["writer"] == writer
        ] == list(range(4))


def test_read_jsonl_repairs_hard_crash_tail_then_append_is_prefix_safe(tmp_path):
    store = ExperimentStore(tmp_path)
    store.create("journal-003")
    destination = store.append_jsonl(
        "journal-003",
        "attempts.jsonl",
        {"order": 1, "status": "success"},
    )
    store.append_jsonl(
        "journal-003",
        "attempts.jsonl",
        {"order": 2, "status": "success"},
    )
    with destination.open("ab", buffering=0) as journal:
        journal.write(b'{"order":3,"status":"in-flight"')
        os.fsync(journal.fileno())

    recovered = store.read_jsonl("journal-003", "attempts.jsonl")
    store.append_jsonl(
        "journal-003",
        "attempts.jsonl",
        {"order": 4, "status": "success"},
    )

    assert recovered == (
        {"order": 1, "status": "success"},
        {"order": 2, "status": "success"},
    )
    assert tuple(store.iter_jsonl("journal-003", "attempts.jsonl")) == (
        {"order": 1, "status": "success"},
        {"order": 2, "status": "success"},
        {"order": 4, "status": "success"},
    )
    assert destination.read_bytes().endswith(b"\n")
    assert destination.read_bytes().count(b"\n") == 3


@pytest.mark.parametrize(
    "invalid_tail",
    [b"not-json\n", b"[]\n", b'"text"\n', b'{"score":NaN}\n'],
)
def test_read_jsonl_discards_only_invalid_final_complete_record(
    tmp_path,
    invalid_tail,
):
    store = ExperimentStore(tmp_path)
    store.create("journal-004")
    destination = store.append_jsonl(
        "journal-004",
        "attempts.jsonl",
        {"order": 1},
    )
    prefix = destination.read_bytes()
    with destination.open("ab", buffering=0) as journal:
        journal.write(invalid_tail)
        os.fsync(journal.fileno())

    assert store.read_jsonl("journal-004", "attempts.jsonl") == ({"order": 1},)
    assert destination.read_bytes() == prefix


def test_read_jsonl_repairs_empty_and_no_newline_files(tmp_path):
    store = ExperimentStore(tmp_path)
    root = store.create("journal-005")
    destination = root / "attempts.jsonl"
    destination.write_bytes(b"")
    assert store.read_jsonl("journal-005", "attempts.jsonl") == ()

    destination.write_bytes(b'{"order":1}')
    assert store.read_jsonl("journal-005", "attempts.jsonl") == ()
    assert destination.read_bytes() == b""


def test_read_jsonl_never_discards_invalid_earlier_record(tmp_path):
    store = ExperimentStore(tmp_path)
    root = store.create("journal-006")
    destination = root / "attempts.jsonl"
    content = b'{"order":1}\nnot-json\n{"order":3}\n'
    destination.write_bytes(content)

    with pytest.raises(SafetyError, match="record 2"):
        store.read_jsonl("journal-006", "attempts.jsonl")

    assert destination.read_bytes() == content


def test_concurrent_append_repairs_one_crash_tail_once(tmp_path):
    store = ExperimentStore(tmp_path)
    store.create("journal-007")
    destination = store.append_jsonl(
        "journal-007",
        "attempts.jsonl",
        {"order": 1},
    )
    with destination.open("ab", buffering=0) as journal:
        journal.write(b'{"order":999')
        os.fsync(journal.fileno())
    workers = 16
    barrier = Barrier(workers)

    def append(order: int) -> None:
        barrier.wait()
        store.append_jsonl("journal-007", "attempts.jsonl", {"order": order})

    with ThreadPoolExecutor(max_workers=workers) as pool:
        list(pool.map(append, range(2, workers + 2)))

    records = store.read_jsonl("journal-007", "attempts.jsonl")
    assert records[0] == {"order": 1}
    assert len(records) == workers + 1
    assert {item["order"] for item in records} == set(range(1, workers + 2))
    assert destination.read_bytes().count(b"\n") == workers + 1


def test_jsonl_append_and_read_reject_symlink_leaf(tmp_path):
    store = ExperimentStore(tmp_path / "experiments")
    root = store.create("journal-008")
    target = root / "sources" / "target.jsonl"
    target.write_text('{"order":1}\n', encoding="utf-8")
    alias = root / "attempts.jsonl"
    try:
        alias.symlink_to(target)
    except (NotImplementedError, OSError):
        pytest.skip("file symlinks are not supported")

    with pytest.raises(SafetyError, match="symlink or junction"):
        store.read_jsonl("journal-008", "attempts.jsonl")
    with pytest.raises(SafetyError, match="symlink or junction"):
        store.append_jsonl("journal-008", "attempts.jsonl", {"order": 2})
    assert target.read_text(encoding="utf-8") == '{"order":1}\n'


def test_jsonl_lock_file_rejects_symlink_leaf(tmp_path):
    store = ExperimentStore(tmp_path / "experiments")
    root = store.create("journal-lock-001")
    target = tmp_path / "outside-lock-target"
    target.write_bytes(b"outside")
    lock_path = root / "attempts.jsonl.lock"
    try:
        lock_path.symlink_to(target)
    except (NotImplementedError, OSError):
        pytest.skip("file symlinks are not supported")

    with pytest.raises(SafetyError, match="symlink or junction"):
        store.append_jsonl("journal-lock-001", "attempts.jsonl", {"order": 1})
    with pytest.raises(SafetyError, match="symlink or junction"):
        store.read_jsonl("journal-lock-001", "attempts.jsonl")
    assert target.read_bytes() == b"outside"


def test_jsonl_append_and_recovery_are_linear_at_scale(tmp_path):
    store = ExperimentStore(tmp_path)
    store.create("journal-009")
    started = perf_counter()
    for order in range(1_000):
        store.append_jsonl("journal-009", "attempts.jsonl", {"order": order})
    append_elapsed = perf_counter() - started

    started = perf_counter()
    records = store.read_jsonl("journal-009", "attempts.jsonl")
    read_elapsed = perf_counter() - started

    assert append_elapsed < 30.0
    assert read_elapsed < 5.0
    assert len(records) == 1_000
    assert records[0] == {"order": 0}
    assert records[-1] == {"order": 999}


def test_write_text_creates_parent_directories_atomically(tmp_path):
    store = ExperimentStore(tmp_path)
    store.create("text-001")

    destination = store.write_text(
        "text-001",
        "plans/nested/plan.txt",
        "angle plan\n",
    )

    assert destination.read_text(encoding="utf-8") == "angle plan\n"
    assert not list(destination.parent.glob("*.tmp"))


@pytest.mark.parametrize("relative_path", ["../escape.json", "runs/../../escape.json"])
def test_store_rejects_relative_path_traversal(tmp_path, relative_path):
    store = ExperimentStore(tmp_path)
    store.create("escape-001")

    with pytest.raises(SafetyError, match="traversal"):
        store.write_json("escape-001", relative_path, {})


def test_store_rejects_absolute_artifact_path(tmp_path):
    store = ExperimentStore(tmp_path)
    store.create("escape-001")

    with pytest.raises(SafetyError, match="relative"):
        store.write_json("escape-001", tmp_path / "outside.json", {})


def test_read_json_rejects_path_escape(tmp_path):
    store = ExperimentStore(tmp_path)
    store.create("escape-001")

    with pytest.raises(SafetyError, match="traversal"):
        store.read_json("escape-001", "../outside.json")


def test_store_rejects_symlink_escape_when_supported(tmp_path):
    store = ExperimentStore(tmp_path / "experiments")
    root = store.create("symlink-001")
    outside = tmp_path / "outside"
    outside.mkdir()
    link = root / "runs" / "outside"
    try:
        link.symlink_to(outside, target_is_directory=True)
    except (NotImplementedError, OSError):
        pytest.skip("directory symlinks are not supported")

    with pytest.raises(SafetyError, match="symlink or junction"):
        store.write_text("symlink-001", "runs/outside/result.txt", "unsafe")


def test_store_rejects_in_root_symlink_leaf_without_modifying_target(tmp_path):
    store = ExperimentStore(tmp_path / "experiments")
    root = store.create("symlink-001")
    target = root / "sources" / "target.txt"
    target.write_text("original", encoding="utf-8")
    alias = root / "runs" / "alias.txt"
    try:
        alias.symlink_to(target)
    except (NotImplementedError, OSError):
        pytest.skip("file symlinks are not supported")

    with pytest.raises(SafetyError, match="symlink or junction"):
        store.write_text("symlink-001", "runs/alias.txt", "replacement")

    assert target.read_text(encoding="utf-8") == "original"


def test_store_rejects_symlink_experiment_root(tmp_path):
    store = ExperimentStore(tmp_path / "experiments")
    target = store.create("target-001")
    alias = tmp_path / "experiments" / "alias-001"
    try:
        alias.symlink_to(target, target_is_directory=True)
    except (NotImplementedError, OSError):
        pytest.skip("directory symlinks are not supported")

    with pytest.raises(SafetyError, match="symlink or junction"):
        store.root("alias-001")


def test_store_rejects_in_root_junction_without_pathlib_helper(
    tmp_path,
    monkeypatch,
):
    if os.name != "nt":
        pytest.skip("directory junctions are Windows-specific")

    store = ExperimentStore(tmp_path / "experiments")
    root = store.create("junction-001")
    target = root / "sources"
    target_file = target / "target.txt"
    target_file.write_text("original", encoding="utf-8")
    alias = root / "runs" / "alias"
    completed = subprocess.run(
        ["cmd.exe", "/d", "/c", "mklink", "/J", str(alias), str(target)],
        capture_output=True,
        check=False,
        text=True,
    )
    if completed.returncode != 0:
        pytest.skip("directory junctions are not supported")

    monkeypatch.setattr(Path, "is_junction", None, raising=False)

    with pytest.raises(SafetyError, match="symlink or junction"):
        store.write_text("junction-001", "runs/alias/target.txt", "replacement")

    assert target_file.read_text(encoding="utf-8") == "original"


def test_replace_failure_leaves_no_temporary_artifacts(tmp_path, monkeypatch):
    store = ExperimentStore(tmp_path)
    root = store.create("replace-001")
    destination = root / "runs" / "result.json"
    destination.write_text("original\n", encoding="utf-8")

    def fail_replace(source, destination):
        raise OSError("simulated replace failure")

    monkeypatch.setattr(store_module.os, "replace", fail_replace)

    with pytest.raises(OSError, match="simulated replace failure"):
        store.write_json("replace-001", "runs/result.json", {"ok": True})

    assert destination.read_text(encoding="utf-8") == "original\n"
    assert not list(root.rglob("*.tmp"))


def test_fsync_failure_preserves_destination_and_cleans_temporary_file(
    tmp_path,
    monkeypatch,
):
    store = ExperimentStore(tmp_path)
    root = store.create("fsync-001")
    destination = root / "runs" / "result.txt"
    destination.write_text("original", encoding="utf-8")

    def fail_fsync(file_descriptor):
        raise OSError("simulated fsync failure")

    monkeypatch.setattr(store_module.os, "fsync", fail_fsync)

    with pytest.raises(OSError, match="simulated fsync failure"):
        store.write_text("fsync-001", "runs/result.txt", "replacement")

    assert destination.read_text(encoding="utf-8") == "original"
    assert not list(root.rglob("*.tmp"))


def test_serialization_failure_leaves_no_temporary_artifacts(tmp_path):
    store = ExperimentStore(tmp_path)
    root = store.create("serialize-001")

    with pytest.raises(TypeError):
        store.write_json("serialize-001", "runs/result.json", {"bad": object()})

    assert not list(root.rglob("*.tmp"))


def test_publish_file_new_streams_with_max_byte_guard(tmp_path: Path) -> None:
    store = ExperimentStore(tmp_path / "experiments")
    root = store.create("stream-output-001")
    source = tmp_path / "source.bin"
    source.write_bytes(b"12345")

    with pytest.raises(SafetyError, match="byte limit"):
        store.publish_file_new(
            "stream-output-001",
            "outputs/baseline/0000-9.bin",
            source,
            max_bytes=4,
            reuse_matching=True,
        )

    assert not (root / "outputs" / "baseline" / "0000-9.bin").exists()
    assert not list(root.rglob("*.tmp"))


def test_publish_file_new_reuses_matching_content_and_rejects_mismatch(
    tmp_path: Path,
) -> None:
    store = ExperimentStore(tmp_path / "experiments")
    root = store.create("stream-output-002")
    source = tmp_path / "source.bin"
    source.write_bytes(b"approved output bytes")
    relative_path = "outputs/baseline/0000-9.bin"

    first = store.publish_file_new(
        "stream-output-002",
        relative_path,
        source,
        max_bytes=1_024,
        reuse_matching=True,
    )
    second = store.publish_file_new(
        "stream-output-002",
        relative_path,
        source,
        max_bytes=1_024,
        reuse_matching=True,
    )

    assert first == second
    assert first[1:] == (
        hashlib.sha256(b"approved output bytes").hexdigest(),
        len(b"approved output bytes"),
    )

    source.write_bytes(b"different output bytes")
    with pytest.raises(SafetyError, match="different content"):
        store.publish_file_new(
            "stream-output-002",
            relative_path,
            source,
            max_bytes=1_024,
            reuse_matching=True,
        )

    assert (root / relative_path).read_bytes() == b"approved output bytes"
    assert not list(root.rglob("*.tmp"))
