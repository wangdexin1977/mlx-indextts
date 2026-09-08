from pathlib import Path

import pytest

import mlx_indextts.webui as webui
from mlx_indextts.webui import (
    add_documents_to_queue,
    clear_imported_document,
    confirm_queue_document,
    load_document_chapters,
    load_entire_document,
    move_queue_document_down,
    parse_uploaded_document,
    preview_queue_document,
    remove_queue_document,
    render_document_queue,
)


class DummyProgress:
    def __init__(self):
        self.events = []

    def __call__(self, value, desc=None):
        self.events.append((value, desc))


def test_webui_document_parse_load_selected_and_clear(tmp_path: Path):
    source = tmp_path / "webui.txt"
    source.write_text("界面导入测试文本。", encoding="utf-8")
    progress = DummyProgress()

    document_data, summary, selector, preview = parse_uploaded_document(
        str(source), progress=progress
    )

    assert document_data is not None
    assert "解析完成" in summary
    assert "界面导入测试文本" in preview
    assert selector.choices

    selected_value = selector.choices[0][1]
    selected_text, selected_status = load_document_chapters(
        document_data, [selected_value]
    )
    assert "界面导入测试文本" in selected_text
    assert "已载入 1 个章节" in selected_status

    all_text, all_status = load_entire_document(document_data)
    assert all_text == selected_text
    assert "已载入全文" in all_status

    cleared = clear_imported_document()
    assert cleared[0] is None
    assert cleared[1] is None
    assert "尚未导入文档" in cleared[2]
    assert cleared[4] == ""
    assert cleared[5] == ""


def test_webui_requires_document_and_selection():
    with pytest.raises(Exception, match="请先上传"):
        load_entire_document(None)
    with pytest.raises(Exception, match="至少选择"):
        load_document_chapters(
            {
                "filename": "x.txt",
                "file_type": "TXT",
                "title": "x",
                "chapters": [
                    {
                        "title": "x",
                        "text": "正文",
                        "source_index": 0,
                        "page_start": None,
                        "page_end": None,
                        "include_title": True,
                    }
                ],
            },
            [],
        )


def test_multiple_documents_can_be_queued_previewed_confirmed_and_reordered(tmp_path: Path):
    first = tmp_path / "first.txt"
    second = tmp_path / "second.txt"
    first.write_text("第一份文档正文。", encoding="utf-8")
    second.write_text("第二份文档正文。", encoding="utf-8")

    queue, selector, summary, cleared, status = add_documents_to_queue(
        [str(first), str(second)], [], progress=DummyProgress()
    )

    assert len(queue) == 2
    assert cleared is None
    assert "已加入 2 份文档" in status
    assert "待确认" in summary
    selected_id = selector.value
    preview, preview_status = preview_queue_document(queue, selected_id)
    assert "第一份文档正文" in preview
    assert "正在预览" in preview_status

    queue, selector, summary, confirm_status = confirm_queue_document(
        queue, selected_id, preview + "\n\n已检查。"
    )
    assert queue[0]["confirmed"] is True
    assert queue[0]["status"] == "confirmed"
    assert "已确认" in summary
    assert "已确认" in confirm_status

    queue, selector, summary, _status = move_queue_document_down(queue, selected_id)
    assert queue[1]["id"] == selected_id
    assert selector.value == selected_id
    assert "排队文档 2 份" in summary

    queue, selector, summary, next_text, remove_status = remove_queue_document(
        queue, selected_id
    )
    assert len(queue) == 1
    assert selector.value == queue[0]["id"]
    assert "第二份文档正文" in next_text
    assert "已从队列移除" in remove_status
    assert "排队文档 1 份" in render_document_queue(queue)


def test_confirmed_document_queue_runs_sequentially_and_writes_manifest(
    tmp_path: Path, monkeypatch
):
    queue = [
        {
            "id": "first",
            "filename": "first.txt",
            "title": "第一份",
            "text": "第一份正文。",
            "status": "confirmed",
            "confirmed": True,
        },
        {
            "id": "second",
            "filename": "second.txt",
            "title": "第二份",
            "text": "第二份正文。",
            "status": "confirmed",
            "confirmed": True,
        },
    ]
    generated = []

    def fake_synthesize(text, *_args, **_kwargs):
        path = tmp_path / f"generated-{len(generated) + 1}.mp3"
        path.write_bytes(text.encode("utf-8"))
        generated.append(text)
        return str(path), "生成完成", str(path)

    monkeypatch.setattr(webui, "_synthesize_unlocked", fake_synthesize)
    monkeypatch.setattr(webui, "_start_sleep_prevention", lambda: None)
    monkeypatch.setattr(webui, "_stop_sleep_prevention", lambda _process: None)

    updates = list(
        webui.synthesize_document_queue_stream(
            queue,
            "voice-id",
            "自然/平静",
            0.6,
            1.0,
            42,
            250,
            50,
            120,
            0.8,
            25,
            1500,
            0.8,
            30,
            10.0,
            0.7,
            False,
            "mp3",
            str(tmp_path),
        )
    )

    final_queue = updates[-1][0]
    assert generated == ["第一份正文。", "第二份正文。"]
    assert [item["status"] for item in final_queue] == ["completed", "completed"]
    assert all(Path(item["output"]).exists() for item in final_queue)
    assert len(list(tmp_path.glob("文档转换队列_*.json"))) == 1
    assert "成功 2 份" in updates[-1][3]
