import shutil
import subprocess
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
    preview_document_chapters,
    preview_queue_book,
    preview_queue_book_chapter,
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


@pytest.mark.parametrize("extension", ["epub", "mobi"])
def test_ebook_import_only_selected_chapter(tmp_path: Path, monkeypatch, extension: str):
    from ebooklib import epub

    source_epub = tmp_path / "chapters.epub"
    book = epub.EpubBook()
    book.set_identifier("indextts-chapter-selection")
    book.set_title("章节导入测试")
    book.set_language("zh-CN")
    chapters = []
    for index, title in enumerate(("第一章", "第二章", "第三章"), start=1):
        chapter = epub.EpubHtml(title=title, file_name=f"chapter{index}.xhtml", lang="zh-CN")
        chapter.content = f"<h1>{title}</h1><p>{title}独有正文内容。</p>"
        book.add_item(chapter)
        chapters.append(chapter)
    book.toc = tuple(chapters)
    book.spine = ["nav", *chapters]
    book.add_item(epub.EpubNav())
    book.add_item(epub.EpubNcx())
    epub.write_epub(str(source_epub), book)

    source = source_epub
    if extension == "mobi":
        converter = shutil.which("ebook-convert") or "/Applications/calibre.app/Contents/MacOS/ebook-convert"
        if not Path(converter).is_file():
            pytest.skip("Calibre ebook-convert is unavailable")
        source = tmp_path / "chapters.mobi"
        process = subprocess.run(
            [converter, str(source_epub), str(source)],
            capture_output=True,
            text=True,
            timeout=300,
        )
        assert process.returncode == 0, process.stderr

    monkeypatch.setattr(webui, "DOCUMENT_CACHE_DIR", tmp_path / "cache")
    document_data, summary, selector, preview = parse_uploaded_document(
        str(source), progress=DummyProgress()
    )
    assert "解析完成" in summary
    assert selector.value == []
    assert "第一章独有正文" not in preview
    assert len(selector.choices) == 3

    selected_value = next(value for label, value in selector.choices if "第二章" in label)
    selected_preview = preview_document_chapters(document_data, [selected_value])
    assert "第二章独有正文" in selected_preview
    assert "第一章独有正文" not in selected_preview
    selected_text, status = load_document_chapters(document_data, [selected_value])
    assert "第二章独有正文" in selected_text
    assert "第一章独有正文" not in selected_text
    assert "已载入 1 个章节" in status

    actual_import = webui.import_document
    import_calls = []

    def tracked_import(path, **kwargs):
        import_calls.append(path)
        return actual_import(path, **kwargs)

    monkeypatch.setattr(webui, "import_document", tracked_import)
    queue, sources, queue_selector, book_selector, queue_summary, _, _ = add_documents_to_queue(
        [str(source)], [], [], progress=DummyProgress()
    )
    assert queue == []
    assert len(sources) == 1
    assert queue_selector.value is None
    assert book_selector.value == sources[0]["id"]
    with pytest.raises(Exception, match="选择电子书"):
        confirm_queue_document(queue, None, "", sources, book_selector.value, None)

    _, _, book_status, chapter_selector = preview_queue_book(
        sources, book_selector.value, queue
    )
    assert chapter_selector.visible is True
    assert len(chapter_selector.choices) == 3
    assert "已解析" in book_status
    jump_selection = [chapter_selector.choices[2][1], chapter_selector.choices[0][1]]
    jump_text, _ = preview_queue_book_chapter(
        sources, book_selector.value, jump_selection
    )
    assert "第一章独有正文" in jump_text
    assert "第三章独有正文" in jump_text
    assert "第二章独有正文" not in jump_text
    jump_queue, _, _, _, _, _ = confirm_queue_document(
        [], None, jump_text, sources, book_selector.value, jump_selection
    )
    assert len(jump_queue) == 1
    assert jump_queue[0]["selected_chapters"] == ["1. 第一章", "3. 第三章"]
    for number, title in enumerate(("第一章", "第二章", "第三章"), start=1):
        chapter_id = next(
            value for label, value in chapter_selector.choices if title in label
        )
        chapter_text, status = preview_queue_book_chapter(
            sources, book_selector.value, chapter_id
        )
        assert f"{title}独有正文" in chapter_text
        assert all(
            f"{other}独有正文" not in chapter_text
            for other in ("第一章", "第二章", "第三章") if other != title
        )
        assert "请检查文案并确认" in status
        queue, queue_selector, queue_summary, confirm_status, cleared_chapter, cleared_text = (
            confirm_queue_document(
                queue, None, chapter_text, sources, book_selector.value, chapter_id
            )
        )
        assert len(queue) == number
        assert all(item["confirmed"] for item in queue)
        assert queue_selector.value is None
        assert cleared_chapter.value == []
        assert cleared_text == ""
        assert f"已确认 {number}/{number}" in queue_summary
        assert "可继续选择剩余章节" in confirm_status or "已全部加入队列" in confirm_status
    assert len(sources) == 1
    assert import_calls == [str(source)]
    assert len({item["id"] for item in queue}) == 3
    assert [item["selected_chapters"] for item in queue] == [
        [f"{n}. {title}"]
        for n, title in enumerate(("第一章", "第二章", "第三章"), start=1)
    ]


@pytest.mark.parametrize("file_type", ["EPUB", "MOBI"])
def test_ebook_can_confirm_multiple_chapters_then_select_more(file_type: str):
    sources = [{
        "id": "ebook",
        "filename": f"chapters.{file_type.lower()}",
        "title": "章节书",
        "document": {
            "filename": f"chapters.{file_type.lower()}",
            "file_type": file_type,
            "title": "章节书",
            "chapters": [
                {"title": title, "text": f"{title}正文。", "source_index": index}
                for index, title in enumerate(("第一章", "第二章", "第三章", "第四章"))
            ],
        },
    }]
    _, _, _, selector = preview_queue_book(sources, "ebook", [])
    assert len(selector.choices) == 4
    first_batch = [selector.choices[2][1], selector.choices[0][1]]
    preview, status = preview_queue_book_chapter(sources, "ebook", first_batch)
    daytime_text, _ = load_document_chapters(sources[0]["document"], first_batch)
    assert preview == daytime_text
    assert "已选择 2 章" in status
    assert preview.index("第一章正文") < preview.index("第三章正文")
    assert "第二章正文" not in preview
    assert "第四章正文" not in preview
    preview = preview.replace("第三章正文。", "第三章已校订正文。")
    queue, _, summary, message, next_selector, cleared_text = confirm_queue_document(
        [], None, preview, sources, "ebook", first_batch
    )
    assert len(queue) == 1
    assert queue[0]["selected_chapters"] == ["1. 第一章", "3. 第三章"]
    assert queue[0]["text"] == "第一章正文。\n\n第三章已校订正文。"
    assert all(item["confirmed"] for item in queue)
    assert next_selector.value == []
    assert [value for _label, value in next_selector.choices] == ["2. 第二章", "4. 第四章"]
    assert cleared_text == ""
    assert "已确认 1/1" in summary
    assert "合并为 1 个队列任务、生成 1 个音频" in message

    second_batch = [value for _label, value in next_selector.choices]
    second_text, _ = preview_queue_book_chapter(sources, "ebook", second_batch)
    queue, _, _, message, next_selector, _ = confirm_queue_document(
        queue, None, second_text, sources, "ebook", second_batch
    )
    assert len(queue) == 2
    assert queue[-1]["selected_chapters"] == ["2. 第二章", "4. 第四章"]
    assert queue[-1]["text"] == "第二章正文。\n\n第四章正文。"
    assert next_selector.choices == []
    assert "已全部加入队列" in message
    with pytest.raises(Exception, match="已在队列"):
        confirm_queue_document(queue, None, preview, sources, "ebook", first_batch)


def test_ebook_group_respects_combined_text_limit():
    source = {
        "id": "ebook",
        "filename": "long.epub",
        "title": "长书",
        "document": {
            "filename": "long.epub", "file_type": "EPUB", "title": "长书",
            "chapters": [
                {"title": "第一章", "text": "甲" * 60_000, "source_index": 0},
                {"title": "第二章", "text": "乙" * 60_000, "source_index": 1},
            ],
        },
    }
    chapter_ids = ["1. 第一章", "2. 第二章"]
    preview, status = preview_queue_book_chapter([source], "ebook", chapter_ids)
    assert "合并后超过单次合成上限" in status
    with pytest.raises(Exception, match="超过长文合成上限"):
        confirm_queue_document([], None, preview, [source], "ebook", chapter_ids)


def test_queue_chapter_selector_allows_multiple_chapters():
    demo = webui.build_ui()
    components = demo.config["components"]
    selector = next(
        component for component in components
        if component.get("props", {}).get("label") == "选择本次要生成的章节（可多选）"
    )
    assert selector["props"]["multiselect"] is True
    queue_text = next(
        component for component in components
        if component.get("props", {}).get("label") == "合成文字（本次任务）"
    )
    daytime_text = next(
        component for component in components
        if component.get("props", {}).get("label") == "合成文字"
    )
    events = {item["api_name"]: item for item in demo.config["dependencies"]}
    assert events["preview_queue_book_chapter"]["outputs"][0] == queue_text["id"]
    assert events["confirm_queue_document"]["inputs"][2] == queue_text["id"]
    assert events["confirm_queue_document"]["outputs"][-1] == queue_text["id"]
    assert events["mark_queue_document_edited"]["inputs"][-1] == queue_text["id"]
    assert queue_text["id"] != daytime_text["id"]


def test_multiple_documents_can_be_queued_previewed_confirmed_and_reordered(tmp_path: Path):
    first = tmp_path / "first.txt"
    second = tmp_path / "second.txt"
    first.write_text("第一份文档正文。", encoding="utf-8")
    second.write_text("第二份文档正文。", encoding="utf-8")

    queue, sources, selector, book_selector, summary, cleared, status = add_documents_to_queue(
        [str(first), str(second)], [], [], progress=DummyProgress()
    )

    assert len(queue) == 2
    assert cleared is None
    assert "已解析 2 份文档" in status
    assert sources == []
    assert book_selector.visible is False
    assert "待确认" in summary
    selected_id = selector.value
    preview, preview_status, chapter_selector, _ = preview_queue_document(queue, selected_id)
    assert "第一份文档正文" in preview
    assert "正在预览" in preview_status
    assert chapter_selector.visible is False

    queue, selector, summary, confirm_status, _, _ = confirm_queue_document(
        queue, selected_id, preview + "\n\n已检查。"
    )
    assert queue[0]["confirmed"] is True
    assert queue[0]["status"] == "confirmed"
    assert "已确认" in summary
    assert "已确认" in confirm_status

    queue, selector, summary, _status = move_queue_document_down(queue, selected_id)
    assert queue[1]["id"] == selected_id
    assert selector.value == selected_id
    assert "排队任务 2 个" in summary

    queue, selector, summary, next_text, remove_status = remove_queue_document(
        queue, selected_id
    )
    assert len(queue) == 1
    assert selector.value == queue[0]["id"]
    assert "第二份文档正文" in next_text
    assert "已从队列移除" in remove_status
    assert "排队任务 1 个" in render_document_queue(queue)


def test_confirmed_document_queue_runs_sequentially_and_writes_manifest(
    tmp_path: Path, monkeypatch
):
    sources = [{
        "id": "ebook",
        "filename": "chapters.epub",
        "title": "章节书",
        "document": {
            "filename": "chapters.epub",
            "file_type": "EPUB",
            "title": "章节书",
            "chapters": [
                {"title": "第一章", "text": "第一章正文。", "source_index": 0},
                {"title": "第二章", "text": "第二章正文。", "source_index": 1},
                {"title": "第三章", "text": "第三章正文。", "source_index": 2},
            ],
        },
    }]
    ebook_queue = []
    first_batch = ["1. 第一章", "2. 第二章"]
    selected_text, _ = preview_queue_book_chapter(sources, "ebook", first_batch)
    ebook_queue, _, _, _, _, _ = confirm_queue_document(
        ebook_queue, None, selected_text, sources, "ebook", first_batch
    )
    last_chapter = ["3. 第三章"]
    selected_text, _ = preview_queue_book_chapter(sources, "ebook", last_chapter)
    ebook_queue, _, _, _, _, _ = confirm_queue_document(
        ebook_queue, None, selected_text, sources, "ebook", last_chapter
    )
    queue = ebook_queue + [
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
            "IndexTTS 2.5",
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
            "clone", "chinese", "", "", 0.0, 32, 2.0, 0.0, 5.0, 5.0, 0.1, 10.0,
            "mp3",
            str(tmp_path),
        )
    )

    final_queue = updates[-1][0]
    assert generated == [
        "第一章正文。\n\n第二章正文。", "第三章正文。",
        "第一份正文。", "第二份正文。",
    ]
    assert [item["status"] for item in final_queue] == ["completed"] * 4
    assert len({item["output"] for item in final_queue}) == 4
    assert all(Path(item["output"]).exists() for item in final_queue)
    assert len(list(tmp_path.glob("文档转换队列_*.json"))) == 1
    assert "成功 4 份" in updates[-1][3]


def test_document_queue_uses_multi_voice_for_every_file(tmp_path: Path, monkeypatch):
    voices = {
        "voice-a": {"id": "voice-a", "name": "甲声"},
        "voice-b": {"id": "voice-b", "name": "乙声"},
    }
    calls = []
    monkeypatch.setattr(webui, "_load_voice_entry", lambda voice_id: voices.get(voice_id))
    monkeypatch.setattr(webui, "_activate_voice_entry", lambda *_args, **_kwargs: (None, "", None))
    monkeypatch.setattr(webui, "read_user_config", lambda: {"voice_library_id": "voice-a"})
    monkeypatch.setattr(webui, "update_user_config", lambda **_kwargs: None)
    monkeypatch.setattr(webui, "_start_sleep_prevention", lambda: None)
    monkeypatch.setattr(webui, "_stop_sleep_prevention", lambda _process: None)

    def fake_synthesize(**kwargs):
        calls.append((kwargs["text"], kwargs["voice_library_id"]))
        part = Path(kwargs["output_directory"]) / f"part-{len(calls)}.wav"
        part.write_bytes(f"{kwargs['voice_library_id']}:{kwargs['text']}".encode())
        return str(part), "生成完成", str(part)

    def fake_concat(parts, target, _gap):
        target.write_bytes(b"|".join(part.read_bytes() for part in parts))

    monkeypatch.setattr(webui, "_synthesize_unlocked", fake_synthesize)
    monkeypatch.setattr(webui, "_concatenate_wav_batches", fake_concat)
    queue = [
        {"id": "first", "title": "第一份", "filename": "first.txt",
         "text": "第一段。\n\n第二段。", "confirmed": True},
        {"id": "second", "title": "第二份", "filename": "second.txt",
         "text": "第三段。\n\n第四段。", "confirmed": True},
    ]

    updates = list(webui.synthesize_document_queue_stream(
        queue, "voice-a", "IndexTTS 2.5", "自然/平静", 0.6, 1.0, 42,
        250, 50, 120, 0.8, 25, 1500, 0.8, 30, 10.0, 0.7, False,
        "clone", "chinese", "", "", 0.0, 32, 2.0, 0.0, 5.0, 5.0, 0.1, 10.0,
        "wav", str(tmp_path), multi_voice_enabled=True,
        multi_voice_ids=["voice-a", "voice-b"],
        multi_voice_split_mode="paragraph", multi_voice_switch_every=1,
    ))

    assert calls == [
        ("第一段。", "voice-a"), ("第二段。", "voice-b"),
        ("第三段。", "voice-a"), ("第四段。", "voice-b"),
    ]
    final_queue = updates[-1][0]
    assert [item["status"] for item in final_queue] == ["completed", "completed"]
    assert len({item["output"] for item in final_queue}) == 2
    assert Path(final_queue[0]["output"]).read_bytes() == "voice-a:第一段。|voice-b:第二段。".encode()
    assert Path(final_queue[1]["output"]).read_bytes() == "voice-a:第三段。|voice-b:第四段。".encode()
    assert "成功 2 份" in updates[-1][3]
