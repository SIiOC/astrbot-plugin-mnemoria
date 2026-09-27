"""前端静态契约测试（无需浏览器）。

背景：宿主把插件页放在沙箱 iframe 中，sandbox 为
    "allow-scripts allow-forms allow-downloads"
——**缺 allow-modals**。浏览器会**静默屏蔽**原生 confirm()/prompt()/alert()
（confirm 直接返回 false、不弹窗），导致所有确认类操作"点了没反应"。
2026-09-16 真机实测：记忆库删除按钮点击无效，即此因。

因此控制台必须使用自绘弹窗（app.js 的 confirmBox/promptBox → window.MN.confirm/prompt）。
本测试锁死该约束，防止将来回退到原生弹窗。
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

CONSOLE = Path(__file__).resolve().parents[1] / "pages" / "console"
JS_FILES = ["app.js", "library.js", "starmap.js", "configview.js"]


def _strip_comments(src: str) -> str:
    """去掉注释，避免注释里提到的 confirm( 造成误判。"""
    src = re.sub(r"/\*.*?\*/", "", src, flags=re.S)
    src = re.sub(r"^\s*//.*$", "", src, flags=re.M)
    return src


class TestNoSandboxBlockedDialogs:
    def test_no_native_confirm_prompt_alert(self):
        """控制台 JS 不得直接调用原生 confirm/prompt/alert（沙箱会静默拦截）。"""
        offenders = []
        for name in JS_FILES:
            path = CONSOLE / name
            if not path.exists():
                continue
            src = _strip_comments(path.read_text(encoding="utf-8"))
            for m in re.finditer(r"(?<![.\w])(confirm|prompt|alert)\s*\(", src):
                line = src[: m.start()].count("\n") + 1
                offenders.append(f"{name}:{line} -> {m.group(1)}()")
        assert not offenders, (
            "发现原生弹窗调用（沙箱 iframe 缺 allow-modals，会被静默屏蔽）：\n  "
            + "\n  ".join(offenders)
            + "\n请改用 MN.confirm / MN.prompt（app.js 自绘弹窗）。"
        )

    def test_dialog_helpers_defined_and_exported(self):
        """app.js 必须定义并导出自绘弹窗接口。"""
        src = CONSOLE.joinpath("app.js").read_text(encoding="utf-8")
        assert "function confirmBox" in src, "缺少 confirmBox 实现"
        assert "function promptBox" in src, "缺少 promptBox 实现"
        assert re.search(r"confirm:\s*confirmBox", src), "window.MN 未导出 confirm"
        assert re.search(r"prompt:\s*promptBox", src), "window.MN 未导出 prompt"

    def test_dialog_css_present(self):
        src = CONSOLE.joinpath("style.css").read_text(encoding="utf-8")
        assert ".mn-mask" in src and ".mn-dialog" in src, "缺少自绘弹窗样式"

    def test_module_views_use_mn_confirm(self):
        """library.js（记忆库）的确认操作必须走 MN.confirm。"""
        src = _strip_comments(CONSOLE.joinpath("library.js").read_text(encoding="utf-8"))
        assert "MN.confirm(" in src, "记忆库未使用 MN.confirm"
        # 删除按钮的事件绑定必须存在
        assert re.search(r'\[data-del\].*addEventListener', src), "删除按钮未绑定事件"

    def test_create_table_has_assistant_only_ledger(self):
        """账本接口必须按 scope 过滤，且只返回助手消息（面板隐私要求）。"""
        wa = Path(__file__).resolve().parents[1] / "core" / "web_api.py"
        src = wa.read_text(encoding="utf-8")
        block = src[src.index("def _list_ledger"):src.index("async def _recall")]
        compact = block.replace(" ", "")
        assert "role='assistant'" in compact, "账本列表未在 SQL 层过滤助手消息"
        assert "scope=?" in compact, "账本列表未按 scope 过滤"

    def test_trash_view_shows_created_at(self):
        """回收站视图必须展示创建时间列。"""
        html = CONSOLE.joinpath("index.html").read_text(encoding="utf-8")
        assert "创建于" in html, "回收站表头缺少创建时间列"
        js = CONSOLE.joinpath("app.js").read_text(encoding="utf-8")
        block = js[js.index("async function loadTrash"):]
        block = block[:block.index("\n  }")]
        assert "created_at_local" in block, "回收站行未渲染创建时间"


class TestStarmapPerformance:
    """星图 rAF 循环必须在视图不可见时跳过重绘。

    缺陷背景（2026-09-16）：frame() 是无限 rAF 自循环，视图切换只是 CSS
    隐藏——无守卫时 887 节点×每帧约 1774 个渐变在后台永久空转烧 CPU。
    """

    def test_frame_has_visibility_guard(self):
        src = (CONSOLE / "starmap.js").read_text(encoding="utf-8")
        seg = src[src.index("function frame"):]
        seg = seg[:seg.index("\n  }")]
        assert "offsetParent === null" in seg, \
            "frame() 缺可见性守卫：后台视图下仍全量重绘（CPU 空转）"

    def test_starmap_content_escaped(self):
        """详情卡/提示/探针结果中的记忆正文必须转义（XSS 防线）。"""
        src = (CONSOLE / "starmap.js").read_text(encoding="utf-8")
        for pat in ("MN.esc(m.content)", "MN.esc(best.content)",
                    "MN.esc(o.content)", "MN.esc(c.content"):
            assert pat in src, f"缺少转义调用: {pat}"
