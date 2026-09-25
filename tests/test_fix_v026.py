"""v0.2.6：发版卫生回归（审查缺陷修复）。

背景（2026-09-22 全量审查发现，均为 P3 但属"每次发版都要人肉盯"的
可自动化的漂移）：
1. main.py 的 @register 版本号硬编码 "0.2.3"，v0.2.4/v0.2.5 两次发版
   都忘了同步，插件面板一直显示旧版本号；
2. v0.2.3 新增的两个 admission 配置键（text_dedup_similarity /
   conservative_fallback_similarity）没写进 _conf_schema.json，
   配置面板里看不到也无法调整。

本文件用两条测试把这两类漂移钉死：
- register 版本必须等于 metadata.yaml 的 version（且不得硬编码字面量）；
- 代码里 cfg.get 读取的每个配置键都必须出现在 _conf_schema.json
  （⚠️ 正则必须含数字：tier0_threshold / glm5 这类键名带数字，
  首版检查脚本用 [a-z_]+ 漏掉了它们，差点漏报真实缺口）。
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

PLUGIN_ROOT = Path(__file__).resolve().parents[1]


def _metadata_version() -> str:
    import yaml
    meta = yaml.safe_load((PLUGIN_ROOT / "metadata.yaml").read_text(encoding="utf-8"))
    return str(meta["version"])


class TestRegisterVersionSync:
    """@register 的版本号必须跟 metadata.yaml 走（v0.2.6 起动态读取）。"""

    def test_register_uses_metadata_version(self):
        from astrbot_plugin_mnemoria import main
        got = main._metadata_version()
        assert got == _metadata_version(), \
            "register 版本号与 metadata.yaml 不一致（面板会显示旧版本）"

    def test_register_not_hardcoded_literal(self):
        """装饰器实参必须是 _metadata_version()，不能再写死字面量。"""
        src = (PLUGIN_ROOT / "main.py").read_text(encoding="utf-8")
        block = src.split("@register(", 1)[1].split("class MnemoriaPlugin", 1)[0]
        assert "_metadata_version()" in block, \
            "@register 又变成硬编码版本号了——请改回 _metadata_version()"
        assert not re.search(r'"\d+\.\d+\.\d+"', block), \
            "@register 里出现了写死的版本号字面量——请改回 _metadata_version()"


class TestConfSchemaCompleteness:
    """代码读取的配置键必须都在 _conf_schema.json 里（面板可见可调）。"""

    # 整字典读取（子键已在 schema 逐条定义）与测试夹具专用键，白名单豁免
    _EXEMPT = {
        "decay_policy.type_weights",   # 整本字典读后按类型下标，子键已逐条定义
        "provider_id",                 # 顶层键，读法同 group 名
    }

    def _schema_keys(self) -> set[str]:
        raw = json.loads((PLUGIN_ROOT / "_conf_schema.json").read_text(encoding="utf-8"))
        keys: set[str] = set()

        def walk(d: dict, prefix: str = "") -> None:
            for k, v in d.items():
                if not isinstance(v, dict):
                    continue
                if isinstance(v.get("items"), dict):
                    walk(v["items"], f"{prefix}{k}.")
                elif "type" in v:
                    keys.add(f"{prefix}{k}")

        walk(raw)
        return keys

    def test_every_read_key_is_documented(self):
        schema = self._schema_keys()
        used: set[str] = set()
        # ⚠️ 字符类必须含 0-9：tier0_threshold / glm5 / v4 这类键名带数字，
        # [a-z_]+ 会整键漏扫（2026-09-22 审查时我自己的检查脚本就栽在这）
        pat = re.compile(r'(?:cfg|config|self\.config)\.get\(\s*"([a-z0-9_]+\.[a-z0-9_]+)"')
        for py in PLUGIN_ROOT.rglob("*.py"):
            if "tests" in py.parts:
                continue
            for m in pat.finditer(py.read_text(encoding="utf-8", errors="ignore")):
                used.add(m.group(1))
        missing = sorted(k for k in used - schema if k not in self._EXEMPT)
        assert not missing, (
            f"这些配置键代码读了但 _conf_schema.json 没定义（面板不可见）: {missing}"
        )

    def test_schema_has_the_v023_guards(self):
        """v0.2.3 两个守卫键必须带默认值在 schema 里（本轮审查的真实缺口）。

        v0.2.11：conservative_fallback_similarity 默认 0.85→0.90（dashscope
        qwen3.7-flash 实测不同事实峰值 0.86，0.85 会误强并丢事实）。
        """
        schema = json.loads(
            (PLUGIN_ROOT / "_conf_schema.json").read_text(encoding="utf-8"))
        adm = schema["admission"]["items"]
        for key, default in (("text_dedup_similarity", 0.80),
                             ("conservative_fallback_similarity", 0.90)):
            assert key in adm, f"admission.{key} 未在 schema 定义"
            assert float(adm[key].get("default", -1)) == default, \
                f"admission.{key} 默认值应为 {default}"
