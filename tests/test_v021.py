"""v0.2.1 画像体系与解析加固回归测试（借鉴 angel_memory）。

覆盖：
1. 画像固定五维归一（写入路径：抽取 / 工具）；
2. 画像展示按维度聚合（历史漂移键归并，省注入预算）；
3. 抽取提示词注入已有画像 + 空画像硬底线；
4. 提示词含 tags 场合词案例与「」引号约定；
5. 退休规则移植（必须保留 / 可以删除 / 不构成删除理由）；
6. JSON 解析多候选加固；
7. 画像维度归并脚本（dry-run / apply + 快照 + 不碰未知键）。
"""

from __future__ import annotations

import asyncio
import json
import sqlite3
import sys
from pathlib import Path

PLUGIN = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PLUGIN / "scripts"))


class FakeLLM:
    enabled = True
    provider_id = "test-v021"

    def __init__(self, payload):
        self.payload = payload
        self.prompts: list[str] = []

    async def generate_json(self, prompt, system_prompt=None):
        self.prompts.append(prompt)
        return self.payload

    async def generate(self, prompt, system_prompt=None):
        return ""


def _mk(tmp_path, conf=None, llm=None, emb=None, name="pdv021"):
    from core import config as cfgmod, db as dbm
    from core.engine import MemoryEngine
    from core.paths import DataPaths
    from core.store import MemoryStore

    paths = DataPaths(tmp_path / name).ensure()
    conn = dbm.connect(paths.db)
    dbm.init_schema(conn)
    store = MemoryStore(conn)
    eng = MemoryEngine(store, cfgmod.Config(conf or {}, paths.meta),
                       embedder=emb, llm=llm)
    return eng, store, conn, paths


class _Emb:
    enabled = True

    async def embed_one(self, text, timeout=None):
        return [1.0, 0.0, 0.0]

    async def embed(self, texts):
        return [[1.0, 0.0, 0.0] for _ in texts]


# ------------------------------------------------------------------ 1. 维度归一
class TestProfileTaxonomy:
    def test_normalize_mapping(self):
        from core import profile_taxonomy as ptax
        cases = {
            "称呼": "用户别名", "昵称": "用户别名",
            "喜好": "事实属性", "作息": "事实属性", "使用习惯": "事实属性",
            "技能": "技能树", "关系状态": "关系图谱", "业务动态": "活跃项目",
            "用户别名": "用户别名", "事实属性": "事实属性",
            "自定义维度": "自定义维度",  # 未知键不猜
            "": "", "名字": "名字",  # 名字=助手人设占用键，刻意不映射
        }
        for raw, want in cases.items():
            assert ptax.normalize_profile_key(raw) == want, (raw, want)
        assert ptax.is_canonical("用户别名") and not ptax.is_canonical("称呼")

    def test_facts_merge_not_overwrite(self, tmp_path):
        """事实类维度是累积型：新增追加去重，不整行覆盖（防模型只写片段冲掉旧值）。"""
        eng, store, conn, _ = _mk(tmp_path)
        try:
            eng.write_profile("default", "u1", "事实属性", "喜欢深夜听广播")
            eng.write_profile("default", "u1", "事实属性", "对猫过敏")
            eng.write_profile("default", "u1", "事实属性", "喜欢深夜听广播")  # 重复不追加
            row = [r for r in store.get_profile("default", "u1")
                   if r["key"] == "事实属性"][0]
            assert row["value"] == "喜欢深夜听广播；对猫过敏"
        finally:
            conn.close()

    def test_alias_overwrites(self, tmp_path):
        """用户别名是覆盖型：新称呼覆盖旧称呼，同义键就地清理。"""
        eng, store, conn, _ = _mk(tmp_path)
        try:
            eng.write_profile("default", "u1", "用户别名", "小明")
            eng.write_profile("default", "u1", "称呼", "阿明")
            rows = {r["key"]: r["value"] for r in store.get_profile("default", "u1")}
            assert rows == {"用户别名": "阿明"}, rows
        finally:
            conn.close()

    def test_unknown_key_overwrite_not_merge(self, tmp_path):
        """未知键（含助手人设历史占用键「名字」）保持 v0.2.0 覆盖语义：
        不合并追加，避免人设/自定义键累积多个历史版本（再审发现）。"""
        eng, store, conn, _ = _mk(tmp_path)
        try:
            eng.write_profile("default", "u1", "名字", "星棠(青鸾)")
            eng.write_profile("default", "u1", "名字", "星棠")
            eng.write_profile("default", "u1", "自定义维度", "旧值")
            eng.write_profile("default", "u1", "自定义维度", "新值")
            rows = {r["key"]: r["value"] for r in store.get_profile("default", "u1")}
            assert rows["名字"] == "星棠", rows
            assert rows["自定义维度"] == "新值", rows
        finally:
            conn.close()

    def test_merge_truncation_keeps_pieces_intact(self, tmp_path):
        """合并限长必须按片段边界截断，不允许把片段切成半截（再审发现）。
        注：max_total 有 200 字下限保护，用例用 90 字片段真实触界。"""
        eng, store, conn, _ = _mk(tmp_path)
        try:
            store.merge_profile("default", "u1", "事实属性",
                                "甲" * 90, max_total=200)
            store.merge_profile("default", "u1", "事实属性",
                                "乙" * 90, max_total=200)
            store.merge_profile("default", "u1", "事实属性",
                                "丙" * 90, max_total=200)
            val = [r["value"] for r in store.get_profile("default", "u1")
                   if r["key"] == "事实属性"][0]
            pieces = val.split("；")
            assert pieces == ["甲" * 90, "乙" * 90], pieces  # 丙装不下整条丢弃
            assert "丙" not in val
        finally:
            conn.close()

    def test_aggregate_rows(self):
        from core import profile_taxonomy as ptax
        rows = [
            {"key": "兴趣", "value": "跑步"},
            {"key": "事实属性", "value": "爱运动"},
            {"key": "喜好", "value": "跑步"},   # 与兴趣同值，去重
            {"key": "自定义维度", "value": "保留"},
        ]
        out = dict(ptax.aggregate_profile_rows(rows, limit=10, max_value=240))
        assert out["事实属性"] == "跑步；爱运动"
        assert out["自定义维度"] == "保留"

    def test_profile_block_aggregates(self, make_engine):
        eng, store, conn = make_engine()
        try:
            store.upsert_profile("default", "u1", "喜好", "猫", confidence=0.6)
            store.upsert_profile("default", "u1", "兴趣", "跑步", confidence=0.9)
            store.upsert_profile("default", "u1", "事实属性", "爱运动", confidence=0.8)
            block = eng.profile_block("default", "u1")
            assert "事实属性：" in block
            assert "跑步" in block and "爱运动" in block and "猫" in block
            assert "喜好" not in block and "兴趣" not in block, "同义键应归并到固定维度"
        finally:
            conn.close()

    def test_update_profile_normalizes_and_migrates(self, tmp_path):
        llm = FakeLLM({"memories": [], "profile": [
            {"key": "称呼", "value": "晓晓", "confidence": 0.9}]})
        eng, store, conn, _ = _mk(tmp_path, {"profile": {"auto_extract": True}},
                                  llm=llm, emb=_Emb())
        try:
            store.upsert_profile("default", "u1", "称呼", "宝宝", confidence=0.5)
            eng.record_turn("s", "user", "叫我晓晓就好", scope="default")
            asyncio.run(eng.extract_session("s", scope="default", user_key="u1"))
            keys = {r["key"]: r["value"] for r in store.get_profile("default", "u1")}
            assert keys.get("用户别名") == "晓晓", "抽取写入必须归一到固定维度"
            assert "称呼" not in keys, "被替换的同义旧键必须删除，防并存漂移"
        finally:
            conn.close()

    def test_profile_tool_normalizes(self, make_engine):
        from tools.profile import ProfileUpdateTool
        eng, store, conn = make_engine()

        class _Ev:
            mnemoria_engine = None

            def get_sender_id(self):
                return "u1"

            def get_session_id(self):
                return "s1"

        try:
            ev = _Ev()
            ev.mnemoria_engine = eng
            store.upsert_profile("default", "u1", "称呼", "宝宝")
            out = asyncio.run(ProfileUpdateTool().run(ev, key="称呼", value="晓晓"))
            assert "用户别名" in out
            keys = {r["key"]: r["value"] for r in store.get_profile("default", "u1")}
            assert keys.get("用户别名") == "晓晓" and "称呼" not in keys
        finally:
            conn.close()

    def test_user_label_reads_canonical_alias(self, make_engine):
        eng, store, conn = make_engine()
        try:
            store.upsert_profile("default", "u1", "用户别名", "阿崇")
            label = eng._user_label("default", "u1")
            assert "阿崇" in label and "u1" in label
            # 句子型脏值（旧数据）不得当称呼
            store.upsert_profile("default", "u3", "用户别名",
                                 "这是一句很长的描述，不应该被当成称呼使用。")
            assert "这是一句" not in eng._user_label("default", "u3")
            # 人设占用键仍不得参与指称（v0.1.6 红线）
            store.upsert_profile("default", "u2", "名字", "星棠")
            assert "星棠" not in eng._user_label("default", "u2")
        finally:
            conn.close()


# ------------------------------------------------------------------ 2. 提示词
class TestPromptsV021:
    def test_extract_prompt_has_profile_section(self, tmp_path):
        llm = FakeLLM({"memories": [], "profile": []})
        eng, store, conn, _ = _mk(tmp_path, {}, llm=llm, emb=_Emb())
        try:
            store.upsert_profile("default", "u1", "用户别名", "阿崇")
            store.upsert_profile("default", "u1", "喜好", "跑步")
            eng.record_turn("s", "user", "今天聊了聊最近的生活近况", scope="default")
            asyncio.run(eng.extract_session("s", scope="default", user_key="u1"))
            prompt = llm.prompts[0]
            assert "该用户已有画像" in prompt
            assert "用户别名：阿崇" in prompt
            assert "事实属性：跑步" in prompt, "已有画像应按固定维度聚合后注入"
        finally:
            conn.close()

    def test_empty_profile_floor(self, tmp_path):
        llm = FakeLLM({"memories": [], "profile": []})
        eng, store, conn, _ = _mk(tmp_path, {}, llm=llm, emb=_Emb())
        try:
            eng.record_turn("s", "user", "今天聊了聊最近的生活近况", scope="default")
            asyncio.run(eng.extract_session("s", scope="default", user_key="u1"))
            prompt = llm.prompts[0]
            assert "（无）" in prompt
            assert "空画像硬底线" in prompt
        finally:
            conn.close()

    def test_extract_prompt_taxonomy_and_style(self):
        from core import templates
        p = templates.EXTRACT_PROMPT
        for kw in ("用户别名", "事实属性", "技能树", "关系图谱", "活跃项目"):
            assert kw in p, f"固定五维缺 {kw}"
        for kw in ("空画像硬底线", "同维度同 key", "冲突修正", "引号约定", "「」"):
            assert kw in p, f"规则缺 {kw}"
        assert "场合词" in p and "例 1" in p and "例 3" in p, "tags 案例必须保留"

    def test_retire_prompt_has_angel_rules(self):
        from core import templates
        p = templates.RETIRE_PROMPT
        for kw in ("用户画像", "助理自身的身份", "约定与承诺", "不可再生的经历",
                   "不能作为 delete 的理由", "从未被召回", "「」"):
            assert kw in p, f"退休规则缺 {kw}"


# ------------------------------------------------------------------ 3. JSON 解析
class TestParseJsonRobust:
    def test_fenced_and_plain(self):
        from core.llm import parse_json_loose
        assert parse_json_loose('```json\n{"x": 1}\n```') == {"x": 1}
        assert parse_json_loose('{"x": 1}') == {"x": 1}

    def test_nested_braces_and_quotes_in_strings(self):
        from core.llm import parse_json_loose
        text = '前缀 {"a": "他说「{这个}」", "b": {"c": [1, 2]}} 后缀'
        obj = parse_json_loose(text)
        assert obj["a"] == "他说「{这个}」" and obj["b"]["c"] == [1, 2]

    def test_skips_invalid_first_brace_region(self):
        from core.llm import parse_json_loose
        # 前面有非 JSON 的 {} 噪音，后面才是真 JSON
        text = '说明 {这不是 JSON} 结果: {"x": 1}'
        assert parse_json_loose(text) == {"x": 1}

    def test_unbalanced_prefix(self):
        from core.llm import parse_json_loose
        text = '前面有个没闭合的 { 噪音，然后 {"x": 1}'
        assert parse_json_loose(text) == {"x": 1}

    def test_scoring_prefers_required_fields(self):
        from core.llm import parse_json_loose
        text = '{"noise": 1} 然后 {"memories": [{"content": "a"}]}'
        obj = parse_json_loose(text, required=("memories",))
        assert isinstance(obj, dict) and "memories" in obj
        assert obj["memories"][0]["content"] == "a"

    def test_multiline_model_like_output(self):
        from core.llm import parse_json_loose
        text = (
            "好的，下面是结果：\n"
            "```json\n"
            "{\n  \"action\": \"merge\",\n  \"target_ids\": [\"1\"],\n"
            "  \"content\": \"小明喜欢猫科动物\",\n  \"confidence\": 0.9\n}\n"
            "```\n以上。"
        )
        obj = parse_json_loose(text, required=("action",))
        assert obj["action"] == "merge" and obj["confidence"] == 0.9


# ------------------------------------------------------------------ 4. 归并脚本
class TestProfileMigrationScript:
    def _seed(self, tmp_path):
        from core import db as dbm
        from core.paths import DataPaths
        paths = DataPaths(tmp_path / "pdmig").ensure()
        conn = dbm.connect(paths.db)
        dbm.init_schema(conn)
        conn.execute(
            "INSERT INTO profiles(scope,user_key,key,value,confidence,updated_at) "
            "VALUES('default','u1','称呼','宝宝',0.9,100),"
            "('default','u1','喜好','猫',0.6,200),"
            "('default','u1','兴趣','跑步',0.8,300),"
            "('default','u1','事实属性','爱运动',0.7,400),"
            "('default','u1','自定义维度','保留我',0.5,500)")
        conn.commit()
        return conn, paths

    def test_dry_run_and_apply(self, tmp_path):
        import migrate_profile_taxonomy as mig
        conn, paths = self._seed(tmp_path)
        try:
            groups, untouched = mig.plan(conn)
            assert ("default", "u1", "用户别名") in groups
            assert len(untouched) == 1 and untouched[0]["key"] == "自定义维度"
            before = {r["key"]: r["value"] for r in conn.execute("SELECT key,value FROM profiles")}
            assert before["称呼"] == "宝宝"  # dry-run 不改库
            snap, merged = mig.apply(conn, groups, paths.backups)
            assert snap.exists()
            payload = json.loads(snap.read_text(encoding="utf-8"))
            assert any(r["key"] == "称呼" for r in payload["profiles"]), "快照须含迁移前数据"
            keys = {r["key"]: r["value"] for r in conn.execute("SELECT key,value FROM profiles")}
            assert keys["用户别名"] == "宝宝" and "称呼" not in keys
            assert keys["事实属性"] == "爱运动；跑步；猫", f"按 updated_at 降序归并，实得 {keys['事实属性']}"
            assert keys["自定义维度"] == "保留我", "未知键不得被合并"
        finally:
            conn.close()
