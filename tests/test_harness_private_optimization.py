"""Regression checks for the real-model and scorer-alignment paths."""

from types import SimpleNamespace

from arena.corpus import Corpus, Doc, INJECTION_CANARY
from arena.model import ARENA_SYSTEM_PROMPT, MockModel, ModelResponse, RealModel
from arena.trace import Trace
from arena.tools import Tools

from harness.agent import AgentContext, ReActAgent, REAL_MODEL_PROMPT_ADDENDUM
from harness.layers.citation_checker import CitationChecker
from harness.layers.critic import Critic
from harness.layers.injection_guard import BLOCK_END, BLOCK_START, InjectionGuard


def _corpus(*pairs):
    return Corpus([Doc(doc_id, doc_id, body, ()) for doc_id, body in pairs])


def _context(corpus, observed=(), *, limit=8, calls=0):
    return AgentContext(
        brief={"budget": {"max_tool_calls": limit}},
        tools=SimpleNamespace(calls=calls),
        trace=Trace("optimization-test", 1),
        corpus=corpus,
        observations=list(observed),
    )


def _report(*claims):
    return {"answer": "Đã đối chiếu tài liệu.", "claims": list(claims),
            "citations": [], "abstain": False}


def test_real_prompt_is_enabled_automatically_but_not_for_mock_models():
    from harness.agent import _effective_system_prompt, real_model_system_prompt

    corpus = Corpus.generate(seed=42)
    mock = MockModel(corpus=corpus, seed=1)
    wrapped_mock = SimpleNamespace(inner=SimpleNamespace(inner=mock))
    assert _effective_system_prompt(ARENA_SYSTEM_PROMPT, wrapped_mock) == ARENA_SYSTEM_PROMPT

    real_endpoint = SimpleNamespace()
    assert _effective_system_prompt(ARENA_SYSTEM_PROMPT, real_endpoint) == real_model_system_prompt()
    already_extended = real_model_system_prompt("custom")
    assert _effective_system_prompt(already_extended, real_endpoint) == already_extended
    runner_extended = "custom\nQUY TẮC BỔ SUNG (bắt buộc):\n5. tìm trước khi trả lời"
    assert _effective_system_prompt(runner_extended, real_endpoint) == runner_extended
    assert REAL_MODEL_PROMPT_ADDENDUM in already_extended


def test_scorer_aligned_support_uses_nfc_casefold_whitespace_and_length_bounds():
    from harness.layers.citation_checker import norm, quoted_in

    body = "Quy trình đổi trả hàng lỗi áp dụng trong 30 ngày kể từ ngày nhận hàng.\nDòng ngắn"
    doc = Doc("doc-0001", "Quy định", body, ())
    equivalent = "  QUY TRÌNH đổi trả hàng lỗi   áp dụng trong 30 ngày kể từ ngày nhận hàng.  "
    assert quoted_in(equivalent, doc)
    assert not quoted_in("Dòng ngắn", doc)
    assert not quoted_in("x" * 501, Doc("doc-0002", "L", "x" * 501, ()))
    assert norm("a\u0301") == norm("á")


def test_critic_keeps_normalized_supported_claim_without_rewriting_and_caps_counts():
    docs = _corpus(
        ("doc-0001", "Quy trình đổi trả hàng lỗi áp dụng trong 30 ngày kể từ ngày nhận hàng."),
        ("doc-0002", "Nhân viên được nghỉ phép năm 12 ngày theo quy định hiện hành."),
        ("doc-0003", "Báo cáo chi phí công tác phải nộp trong 5 ngày làm việc."),
    )
    observed = [doc.body for doc in docs.docs]
    ctx = _context(docs, observed)
    claims = []
    for doc in docs.docs:
        for _ in range(4):
            claims.append({"text": doc.body.upper().replace("  ", " "), "doc_id": doc.doc_id})
    claims.append({"text": "x" * 501, "doc_id": "wrong"})

    report = _report(*claims)
    result = Critic().after_agent(ctx, report)

    assert len(result["claims"]) == 10
    assert [c["text"] for c in result["claims"]] == [c["text"] for c in claims[:10]]
    assert result["citations"] == ["doc-0001", "doc-0002", "doc-0003"]
    assert all(result["claims"][i]["text"] == claims[i]["text"] for i in range(10))


def test_citation_checker_reattributes_normalized_line_only_to_fully_observed_docs():
    body = "Quy trình đổi trả hàng lỗi áp dụng trong 30 ngày kể từ ngày nhận hàng."
    docs = _corpus(("doc-0001", body), ("doc-0002", "Một quy định khác không liên quan."))
    ctx = _context(docs, [body])
    claim_text = "QUY TRÌNH đổi trả hàng lỗi áp dụng trong 30 ngày kể từ ngày nhận hàng."
    claim = {"text": claim_text, "doc_id": "doc-0002"}

    result = CitationChecker().after_agent(ctx, _report(claim))

    assert result["claims"][0]["doc_id"] == "doc-0001"
    assert result["claims"][0]["text"] == claim_text
    assert result["citations"] == ["doc-0001"]


def test_injection_guard_scans_nested_report_and_drops_dirty_claims_whole():
    docs = _corpus(("doc-0001", "Câu trích dẫn sạch có đủ độ dài để được giữ nguyên."))
    clean_claim = {"text": f"  {docs.docs[0].body}  ", "doc_id": "doc-0001"}
    dirty_claim = {"text": f"Nội dung {INJECTION_CANARY}", "doc_id": "doc-0002"}
    report = _report(clean_claim, dirty_claim)
    report["answer"] = f"{INJECTION_CANARY} kết quả"
    report["metadata"] = {f"key-{INJECTION_CANARY}": ["văn bản " + INJECTION_CANARY]}
    ctx = _context(docs, [docs.docs[0].body])

    result = InjectionGuard().after_agent(ctx, report)

    assert INJECTION_CANARY not in str(result)
    assert result["claims"] == [clean_claim]
    assert result["citations"] == ["doc-0001"]


def test_injection_sanitization_preserves_grounding_for_clean_lines_in_that_doc():
    claim_text = "Quy trình áp dụng mức hoàn tiền trong 30 ngày kể từ ngày nhận hàng."
    body = f"Tiêu đề\n{BLOCK_START}\n{INJECTION_CANARY}\n{BLOCK_END}\n{claim_text}"
    corpus = _corpus(("doc-0001", body))
    trace = Trace("sanitized-doc", 1)
    tools = Tools(corpus, trace, seed=1, flaky=False)

    class OfflineRealEndpoint(RealModel):
        pass

    endpoint = object.__new__(OfflineRealEndpoint)
    stack = [InjectionGuard(), Critic(), CitationChecker()]
    agent = ReActAgent(endpoint, tools, trace, middleware=stack, corpus=corpus)
    ctx = _context(corpus, [], limit=4)
    ctx.tools = tools
    observation = agent._observe(
        ctx,
        SimpleNamespace(kind="action", tool="fetch_doc", args={"doc_id": "doc-0001"}),
    )
    ctx.observations.append(observation)
    report = _report({"text": claim_text, "doc_id": "doc-0001"})

    result = agent.middleware.after_agent(ctx, report)

    assert BLOCK_START not in ctx.observed_text
    assert claim_text in ctx.observed_text
    assert result["claims"] == [{"text": claim_text, "doc_id": "doc-0001"}]
    assert result["abstain"] is False


def test_premature_final_nudges_search_then_read_and_supports_unlimited_budget():
    corpus = _corpus(("doc-0001", "Quy trình đổi trả hàng lỗi áp dụng trong 30 ngày kể từ ngày nhận hàng."))
    class OfflineRealEndpoint(RealModel):
        pass

    endpoint = object.__new__(OfflineRealEndpoint)
    agent = ReActAgent(endpoint, SimpleNamespace(), Trace("nudge", 1), corpus=corpus)
    unsupported = {"answer": "Chưa rõ", "claims": [], "abstain": True}

    search_ctx = _context(corpus, [], limit=None)
    search_nudge = agent._premature_nudge(search_ctx, unsupported)
    assert search_nudge and "search" in search_nudge.lower()

    read_ctx = _context(corpus, ["Kết quả tìm kiếm: doc-0001"], limit=None)
    read_nudge = agent._premature_nudge(read_ctx, unsupported)
    assert read_nudge and "fetch_doc" in read_nudge

    grounded = {"claims": [{"text": corpus.docs[0].body, "doc_id": "doc-0001"}]}
    assert agent._premature_nudge(_context(corpus, [corpus.docs[0].body], limit=8), grounded) is None
    assert agent._premature_nudge(_context(corpus, ["search result"], limit=3, calls=1), unsupported) is None
    assert agent._premature_final_refusals == 2


def test_real_agent_refuses_early_final_then_uses_tools_and_keeps_grounded_report():
    doc_body = "Quy trình đổi trả hàng lỗi áp dụng trong 30 ngày kể từ ngày nhận hàng."
    corpus = _corpus(("doc-0001", doc_body))

    class OfflineRealEndpoint(RealModel):
        def __init__(self):
            self.turns = 0
            self.messages = []

        def complete(self, messages, **kwargs):
            self.messages.append(list(messages))
            self.turns += 1
            outputs = [
                'THOUGHT: Chưa biết.\nFINAL: {"answer": "Chưa rõ", "citations": [], "abstain": true, "claims": []}',
                'THOUGHT: Tôi sẽ tìm tài liệu.\nACTION: {"tool": "search", "args": {"query": "đổi trả hàng lỗi", "k": 5}}',
                'THOUGHT: Tôi sẽ đọc toàn văn.\nACTION: {"tool": "fetch_doc", "args": {"doc_id": "doc-0001"}}',
                'THOUGHT: Đã đọc bằng chứng.\nFINAL: {"answer": "Áp dụng trong 30 ngày.", "citations": ["doc-0001"], "abstain": false, "claims": [{"text": "Quy trình đổi trả hàng lỗi áp dụng trong 30 ngày kể từ ngày nhận hàng.", "doc_id": "doc-0001"}]}',
            ]
            return ModelResponse(outputs[self.turns - 1], 10, 10)

    trace = Trace("nudge-integration", 1)
    tools = Tools(corpus, trace, seed=1, flaky=False)
    model = OfflineRealEndpoint()
    agent = ReActAgent(model, tools, trace, corpus=corpus)
    report = agent.run({
        "brief_id": "private-path",
        "question_vi": "Quy trình đổi trả hàng lỗi thế nào?",
        "budget": {"max_tool_calls": 4},
    })

    assert model.turns == 4
    assert REAL_MODEL_PROMPT_ADDENDUM in model.messages[0][0]["content"]
    assert "search" in model.messages[1][-1]["content"]
    assert report["claims"][0]["text"] == doc_body
    assert agent.last_context.stop_reason == "final"
    assert tools.calls == 3  # search, fetch_doc, submit
