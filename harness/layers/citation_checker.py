"""LỚP `citation_checker` — bài giảng Day 16, §11 (Grounding & Citations).

NHIỆM VỤ: chỉ cần MỘT tài liệu gắn nhãn `lookalike` hoặc `outdated` lọt
vào bằng chứng là mô hình neo TOÀN BỘ claim vào đúng tài liệu trông có vẻ
"chính thống" đó — dù mỗi câu được lấy nguyên văn từ một tài liệu khác.
Câu thì thật, trích dẫn thì sai. Đây là kiểu sai nguy hiểm nhất trong RAG
vì báo cáo đọc vào vẫn rất thuyết phục.

TÍN HIỆU (chính xác, không cần đoán):

    claim["text"] KHÔNG khớp theo chuẩn hoá của scorer với một DÒNG nào trong
    corpus.get(claim["doc_id"]).body
    nhưng CHÍNH câu đó CÓ trong bằng chứng agent đã quan sát

Chú ý chữ DÒNG: kiểm tra `claim["text"] in doc.body` (cả khối, không
tách dòng) là SAI — scorer chỉ nhận trích dẫn khớp MỘT DÒNG sau khi
chuẩn hoá NFC, hoa thường và khoảng trắng (xem "ĐƯỢC PHÉP VÀ KHÔNG ĐƯỢC
PHÉP" ngay dưới đây). `in doc.body` coi
một câu vắt qua hai dòng là hợp lệ, trong khi scorer thì không — tín
hiệu kiểu đó khiến bạn giữ nguyên một trích dẫn mà scorer vẫn chấm
`HALLUCINATED`.

Vế thứ hai mới là phần quan trọng: nó tách việc của bạn khỏi việc của
`critic` (§2). Câu có trong bằng chứng nhưng gắn sai tài liệu -> GẮN LẠI
(việc của bạn). Câu không có trong bằng chứng nào -> BỊA, để `critic` xoá.
Hai điều kiện loại trừ nhau nên hai lớp không giành điểm của nhau.

ĐƯỢC PHÉP VÀ KHÔNG ĐƯỢC PHÉP:
  * ĐƯỢC: đổi `claim["doc_id"]`, cập nhật `report["citations"]`.
  * KHÔNG: sửa `claim["text"]`. Scorer chỉ cho điểm khi câu khớp một
    DÒNG trong tài liệu được trích sau khi chuẩn hoá, VÀ đúng là chữ mô
    hình đã viết. Thêm dấu chấm, đổi từ, hay vá lại câu bị cắt bằng nội
    dung lấy từ corpus đều làm mất cả hai điều kiện cùng lúc.

CHỈ ĐƯỢC GẮN VÀO TÀI LIỆU ĐÃ QUAN SÁT. Trích một tài liệu mà lượt chạy
chưa từng đọc bị chấm `UNRETRIEVED`. Vì vậy hãy tìm nguồn trong
`ctx.observed_text`, đừng quét cả corpus rồi gắn bừa. Lớp ghi nhận ID từ
kết quả công cụ đã qua middleware, rồi xác nhận nguyên câu trích vẫn còn
trong phần mô hình đã thấy; cách này cũng giữ được câu sạch khi
`InjectionGuard` gỡ một khối khỏi tài liệu.

CÔNG CỤ CÓ SẴN:
    ctx.observed_text  -> toàn bộ quan sát agent đã thấy, nối lại
    ctx.corpus.get(doc_id) -> Doc | None
    ctx.corpus.docs    -> danh sách Doc (doc_id, title, body); qua
                          `ctx.corpus`, `Doc.tags` LUÔN RỖNG — CẢ Ở VÒNG
                          LUYỆN TẬP LẪN VÒNG CHẤM ĐIỂM, vì corpus mà code
                          của bạn cầm bị gỡ nhãn bẫy ('outdated',
                          'contradiction', 'injection'…) ngay khi runner
                          dựng lên nó, không phải chỉ lúc chấm điểm. Đọc
                          nhãn là tra bảng chứ không phải kỹ năng lab này
                          chấm. Ở vòng LUYỆN TẬP seed 42 thì file TRÊN ĐĨA
                          `data/corpus/*.json` (khác với `ctx.corpus`)
                          vẫn có nhãn: hard-code được từ đó, và điều đó
                          được nói thẳng ra ở đây thay vì giấu đi.

Cài đặt:  ReActAgent(..., middleware=[..., CitationChecker(), ...])
Xem `harness/middleware.py` để biết thứ tự các hook.
"""

from __future__ import annotations

import re
import unicodedata

from harness.middleware import Middleware


_WHITESPACE = re.compile(r"\s+")
MIN_SUPPORT_CHARS = 12
MAX_CLAIM_CHARS = 500


def norm(text: str) -> str:
    """Match the scorer's NFC, case-folded, collapsed-whitespace form."""
    if not isinstance(text, str):
        text = "" if text is None else str(text)
    return _WHITESPACE.sub(" ", unicodedata.normalize("NFC", text).casefold()).strip()


def quoted_in(text: str, doc) -> bool:
    """Whether a claim is a scorer-valid quote from one line of ``doc``."""
    claim = norm(text)
    if not MIN_SUPPORT_CHARS <= len(claim) <= MAX_CLAIM_CHARS:
        return False
    body = getattr(doc, "body", "")
    return any(claim in norm(line) for line in body.splitlines())


def _quote_was_observed(text: str, ctx) -> bool:
    claim = norm(text)
    if not claim:
        return False
    return any(
        claim in norm(line)
        for observation in ctx.observations
        if isinstance(observation, str)
        for line in observation.splitlines()
    )


def source_of(text: str, ctx) -> list[str]:
    """Find observed documents that support ``text`` as one scorer-valid line.

    A fetched document can be shortened by a tool layer such as
    ``InjectionGuard``. In that case the document ID proves which source
    was read, while the observation check below proves the quoted line
    itself remained visible to the model.
    """
    corpus = getattr(ctx, "corpus", None)
    if corpus is None or not MIN_SUPPORT_CHARS <= len(norm(text)) <= MAX_CLAIM_CHARS:
        return []
    observed = ctx.observed_text
    observed_ids = getattr(ctx, "state", {}).get("observed_doc_ids", set())
    if not isinstance(observed_ids, (set, frozenset, list, tuple)):
        observed_ids = set()
    observed_ids = set(observed_ids)
    observed_quote = _quote_was_observed(text, ctx)
    if not observed_quote:
        return []
    sources = []
    for doc in corpus.docs:
        if observed_ids:
            was_observed = doc.doc_id in observed_ids
        else:
            was_observed = (
                isinstance(getattr(doc, "body", None), str) and doc.body in observed
            )
        if was_observed and quoted_in(text, doc):
            sources.append(doc.doc_id)
    return sources


class CitationChecker(Middleware):
    """Trỏ mỗi claim về đúng tài liệu thật sự chứa câu đó."""

    name = "citation_checker"

    def after_agent(self, ctx, report):
        claims = report.get("claims")
        if not isinstance(claims, list) or not claims or ctx.corpus is None:
            return report

        for claim in claims:
            if not isinstance(claim, dict) or not isinstance(claim.get("text"), str):
                continue
            text = claim["text"]
            doc_id = claim.get("doc_id")
            current = ctx.corpus.get(doc_id) if isinstance(doc_id, str) else None
            if (
                current is not None
                and current.body in ctx.observed_text
                and quoted_in(text, current)
            ):
                continue
            sources = source_of(text, ctx)
            if sources:
                claim["doc_id"] = sources[0]
        report["citations"] = sorted(
            {claim["doc_id"] for claim in claims
             if isinstance(claim, dict) and isinstance(claim.get("doc_id"), str)}
        )
        return report
