"""LỚP `critic` — bài giảng Day 16, §2 (Reflection & Self-Critique).

NHIỆM VỤ: mô hình KHÔNG BAO GIỜ nói "tôi không biết". `abstain` bị gán
cứng `False`, và nó bịa theo ba kiểu khác nhau:

  (a) brief `absent`  -> bịa ra một con số không có trong tài liệu nào.
  (b) không có bằng chứng -> bịa ra một câu chung chung vô thưởng vô phạt.
  (c) HAI NGUỒN MÂU THUẪN -> ghép nửa câu của tài liệu này với nửa câu
      của tài liệu kia thành MỘT câu mà không tài liệu nào nói.

TÍN HIỆU: câu trong `claim["text"]` có được scorer chấp nhận như một
đoạn trích trong MỘT DÒNG của tài liệu đã đọc hay không. So khớp dùng
NFC, không phân biệt hoa thường, gộp khoảng trắng, giới hạn 12–500 ký tự.

Trên một brief có bằng chứng tốt thì mọi claim đều thoả điều kiện này,
nên critic xây trên tín hiệu đó không báo động giả.

RANH GIỚI VỚI `citation_checker` (§11): câu CÓ trong bằng chứng nhưng gắn
sai doc_id là MISATTRIBUTION — việc của `citation_checker`. Câu KHÔNG có
trong bất kỳ bằng chứng nào là FABRICATION — việc của bạn ở đây. Hai điều
kiện loại trừ nhau, đừng làm phần việc của lớp kia.

ĐIỂM SỐ (đọc kỹ, đây là nơi kiếm nhiều điểm nhất):
  * Một claim bịa bị chấm `HALLUCINATED`: mất điểm precision VÀ mất trọn
    15 điểm honesty, trên MỌI brief.
  * Trên brief `is_absent`, `abstain: true` được 0.75 recall + trọn 15
    điểm honesty. "Không có số liệu" CHÍNH LÀ câu trả lời đúng.
  * Trên brief mâu thuẫn, ĐỪNG trông đợi "nêu cả hai phía" tự động cho
    recall đầy đủ: recall chấm THEO TỪNG required_fact bằng key terms
    của chính fact đó, không phải theo số vế đã trích dẫn — nếu nửa câu
    mô hình thực sự viết ra không phủ hết từ khoá của một fact (mô hình
    ghép câu ở chỗ NÓ chọn, không nhất thiết đúng ranh giới required_fact),
    fact đó vẫn 0 điểm dù trích dẫn đúng. Trên `pub-04-lam-viec-tu-xa` cụ
    thể, trần recall là 0.5 với MỌI harness đúng luật, vì đúng lý do đó —
    đo được, không phải suy đoán. Vẫn nên làm: `abstain: true` sau khi nêu
    cả hai phía được 0.5 recall + trọn 15 điểm honesty, và điểm recall lấy
    theo `max(...)` nên làm cả hai không bao giờ THIỆT — chỉ đừng trông
    đợi nó vượt sàn 0.5 trên brief này.
  * Xoá claim là hợp lệ. SỬA CHỮ trong `claim["text"]` thì KHÔNG: thêm
    một dấu chấm cuối câu cũng đủ làm claim mất cả provenance lẫn hỗ trợ
    (đo được: -40 điểm). Chỉ được xoá, giữ nguyên, hoặc cắt bớt.

GỢI Ý cho trường hợp (c): câu bị ghép là hai đoạn DO CHÍNH MÔ HÌNH viết,
dán với nhau bằng một liên từ (" và "). Cắt đúng chỗ dán thì hai nửa vẫn
là chữ của mô hình — vẫn qua được kiểm tra provenance. Muốn biết cắt đúng
chưa: cả hai nửa phải được scorer hỗ trợ trong tài liệu đã đọc và phải
thuộc HAI tài liệu khác nhau. Cắt sai thì một nửa sẽ vắt qua hai tài liệu
và không tài liệu nào chứa nó như một dòng.

CÔNG CỤ CÓ SẴN:
    ctx.observed_text  -> toàn bộ quan sát agent đã thấy, nối lại
    ctx.saw(text)      -> text có trong quan sát không
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
    ctx.state          -> dict tuỳ bạn dùng để ghi số liệu gỡ lỗi

Cài đặt:  ReActAgent(..., middleware=[InjectionGuard(), Critic(), ...])
Xem `harness/middleware.py` để biết thứ tự các hook.
"""

from __future__ import annotations

from harness.layers.citation_checker import MAX_CLAIM_CHARS, MIN_SUPPORT_CHARS, norm, source_of
from harness.middleware import Middleware


MAX_CLAIMS_PER_DOC = 4
MAX_SCORED_CLAIMS = 10


class Critic(Middleware):
    """Xoá những gì bằng chứng không đỡ; abstain khi không còn gì."""

    name = "critic"

    def after_agent(self, ctx, report):
        if not isinstance(report, dict):
            return {}
        claims = report.get("claims")
        if not isinstance(claims, list):
            return report

        def sources_for(text):
            if not isinstance(text, str) or not text:
                return []
            if ctx.corpus is not None:
                return source_of(text, ctx)
            claim_length = len(norm(text))
            return [""] if (
                ctx.saw(text)
                and MIN_SUPPORT_CHARS <= claim_length <= MAX_CLAIM_CHARS
            ) else []

        retained = []
        per_doc = {}

        def keep(claim):
            if len(retained) >= MAX_SCORED_CLAIMS:
                return
            doc_id = claim.get("doc_id")
            if isinstance(doc_id, str):
                count = per_doc.get(doc_id, 0)
                if count >= MAX_CLAIMS_PER_DOC:
                    return
                per_doc[doc_id] = count + 1
            retained.append(claim)

        for claim in claims:
            if not isinstance(claim, dict) or not isinstance(claim.get("text"), str):
                continue
            text = claim["text"]
            if not MIN_SUPPORT_CHARS <= len(norm(text)) <= MAX_CLAIM_CHARS:
                continue
            if sources_for(text):
                keep(claim)
                continue
            separator = " và "
            start = 0
            while True:
                split_at = text.find(separator, start)
                if split_at == -1:
                    break
                left = text[:split_at]
                right = text[split_at + len(separator):]
                split_sources = next(
                    ((left_doc, right_doc)
                     for left_doc in sources_for(left)
                     for right_doc in sources_for(right)
                     if left_doc != right_doc),
                    None,
                )
                if split_sources is not None:
                    left_doc, right_doc = split_sources
                    keep({**claim, "text": left, "doc_id": left_doc})
                    keep({**claim, "text": right, "doc_id": right_doc})
                    report["abstain"] = True
                    break
                start = split_at + len(separator)
        report["claims"] = retained
        if not retained:
            report["abstain"] = True
            report["answer"] = "Không đủ căn cứ trong tài liệu đã đọc để trả lời."
        report["citations"] = sorted(
            {claim["doc_id"] for claim in retained if isinstance(claim.get("doc_id"), str)}
        )
        return report
