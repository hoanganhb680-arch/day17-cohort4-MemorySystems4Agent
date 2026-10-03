# Day 17 — Memory Systems (bản hoàn thiện)

Chạy từ thư mục gốc với Python 3.11+:

```powershell
python src/benchmark.py
python -m pytest src/test_agents.py -v
```

Benchmark và test chạy **offline**, không cần API key; benchmark tự dùng thư mục trạng thái tạm để kết quả không phụ thuộc lần chạy trước. Để dùng agent trực tiếp, `load_config()` tạo `state/`; `AdvancedAgent` lưu hồ sơ từng người ở `state/profiles/<user_id>/User.md`. Dùng `force_offline=True` nếu muốn chắc chắn không gọi API. Nếu có API key của provider được chọn và đã cài integration LangChain tương ứng, agent gọi model thật; `ollama` được thử khi chọn provider đó. Chế độ live dùng cùng memory layer, nhưng các con số dưới đây chỉ đo offline.

`.env` ở gốc repo hoặc biến môi trường có thể đặt `LLM_PROVIDER`, `LLM_MODEL`, `LLM_TEMPERATURE`, `JUDGE_PROVIDER`, `JUDGE_MODEL`, `COMPACT_THRESHOLD_TOKENS`, `COMPACT_KEEP_MESSAGES`, `PROFILE_CONFIDENCE_THRESHOLD` (0–1, mặc định 0.8); các provider hỗ trợ: `openai` (`OPENAI_API_KEY`), `custom` (`CUSTOM_API_KEY`, `CUSTOM_BASE_URL`), `gemini` (`GEMINI_API_KEY`), `anthropic` (`ANTHROPIC_API_KEY`), `ollama` (`OLLAMA_BASE_URL`), `openrouter` (`OPENROUTER_API_KEY`, tùy chọn `OPENROUTER_BASE_URL`). Judge chỉ có cấu hình dự phòng: benchmark offline không gọi judge.

## Luồng memory

- Baseline giữ toàn bộ user/assistant messages **chỉ trong thread**; thread mới không đọc hồ sơ.
- Advanced trích các facts khai báo rõ ở ngôi thứ nhất vào `User.md`; cập nhật cùng field khi có đính chính, bỏ qua đề cập nghề/nơi ở kiểu câu đùa hoặc câu hỏi. `User.md` có kích thước hữu hạn theo số field thay vì append vô hạn.
- Mỗi thread Advanced giữ messages gần nhất và summary tối đa 600 ký tự. Khi ngữ cảnh vượt ngưỡng, messages cũ được rút ngắn, đếm một compaction. Hồ sơ vẫn dùng được ở thread mới.
- Hai agent dùng cùng hàm trả lời offline đơn giản, để so memory thay vì so hai model khác nhau.

## Bonus: confidence threshold và conflict handling

Extractor gán độ tin cậy heuristic 1.0 cho phát biểu trực tiếp, 0.4 cho câu có dấu hiệu giả định (`nếu`, `giả sử`, `ước gì`, v.v.) và bỏ qua câu hỏi. Advanced chỉ ghi facts đạt ngưỡng cấu hình; khi người dùng khẳng định một giá trị mới, cùng field trong `User.md` được thay thế thay vì giữ hai giá trị mâu thuẫn. Test kiểm tra giả định “ở Huế” không ghi đè nơi ở “Đà Nẵng”, nhưng lời đính chính trực tiếp thì có.

Bonus này giảm nguy cơ lưu sai fact và cải thiện recall sau correction mà không tăng prompt token đáng kể. Đổi lại, quy tắc từ khóa có thể bỏ sót phát biểu thật chứa “nếu”, hoặc chấp nhận câu nói mơ hồ không có từ khóa; điểm confidence **không được hiệu chuẩn như xác suất**. Giảm ngưỡng sẽ tăng recall tiềm năng nhưng tăng nguy cơ ghi sai; dùng xác nhận người dùng hoặc extractor tốt hơn nếu triển khai thực tế.

## Đọc kết quả

Trên dataset mẫu, Standard: baseline 0% recall và advanced 100%; advanced xử lý nhiều prompt tokens hơn (~17k so với ~12k) do chi phí hồ sơ trong những hội thoại ngắn, chưa compact. Stress: baseline ~22k prompt tokens, advanced ~12k với 2 compactions và 100% recall. Đây là phép đo heuristic `len(text)/4`, không phải token billing thực; chạy lệnh benchmark để xem con số chính xác sau thay đổi. `Agent tokens only` là độ dài đầu ra ước lượng, không bao gồm input. `Response quality` offline là proxy dựa trên số expected facts trong câu trả lời, **không phải** đánh giá độc lập bằng LLM judge.

Compact giảm chi phí **prompt tokens processed** do không phải gửi lại toàn bộ lịch sử từng lượt; nó không tự động giảm số token đầu ra. Trade-off: summary có thể mất chi tiết, hồ sơ có thể lưu sai nếu phát biểu mơ hồ, regex hiện chỉ xử lý các facts tiếng Việt xác định trong dataset. Cần bổ sung xác nhận/consent và cơ chế xóa dữ liệu người dùng trước khi dùng production.
