# Kế hoạch nghiên cứu: distillation sentiment cho student dưới 1B bằng corpus đã công bố

Ngày lập: 2026-11-01. Trạng thái: **kế hoạch đề xuất, chưa triển khai pipeline huấn luyện và chưa có kết quả**.

## 1. Mục tiêu và giới hạn

Mục tiêu là tạo một student nhỏ, ưu tiên `Qwen/Qwen3-0.6B`, bằng dữ liệu teacher có sẵn; kiểm chứng chất lượng và chi phí trên cùng giao thức với model công bố của tác giả. Không đặt trước kết luận student sẽ thắng model lớn hơn.

Các điều kiện bắt buộc:

- Không gọi teacher để sinh thêm instruction, demonstration, rationale, nhãn hay đáp án.
- Không viết lại nội dung teacher bằng LLM khác; không tự sửa nhãn theo dự đoán student.
- Được phép chọn tập con, loại bản ghi không hợp lệ/trùng, đổi thứ tự lấy mẫu, tái trọng số loss và đóng gói dữ liệu vào template model.
- Giữ bản gốc bất biến. Mọi phép lọc lưu bằng danh sách ID và lý do; không ghi đè corpus nguồn.
- Đánh giá bằng cùng test, metric, evaluator và giao thức được khóa trước. Không chọn dataset hoặc seed sau khi xem kết quả test.
- Không chạy cài dependency, huấn luyện hay inference trên máy hiện tại. Thực nghiệm chạy trên Kaggle/Colab theo yêu cầu của người dùng.
- Chưa chứng nhận tính mới. Việc đổi model, dùng LoRA hoặc vượt vài điểm riêng lẻ không tự tạo ra đóng góp phương pháp.

## 2. Nguồn và mức độ tái lập

### 2.1 Những thông tin đã xác minh

| Nguồn | Phạm vi công bố | Ý nghĩa với nghiên cứu này |
|---|---|---|
| Corpus v1 | Thư mục `KNOWDIST_200k`, `ICLDIST_100k` | Không mặc định là bản đầy đủ của mọi thí nghiệm v1 |
| Paper v1 | Student nhỏ dùng tới 1,5 triệu mẫu; student khoảng 3B dùng 300k | Dùng bản công bố không đồng nghĩa tái lập đúng ngân sách 1,5 triệu mẫu |
| Corpus v2 / README | Ba file 50k theo teacher Llama, Qwen, Gemma | Chọn riêng file Qwen làm nguồn chính |
| Paper v2 | Teacher–student theo từng dòng; đánh giá SentiBench | Không gộp ba file rồi gọi là thiết lập gốc |
| Phụ lục C v2 | 4-shot, 3 seed; prompt FSA đã chỉnh | Không trộn điểm FSA từ hai giao thức |

Nguồn trực tiếp:

1. [Paper v1 — Targeted Distillation](https://arxiv.org/html/2503.03225v2), mục 4.1.
2. [Corpus v1](https://huggingface.co/datasets/zhang-yice/sentiment-distillation-v1/tree/main).
3. [Paper v2 — CompEffDist](https://arxiv.org/html/2510.24425v1), mục 5.1 và phụ lục C.
4. [Corpus v2](https://huggingface.co/datasets/zhang-yice/sentiment-distillation-v2).
5. [README CompEffDist](https://github.com/HITSZ-HLT/COMPEFFDIST#released-training-corpus).

README CompEffDist còn liệt kê `post_train/`, cấu hình huấn luyện và `eval_sentibench/`. Đây là nguồn cần kiểm tra trước khi bổ sung trainer. Repo v1 hiện tại chủ yếu là mã đánh giá; không áp dụng kết luận “thiếu trainer” của v1 cho toàn bộ repo v2.

### 2.2 Phạm vi chính được chọn

- Paper tham chiếu: **CompEffDist / v2**.
- Nguồn huấn luyện: `zhang-yice/sentiment-distillation-v2`, file `Qwen3-32B_50k.jsonl`.
- Không mặc định trộn v1 và v2, không trộn ba nguồn teacher trong thí nghiệm chính.
- Đếm số bản ghi thực tế sau tải; tên `50k` không thay thế kiểm toán dữ liệu.
- Bản công bố đã qua quy trình của tác giả. Chọn tiếp tập con chỉ được gọi là **chọn lại trong corpus đã phát hành**, không phải tái tạo bước lọc từ pool 100k ban đầu.
- Một file 50k có thể không đủ để tái lập mọi ablation của paper. Không khẳng định có đầy đủ pool trước lọc hoặc mọi intermediate artifact.

### 2.3 Ba mức tuyên bố phải tách riêng

| Mức | Có thể tuyên bố khi nào? |
|---|---|
| Tái lập đánh giá checkpoint | Tải đúng revision checkpoint và chạy đúng giao thức đã khóa |
| Tái triển khai huấn luyện | Dùng corpus công bố, công bố cấu hình và các khác biệt với tác giả |
| Tái lập chính xác kết quả huấn luyện | Chỉ khi kiểm chứng đủ dữ liệu, preprocessing, trainer, precision, hyperparameter và nguồn ngẫu nhiên |

Qwen3 0.6B + LoRA là **model mới trong thí nghiệm của mình**; không gọi nó là bản tái lập chính xác Qwen3 4B của tác giả.

## 3. Câu hỏi nghiên cứu và giả thuyết

### 3.1 Câu hỏi chính

> Với cùng student, cùng nguồn dữ liệu teacher và cùng ngân sách token, chọn mẫu có xét độ phủ nhóm tác vụ có tốt hơn lấy mẫu ngẫu nhiên về khả năng sentiment tổng quát không?

### 3.2 Câu hỏi phụ

1. Student 0.6B sau fine-tuning nằm ở đâu trên đường đánh đổi F1–VRAM–độ trễ so với checkpoint công bố?
2. Cải tiến có giúp nhóm MSA/FSA hay chỉ các tác vụ polarity dễ hơn?
3. Kết quả có ổn định qua seed, hay do một tập con/demo thuận lợi?
4. Chi phí chọn dữ liệu có làm mất lợi ích tiết kiệm huấn luyện không?

Các giả thuyết này có thể sai. Kết quả âm hoặc không khác baseline phải được giữ và giải thích.

### 3.3 Giới hạn tính mới

Chọn dữ liệu, difficulty filtering và LoRA đều có tiền lệ. Trước khi chốt tên phương pháp, lập bảng related work tối thiểu cho:

- [CompEffDist](https://aclanthology.org/2025.emnlp-main.1122/).
- [LESS](https://proceedings.mlr.press/v235/xia24c.html).
- [A Critical Look at Targeted Instruction Selection](https://proceedings.mlr.press/v306/nayak26a.html).
- Các nghiên cứu student dưới 1B, multi-task sampling, token-budget selection và sentiment distillation gần thời điểm nộp bài.

Với mỗi bài, ghi corpus, student, thuật toán chọn, ngân sách, baseline và phần thực sự khác đề xuất. Nếu khác biệt chỉ là tên model hoặc benchmark, chuyển định vị sang nghiên cứu thực nghiệm có kiểm soát; không nhận là thuật toán mới.

## 4. Kiểm toán và đóng băng dữ liệu

### 4.1 Tải và lưu bản gốc

1. Lấy revision SHA cụ thể của dataset; không dùng `main` thay đổi theo thời gian cho bản cuối.
2. Chỉ tải file teacher Qwen và metadata cần thiết. Chưa tải model teacher 32B.
3. Lưu file nguyên bản, revision, kích thước, SHA-256, số dòng và số JSON record đọc được.
4. Gán ID ổn định: `dataset_revision + relative_file + line_number`; thêm hash bản ghi để phát hiện thay đổi.
5. Giữ các file Llama/Gemma ngoài thí nghiệm chính. Nếu bổ sung sau, tạo experiment ID khác.

Có thể tải corpus bằng `huggingface_hub` trên Kaggle/Colab. Không chạy lệnh tải trong máy hiện tại. Lỗi viewer do tên cột teacher khác nhau không có nghĩa file JSONL hỏng; đọc riêng từng file.

### 4.2 Kiểm tra schema trước khi ghép prompt

Preview hiện cho thấy các trường như `text`, `instruction`, `input`, `task_name`, `cluster_id`, `selected_cluster_id`, `ppl`, `ifd`, `avg_rank_score`, `text_length`, và cột output mang tên teacher. Không giả định mọi bản ghi đều có đủ hoặc cùng ý nghĩa.

Kiểm tra tối thiểu 30–50 bản ghi, trải trên các nhóm/task và độ dài khác nhau. Bắt buộc giải quyết:

- `text` có phải input người dùng thực tế không? Nếu `input` rỗng, nội dung test đã nằm trong `instruction` hay phải lấy từ `text`?
- Instruction đã chứa demonstration chưa? Không nối thêm hoặc nhân đôi chúng khi đóng gói.
- Cột output nào là phản hồi cần học? Với file Qwen, kiểm tra trực tiếp cột `Qwen/Qwen3-32B_output`.
- Task nào có output tự do, JSON, danh sách, nhãn hoặc số? Không buộc tất cả đáp án phải là JSON.
- Các điểm `ppl/ifd/avg_rank_score` được tính theo model/giao thức nào? Không diễn giải chúng thành confidence teacher hoặc độ khó của Qwen3 0.6B khi chưa xác minh.

Đối chiếu preprocessing trong repo CompEffDist với một số mẫu đầu vào–đầu ra. **Không huấn luyện nếu còn mơ hồ cách lắp instruction + input + text.**

### 4.3 Quy tắc lọc chung, áp dụng cho mọi nhánh

- Loại record không đọc được, thiếu input thiết yếu hoặc phản hồi rỗng; ghi rõ từng lý do.
- Không loại phản hồi chỉ vì student dự đoán khác.
- Không viết lại, dịch, paraphrase hoặc rút gọn phản hồi teacher.
- Phát hiện trùng bằng hash prompt–response; nếu các dòng cùng input có task khác nhau thì không tự động xem là trùng.
- Kiểm tra trùng nội dung giữa corpus và benchmark bằng tiêu chí văn bản đã đăng ký. Không dùng nhãn test hoặc điểm test để chọn mẫu.
- Nếu phát hiện contamination và cần loại, áp dụng cho **tất cả nhánh tự huấn luyện**, giữ nguyên benchmark và công bố mức thay đổi so với corpus tác giả. Không thể khẳng định checkpoint tác giả đã được sửa contamination tương ứng.
- Tính token bằng tokenizer và chat template thật của student, bao gồm special tokens và EOS.
- Với mẫu vượt ngưỡng, ưu tiên loại khỏi pool chung theo quy tắc trước đăng ký. Không cắt âm thầm phần đáp án rồi vẫn coi nhãn nguyên vẹn.
- Ghi phân bố task/cluster/độ dài trước và sau lọc. Nếu lọc làm rơi một nhóm, phải báo cáo.

Pool sau xử lý gọi là `eligible_pool`; không gọi là “đủ 50k” nếu đã loại record.

### 4.4 Tách dữ liệu phát triển

- Đề xuất holdout khoảng 5% `eligible_pool` làm distillation-validation; chốt tỷ lệ sau audit, trước huấn luyện.
- Group split theo hash văn bản nguồn/input để cùng một nội dung không xuất hiện ở cả hai phía qua các task khác nhau.
- Train pool và validation dùng chung giữa tất cả phương pháp.
- Validation này phục vụ loss/checkpoint; không đồng nhất với validation SentiBench.
- SentiBench-dev dùng điều chỉnh hyperparameter/đánh giá phát triển theo giao thức đã định; SentiBench-test không dùng ra quyết định.
- Việc giữ riêng holdout là một khác biệt so với huấn luyện của tác giả nếu họ dùng toàn file; ghi rõ trong báo cáo.

## 5. Khóa giao thức đánh giá trước thí nghiệm chính

### 5.1 Chọn một evaluator làm chuẩn

1. Kiểm tra `eval_sentibench/` của CompEffDist, khóa Git commit và mọi local patch.
2. Đối chiếu prompt FSA theo v2; không mặc định các prompt trong repo v1 hiện tại tương đương.
3. Khóa dữ liệu test, thứ tự ID, parser, verbalizer và cách tổng hợp điểm.
4. Kiểm tra chấm nhãn token đầu/toàn nhãn. Giao thức chính phải cố định và áp dụng cho mọi model; thay đổi scorer là thí nghiệm phụ riêng.
5. Nếu evaluator có lỗi cần sửa, mô tả lỗi và chạy lại **mọi checkpoint so sánh**; không ghép điểm cũ và điểm sau sửa.

Repo hiện tại đã có sửa kỹ thuật như input cap, resume và giải mã completion. Chúng hữu ích để chạy notebook nhưng chưa được xác nhận tái lập evaluator v2. Không dùng notebook hiện tại làm chuẩn paper khi chưa audit.

### 5.2 Danh sách đánh giá đầy đủ

| Nhóm | Dataset key trong repo hiện tại | Metric cần đối chiếu và khóa |
|---|---|---|
| BSA | `sc/imdb` | Macro-F1 |
| BSA | `sc/yelp2` | Macro-F1 |
| BSA | `sc/sst2` | Macro-F1 |
| BSA | `sc/twitter` | Macro-F1 |
| MSA | `multifaced/irony18` | Macro-F1 |
| MSA | `multifaced/tweeteval` | Macro-F1 |
| MSA | `multifaced/pstance` | Macro-F1 |
| MSA | `multifaced/intimacy` | Macro-F1 |
| FSA | `absa/atsa_rest16` | F1 trích xuất theo evaluator |
| FSA | `absa/acsa_rest16` | F1 trích xuất theo evaluator |
| FSA | `absa/asqp_rest16` | F1 tuple theo evaluator |
| FSA | `absa/opener` | F1 cấu trúc sentiment theo evaluator |

Tên key là ánh xạ của repo hiện tại, cần đối chiếu với repo v2. Không tự thay graph/tuple metric thành Accuracy hoặc F1 phân loại thông thường.

### 5.3 Quy tắc công bằng bắt buộc

- Cùng test file/hash và sample IDs cho tất cả model. Đánh giá cuối dùng toàn bộ test, không phải 5/32 mẫu.
- Chốt 4 demonstrations và ba seed đánh giá. Có thể dùng 42/52/62 cho nghiên cứu này; kiểm tra seed thực tế của evaluator nếu tuyên bố tái lập chính xác.
- Tạo và lưu mapping `dataset + eval_seed + test_id -> demo_ids`; dùng chung mapping giữa các model.
- Demo lấy từ validation theo giao thức v2. Một số tập trong repo v1 hiện tại thiếu `val.json`: phải giải quyết bằng nguồn/split chính thức hoặc công bố giao thức khác; **không tự đổi sang train mà vẫn nhận là cùng điều kiện paper**.
- Demo không lấy từ test. Đầu ra teacher của corpus huấn luyện không thay nhãn chuẩn benchmark.
- Cùng nội dung instruction, demo, thứ tự và câu hỏi. Dùng wrapper/template đúng model khi bắt buộc, nhưng lưu template và nêu rõ khác biệt. Không gọi đó là đầu vào byte-identical nếu wrapper khác.
- Với Qwen3, khóa non-thinking trong thí nghiệm hiệu quả; kiểm tra prefix/template thực tế để nhãn được chấm ở đúng vị trí trả lời. Không để first-token scorer chấm token reasoning.
- Khóa deterministic decoding cho tác vụ sinh có cấu trúc, giới hạn sinh, padding, EOS và quy tắc parse. Nếu chọn giao thức khác evaluator tác giả, đánh dấu riêng và chấm lại mọi baseline.
- Không dùng input cap làm mất nội dung với một model mà không báo cáo. Kiểm toán token hóa trước; nếu không thể giữ cùng nội dung vì context window khác, tạo thí nghiệm giới hạn ngữ cảnh riêng.
- Không thay model/template/thinking/precision chỉ cho một phía sau khi xem test.
- Đầu ra parse lỗi tính theo scorer đã khóa; không bỏ mẫu lỗi khỏi mẫu số.
- `Avg-12` chỉ tính khi đủ 12 dataset; thiếu tập phải hiện N/A/partial, không điền 0 hoặc đổi mẫu số mà vẫn ghi SentiBench average.

### 5.4 So sánh hai mức

- **Mức chính: kiểm chứng phương pháp.** Cùng Qwen3 0.6B, cùng cách fine-tune, cùng pool, cùng ngân sách và cùng evaluator; chỉ khác quy tắc chọn mẫu.
- **Mức tham chiếu: chất lượng–chi phí.** Model 0.6B của mình so với checkpoint tác giả lớn hơn. Không coi đây là bằng chứng duy nhất phương pháp tốt hơn vì model nền và lịch sử huấn luyện khác nhau.

Nếu cần đối chiếu bảng paper, trình bày hai cột: `paper-reported` và `re-evaluated`. Chỉ dùng cột chạy lại cùng giao thức để tính chênh lệch công bằng trong nghiên cứu này.

## 6. Ma trận model và baseline

### 6.1 Bắt buộc ở nghiên cứu chính

| ID | Model / nhánh | Vai trò |
|---|---|---|
| A | `Qwen/Qwen3-0.6B`, chưa fine-tune thêm | Baseline model nền |
| B | Qwen3 0.6B + LoRA, random subset | Baseline trực tiếp của đề xuất |
| C | Qwen3 0.6B + LoRA, đề xuất chọn mẫu | Model nghiên cứu |
| D | Qwen3 0.6B + LoRA, toàn train pool hợp lệ | Mốc dùng nhiều dữ liệu; không phải baseline cùng ngân sách |
| E | `zhang-yice/Qwen3-4B-sentiment-distillation-v2` | Checkpoint tác giả tương ứng dòng Qwen |

Nếu không đủ tài nguyên chạy D, có thể giới hạn bài ở hiệu quả cùng ngân sách B–C và công bố thiếu mốc full-pool. Không khẳng định đạt hiệu quả của full-data khi chưa đo.

### 6.2 Tham chiếu mở rộng

- `zhang-yice/llama-3-3B-sentiment-distillation-v2`.
- `zhang-yice/gemma-3-4b-sentiment-distillation-v2`.
- Qwen3 4B nguyên bản để phân biệt năng lực nền với lợi ích distillation.
- Model v1 như Qwen2.5 1.5B chỉ đưa vào bảng bổ sung, chạy lại cùng evaluator và ghi rõ là checkpoint từ công trình khác.

Mục tiêu cuối là chấm lại các checkpoint v2 đã liệt kê nếu tài nguyên cho phép. Nếu chỉ chạy E, giới hạn kết luận ở checkpoint Qwen; không nhận là thắng mọi model của paper.

Kiểm tra model ID, revision, tokenizer, generation config và quyền truy cập trước khi tải. Không tự thay bằng model gần tên.

## 7. Phương pháp đề xuất tối thiểu

Tên làm việc: **chọn mẫu cân bằng nhóm dưới ngân sách token**. Đây là mô tả thực nghiệm, chưa nhận là thuật toán mới.

### 7.1 Đầu vào

- Train pool cố định từ file teacher Qwen v2.
- Token count `n_i` cho toàn chuỗi huấn luyện của mỗi mẫu, tính bằng tokenizer/template student.
- Nhóm `g_i` lấy từ metadata task/cluster đã audit. Chốt một ánh xạ duy nhất; tránh tạo hàng nghìn nhóm quá nhỏ mà không kiểm tra phân bố.
- Ngân sách token tập con `B_select`, seed selection và thứ tự tie-break cố định.

Không dùng điểm test, nhãn benchmark hoặc câu trả lời teacher mới để tính `g_i`/`n_i`.

### 7.2 Thuật toán phiên bản đầu

1. Dùng cùng train pool sau lọc cho mọi selector.
2. Nhóm record bằng ánh xạ task/cluster đã chốt; xáo trộn trong từng nhóm bằng seed.
3. Theo vòng, chọn nhóm còn dữ liệu có tổng token đã chọn thấp nhất so với quota của nhóm.
4. Lấy ứng viên tiếp theo nếu toàn mẫu còn vừa ngân sách; không cắt đáp án để ép vừa.
5. Dừng khi không còn mẫu vừa budget. Lưu token thực tế, phần budget chưa dùng và coverage.

Quota có thể cân bằng đều theo nhóm hoặc theo phân bố gốc; lựa chọn này phải chốt bằng dev và trở thành ablation, không thay sau test. Nếu thuật toán này không khác công trình đã có, giữ nó như baseline có ý nghĩa, không đổi tên để nhận tính mới.

### 7.3 Đối chứng tối thiểu

| Selector | Kiểm soát điều gì? |
|---|---|
| Random | Baseline bắt buộc, cùng ngân sách token |
| Shortest-first | Kiểm tra lợi ích có chỉ do chọn mẫu ngắn |
| Balanced-by-group | Kiểm tra lợi ích từ coverage |
| Proposed budget-aware selector | Chỉ thêm nếu thực sự có thành phần khác ba baseline trên |

Nếu đề xuất chính là Balanced-by-group dưới budget, không lập hai dòng Balanced và Proposed có cùng thuật toán. Dùng ba dòng Random / Shortest / Balanced, định vị như nghiên cứu so sánh có kiểm soát.

Điểm khó sẵn có trong v2 chỉ là phương án phụ sau khi biết nguồn gốc. Không chọn lại theo chính điểm difficulty của tác giả rồi nhận là phương pháp độc lập mới. Corpus đã lọc không cho phép kết luận về những mẫu đã bị tác giả bỏ trước phát hành.

## 8. Thiết kế huấn luyện

### 8.1 Cấu hình chung B–C

- Cùng revision Qwen3 0.6B, tokenizer và template.
- Supervised fine-tuning trên phản hồi teacher có sẵn; chỉ tính loss trên vùng response, mask prompt/padding đúng.
- Cùng LoRA target modules, rank, alpha, dropout và initialization cho các nhánh trong một seed.
- Khóa precision theo GPU. Chạy FP16/BF16 nếu đủ bộ nhớ; quantization là nhánh riêng, không đổi giữa các phương pháp âm thầm.
- Giữ nguyên response teacher; không ép thành nhãn ngắn nếu dữ liệu gốc yêu cầu giải thích/cấu trúc.
- Nếu dùng packing, phải kiểm tra mask, EOS và biên mẫu. Nếu chưa kiểm chứng, dùng dynamic padding; ghi chi phí padding.

Không sao chép hyperparameter của proxy model trong paper thành cấu hình student; đó là hai mục đích khác nhau.

### 8.2 Pilot và chốt cấu hình

Đề xuất phạm vi nhỏ để thử trên train/dev, không phải cấu hình đã chứng minh tối ưu:

- LoRA rank 8 hoặc 16; chỉ chọn một biến thể cuối bằng dev.
- Learning rate thử tối đa hai mức phù hợp trainer, ví dụ `5e-5` và `1e-4`; ghi rõ là lựa chọn nghiên cứu, không phải cấu hình gốc paper.
- Microbatch 1, tăng gradient accumulation theo VRAM; không đổi effective batch giữa B–C khi không có lý do đã đăng ký.
- Max sequence length ban đầu 2048, nhưng phải audit số mẫu vượt ngưỡng và toàn bộ input/response.
- Một budget nhỏ cho pilot; sau đó chốt 1–2 budget chính theo tốc độ thực đo. Không hứa thời gian GPU trước pilot.

Đồng nhất số lượt qua dữ liệu và lịch checkpoint B–C. Nếu một biến thể được tuning nhiều hơn, phải công bố và dành baseline cơ hội tuning tương đương.

### 8.3 Định nghĩa ngân sách công bằng

Ghi riêng:

- Số record và số nhóm.
- Token input, token response, tổng token không padding của tập con.
- Tổng token thực sự đi qua model trong toàn run, bao gồm lặp epoch.
- Token có loss, token padding, số optimizer update và effective batch.
- Thời gian chọn mẫu, huấn luyện, GPU peak memory và hardware.

Chốt dung sai ngân sách B–C trước khi chạy, đề xuất tối đa 1% chênh tổng token. Nếu không đạt do mẫu nguyên vẹn, báo chênh và điều chỉnh tập con trước huấn luyện, không cắt response.

**Cùng số token không đồng nghĩa cùng FLOPs/thời gian tuyệt đối.** Độ dài chuỗi, padding và số update vẫn ảnh hưởng; báo cáo các số đo đó thay vì gọi mọi thứ là “cùng compute”. Mốc D dùng nhiều dữ liệu phải tách khỏi bảng matched-budget.

## 9. Quy trình triển khai theo giai đoạn

| Giai đoạn | Việc phải làm | Đầu ra / điều kiện chuyển tiếp |
|---|---|---|
| G0 — Khóa phạm vi | Chốt paper v2, teacher Qwen, student, budget dự kiến và novelty audit | Bản đăng ký thí nghiệm; không có tuyên bố thắng trước |
| G1 — Audit corpus | Revision/hash, schema, lắp prompt, token count, lọc và split | Manifest; mapping input/response không còn mơ hồ |
| G2 — Audit evaluator | Repo v2, test/demo mapping, FSA prompts, metrics và model templates | Một protocol ID cố định; giải quyết các dev split thiếu |
| G3 — Smoke trên cloud | Vài mẫu train/dev, save/reload adapter, scorer sanity và resume | Pipeline chạy được; điểm smoke không vào bảng chính |
| G4 — Baseline đánh giá | A và E trên dev, pilot đo tốc độ/VRAM | Ước lượng chi phí, phát hiện lỗi prompt/scorer |
| G5 — Pilot huấn luyện | B và C cùng budget nhỏ, một seed, chỉ xem dev | Chốt cấu hình; quyết định tiếp tục hoặc dừng |
| G6 — Thực nghiệm chính | B/C các seed, D và tham chiếu theo phạm vi đã đăng ký | Checkpoint, log và manifest đầy đủ |
| G7 — Test khóa | Chạy mọi checkpoint theo cùng evaluator, toàn test | Predictions và metrics trên đủ phạm vi |
| G8 — Phân tích/báo cáo | CI, ablation, chi phí, lỗi và giới hạn | Bảng kết quả tái lập được; không bỏ run bất lợi |

Không viết toàn pipeline trước G1/G2. Không chạy mọi model và mọi selector ngay từ đầu.

### Điểm quyết định G5

Chỉ chuyển sang G6 khi:

- Không có lỗi schema/template/mask/scorer đã biết.
- Đã lưu được checkpoint và thực sự tiếp tục được run trong phiên cloud khác.
- Có tốc độ thực đo để ước lượng ngân sách còn lại.
- Cải tiến có tín hiệu trên dev hoặc lợi ích chi phí đáng kiểm tra; chốt tiêu chí trước pilot.

Ví dụ tiêu chí quyết định nghiên cứu: tăng ít nhất 0,5 điểm F1 trên dev mà không vượt budget, hoặc giữ F1 trong biên 0,5 điểm với chi phí thấp hơn. Đây là ngưỡng **đề xuất để quyết định đầu tư**, không phải tiêu chuẩn thống kê hay kết quả đã đạt. Điều chỉnh/chốt trước pilot, không sau khi xem test.

Nếu không có tín hiệu, dừng nhánh phương pháp. Không liên tục thay selector rồi dùng cùng test để tìm một kết quả thắng. Có thể báo cáo nghiên cứu đối chiếu/negative result hoặc quay lại related work.

## 10. Seed, thống kê và cách kết luận

- Phân biệt `selection_seed`, `training_seed`, `evaluation_demo_seed`; lưu cả ba.
- Pilot có thể một seed. Bản kết quả chính nên có ít nhất ba seed huấn luyện cho B/C nếu nhận kết luận ổn định.
- Checkpoint tác giả cố định, chạy cùng ba demo seed. Không xem ba demo seed như ba lần huấn luyện độc lập.
- Dùng seed ghép cặp B–C và cùng demo mapping để giảm nhiễu so sánh.
- Báo mean/std theo đơn vị seed được định nghĩa, kèm kết quả từng run. Không coi 3 training seed × 3 demo seed là 9 model độc lập.
- Có thể tính paired bootstrap bằng predictions cùng sample IDs cho từng dataset; tính lại F1 trong mỗi lần bootstrap. Nêu số bootstrap, đơn vị lấy mẫu và nguồn bất định mà CI phản ánh.
- Chỉ tính CI của Avg-12 theo thủ tục đã đăng ký; không lấy std của 12 task rồi gọi là CI do seed.
- Nếu so nhiều phương pháp/metric, ghi rõ metric chính và phân tích phụ; không chọn p-value nhỏ nhất để kết luận.
- Không gọi “không khác biệt có ý nghĩa” là bằng chứng tương đương. Muốn nhận tương đương/non-inferiority phải đăng ký biên và kiểm định phù hợp.

Metric chính đề xuất: Avg-12 theo protocol đã khóa, cùng bảng đủ 12 tác vụ. Báo BSA/MSA/FSA và từng dataset để nhìn thấy đánh đổi. Accuracy bổ sung ở tác vụ phân loại, không thay F1 chính.

## 11. Đo hiệu năng triển khai

- Cùng GPU, phiên bản thư viện, batch và tập input khi so latency/VRAM.
- Cùng precision cho bảng chính nếu khả thi. Nếu model lớn cần quantization, ghi bảng khác; không gộp như cùng điều kiện số học.
- Loại thời gian tải model khỏi latency suy luận nhưng báo riêng loading/cold start nếu cần.
- Warmup cố định; đồng bộ CUDA trước/sau đo; đo nhiều batch và báo median/p95, throughput.
- Tính cả tokenization/retrieval/postprocess nếu báo end-to-end; cũng có thể báo model-only nhưng phải ghi tên đúng.
- Reset và ghi peak allocated/reserved VRAM; nêu số GPU và có offload hay không.
- Báo token input/output; model nhỏ sinh dài hơn có thể chậm hơn model lớn sinh ngắn hơn.
- Nếu cache/reuse prompt được bật, áp dụng chung hoặc tách thí nghiệm; không so một phía cache nóng với một phía cache lạnh.
- Không dùng tên số tham số để suy ra tốc độ; phải đo.

## 12. Artifacts và cấu trúc triển khai dự kiến

Các đường dẫn dưới đây là **đề xuất cho phần sẽ viết**, không phải script đã tồn tại:

```text
research/
  protocols/protocol-v2.yaml
  configs/{baseline,proposed,full-pool}.yaml
  manifests/{source,eligible,split,selected,demos}.jsonl
  scripts/{audit,select,train,evaluate,summarize}.py
  reports/{data-audit,related-work,evaluator-audit}.md
artifacts/<experiment_id>/
  config.json
  source-revisions.json
  environment.json
  selection-manifest.jsonl
  training-log.jsonl
  checkpoint/
  predictions/<dataset>/<eval_seed>.jsonl
  metrics.json
  timing.json
  failures.jsonl
```

Không commit dataset thô, model weights hoặc token vào Git. Manifest dùng ID/hash để có thể dựng lại lựa chọn từ corpus gốc. Nếu công bố dữ liệu dẫn xuất, kiểm tra điều kiện sử dụng/ghi nguồn tương ứng.

Tối thiểu mỗi prediction phải có: model/checkpoint revision, protocol ID, dataset hash, test ID, demo IDs, raw output hoặc scores, parsed prediction, trạng thái parse và seed. Không lưu chỉ một bảng F1.

## 13. Môi trường Kaggle/Colab và tính liên tục

- Mặc định dùng package nền tảng như người dùng yêu cầu. Không cài đè toàn bộ requirements cũ.
- Ghi Python, torch/CUDA, transformers, peft/trainer, tokenizer, numpy, hardware và lock/freeze hiệu lực.
- Nếu thiếu gói cần cho huấn luyện, lập cài đặt tối thiểu riêng cho cloud; không nâng toàn bộ hệ thống để chữa một lỗi.
- Qwen3 cần thư viện hỗ trợ kiến trúc và non-thinking template; kiểm tra thực tế trước run dài.
- Lưu optimizer, scheduler, scaler nếu có, RNG, sampler position, step/token counter và adapter khi checkpoint; chỉ lưu adapter là chưa đủ để tiếp tục huấn luyện đúng.
- Kiểm tra resume trên train/dev trước G6. Nếu không khôi phục đầy đủ trạng thái, ghi là run mới thay vì ghép log như một run liên tục.
- Dữ liệu trên `/tmp`, Kaggle VM hoặc Colab VM có thể mất. Xuất artifacts theo phiên; restore từ persistent output/Drive ở phiên sau.
- Không dùng thủ thuật né quota hoặc giả định luôn có một loại GPU. Nếu buộc thay hardware, ghi vào bảng chi phí và tránh so thời gian trực tiếp.

Notebook hiện có chỉ phục vụ đánh giá; cần thêm notebook/trainer nghiên cứu sau G1/G2. Đoạn lệnh chạy 5 mẫu SST2 không phải pipeline huấn luyện và không phải kết quả đủ để viết bảng chính.

## 14. Bảng kết quả dự kiến

### 14.1 Bảng chất lượng

| Model/run | Corpus | Train tokens | BSA | MSA | FSA | Avg-12 | Seed variability |
|---|---|---:|---:|---:|---:|---:|---|
| Qwen3 0.6B nguyên bản | — | 0 | TBD | TBD | TBD | TBD | TBD |
| Random subset + LoRA | Qwen v2 | TBD | TBD | TBD | TBD | TBD | TBD |
| Shortest-first + LoRA | Qwen v2 | TBD | TBD | TBD | TBD | TBD | TBD |
| Balanced subset + LoRA | Qwen v2 | TBD | TBD | TBD | TBD | TBD | TBD |
| Full eligible pool + LoRA | Qwen v2 | TBD | TBD | TBD | TBD | TBD | TBD |
| Checkpoint Qwen3 4B tác giả | Công bố | Không ước đoán | TBD | TBD | TBD | TBD | Demo seeds |

Tất cả ô kết quả hiện là TBD. Cần bảng bổ sung đủ 12 dataset, từng seed và run thất bại; không chỉ ba cột nhóm.

### 14.2 Bảng chi phí

| Run | Selection time | Training time | Effective tokens | Peak VRAM | Median/p95 inference | Hardware/precision |
|---|---:|---:|---:|---:|---|---|
| Các nhánh nghiên cứu | TBD | TBD | TBD | TBD | TBD | TBD |

### 14.3 Ablation tối thiểu

- Random so với Balanced ở cùng token budget.
- Shortest-first để kiểm tra tác động độ dài.
- Nếu có thành phần mới: bỏ đúng thành phần đó, giữ phần còn lại.
- Một budget thứ hai nếu tài nguyên cho phép; không bắt đầu bằng ma trận quá lớn.
- Không gộp thay scorer/chat template với thay selector trong cùng so sánh rồi quy hết lợi ích cho selector.

## 15. Checklist kết thúc nghiên cứu

- [ ] Đã giải thích rõ v1/v2, file teacher và phạm vi dùng thực tế.
- [ ] Không gọi teacher hoặc tạo đáp án mới.
- [ ] Corpus/test/model/evaluator đều có revision và hash.
- [ ] Mapping prompt/input/response đã kiểm tra thủ công.
- [ ] Train/dev không trùng nội dung theo tiêu chí đã công bố.
- [ ] Không dùng test để chọn mẫu, tuning hay chọn run tốt nhất.
- [ ] Demo mapping, prompt và scorer được khóa chung.
- [ ] Đã giải quyết dev split còn thiếu; không tráo train/dev âm thầm.
- [ ] Baseline Random/Shortest/Balanced hoặc đối chứng tương đương đã chạy.
- [ ] Chi phí token và tuning được kiểm soát, phần không khớp đã công bố.
- [ ] Checkpoint tác giả đã được chấm lại cùng protocol trong phạm vi kết luận.
- [ ] Báo đủ dataset, seed, lỗi parse và kết quả bất lợi.
- [ ] Đã tách cải thiện do student/model nền khỏi cải thiện phương pháp.
- [ ] Không nhận tính mới, tương đương hoặc vượt toàn paper khi chưa có bằng chứng.
- [ ] Artifacts đủ để người khác dựng lại tập con và tính lại điểm.

## 16. Các quyết định chưa được phép tự giả định

1. Revision chính xác của corpus/model/evaluator cuối cùng.
2. Mapping input–response đúng trong file Qwen v2.
3. Nguồn validation/demonstration đúng cho mọi dataset.
4. Khả năng tương thích và khác biệt evaluator v1–v2 sau audit.
5. Tổng token hợp lệ, GPU thực tế và thời gian còn lại.
6. Ngân sách cuối cùng, metric chính, ngưỡng quyết định và số seed đủ khả thi.
7. Thành phần phương pháp thực sự khác related work.

Đầu ra gần nhất cần thực hiện là **data audit + evaluator audit**, sau đó mới viết trainer và chọn tên đề tài cuối. Kế hoạch này không phải bằng chứng đã tái lập được paper hoặc đã tạo model tốt hơn.
