# Distillation v1 với Plan A và Plan B

Hai phương án dùng cùng model, pool, dev và ICLDIST, output riêng:

| Plan | Chọn KNOWDIST | Fine-tune mặc định |
|---|---|---|
| R | Random từ pool hợp lệ | Uniform response-only CE |
| A | Giữ độ phủ token giữa nhóm, xếp hạng IFD-style trong nhóm | Cùng LoRA và CE |
| B | Giữ độ phủ token giữa nhóm, random trong nhóm, không cần score | Cùng LoRA và CE |

A dùng tỷ số loss phản hồi của base student với/không có prompt, không phải
quy trình IFD gốc có học trước. IFD không chứng nhận nhãn đúng hoặc “siêu mẫu”.
Khoảng [0,1] là control ban đầu, cần audit/dev; mất nhóm thì A dừng, không đổi
âm thầm thành B. B không phụ thuộc scoring, nên chuẩn bị B/R trước.

## Kaggle

### Clone và chạy bằng lệnh

Sau khi clone source vào `/kaggle/working/sentiment-v1-ab-pilot`, không cần upload
notebook standalone. Bật GPU + Internet rồi chạy:

```python
!python /kaggle/working/sentiment-v1-ab-pilot/tools/run_kaggle_pilot.py --plan B --steps 50 --eval-samples 64 --mapping-confirmed --work /kaggle/working/v1-pilot-01
```

Flag mapping xác nhận sử dụng schema/extractor v1 đã mô tả dưới đây; loader
không tự sửa đáp án và sẽ dừng khi tỷ lệ loại vượt 30%. Cần đọc exclusions/preview
khi đánh giá tính đại diện của pool. Đây là smoke pilot, không cấu hình paper cuối.

Lệnh tự tải corpus v1 và khóa revision, normalize/tokenize, tạo B/R, đánh giá
trước train trên 64 mẫu val được chọn bằng seed, train 50 bước/stage và đánh giá
adapter sau ICLDIST. Demos lấy từ train. Lấy F1/Acc validation và forward timing
batch1 sau warm-up cùng peak allocated/reserved; không dùng ASQP chưa audit.
Raw-prompt/first-label-token scorer vẫn cần audit cho báo cáo cuối.

Output: `report-B.json`, `report-B.md`, `environment.json`, `logs/`, `state/`,
prepared datasets và full checkpoints trong `--work`. Download/load/IO nằm ngoài
forward latency; train wall có phạm vi riêng. Không ghép số pilot val với test cũ.

Rerun nguyên lệnh để reuse các bước có receipt hash và resume train. Nếu mất
phiên, restore toàn bộ work vào cùng đường dẫn working cũ trước chạy. Đổi source,
model, seed, steps hoặc eval sample count thì dùng work mới; không tự ghi đè.
Chuyển `--plan A`/`--plan R` giữ cùng work để dùng pool/dev/ICLDIST chung, output
riêng. `--score-seconds` giới hạn scoring A; B dùng được khi A thiếu điểm/lỗi.
`--include-paper-reference` bổ sung checkpoint paper đã distillation trên cùng val
subset và phép đo; không tải teacher. Thư viện thiếu thì dừng, không thay torch/CUDA.

### Notebook standalone tùy chọn

Upload `notebooks/Sentiment_Distillation_V1_AB_Kaggle.ipynb`, bật Internet và GPU
khi cần. Notebook nhúng source/config, không cần clone repo hoặc token GitHub.
Không cài/thay torch/CUDA tự động. Kiểm tra packages/runtime trước khi chạy.
Mặc định mapping chưa xác nhận, score A và train tắt; B/R có thể chuẩn bị trước.
Chỉ xác nhận sau khi xem raw records, preview và exclusions.

Pipeline: download v1 pinned SHA → normalize → tokenize pool → chọn B/R →
score/chọn A → pilot → train KNOWDIST → nạp adapter trainable để ICLDIST.
Chọn ACTIVE_PLAN=A/B/R, RUN_TRAIN=True để chạy. Pilot 50 bước/stage là smoke,
không phải kết quả chính; PILOT_STEPS=0 dùng token budget trong configs/v1_ab.json.
Trần mẫu 3500/1500 và token 1.5M/0.5M là pilot defaults, không bảo đảm chọn đủ 5k.
Giữ token budget giống nhau khi so sánh; runner mô phỏng stream thực để dừng
tại optimizer boundary, overshoot tối đa một effective batch. Source/config/
data hash được lưu; steps không nhất thiết giống nhau khi độ dài khác nhau.

Mỗi invocation mặc định 8h (tối đa 10h), giữ khoảng dư so với giới hạn 12h người
dùng cung cấp. Deadline kiểm tra sau optimizer step, không cưỡng bức kill trong
download/probe/save; cần theo dõi phiên thực tế. Save Version full outputs.
Resume giữ nguyên đường dẫn working, config/source/packages/GPU và checkpoint
đầy đủ; phục hồi từ input read-only vào working cũ. Mỗi plan/seed có thư mục riêng.
Khi resume ICLDIST phải giữ cả adapter KNOWDIST vì đó là ancestry được hash.

## CLI tương đương

```bash
python training/prepare_v1_pruned.py normalize --input /kaggle/input/raw-v1 --revision ACTUAL_SHA --output /kaggle/working/exp/normalized --mapping-confirmed --benchmark-root datasets
python training/prepare_v1_pruned.py tokenize --normalized /kaggle/working/exp/normalized --output /kaggle/working/exp/pool --model Qwen/Qwen2.5-0.5B-Instruct
python training/prepare_v1_pruned.py select --pool /kaggle/working/exp/pool --method coverage --output /kaggle/working/exp/prepared/B
python training/prepare_v1_pruned.py select --pool /kaggle/working/exp/pool --method random --output /kaggle/working/exp/prepared/R
python training/prepare_v1_pruned.py score --pool /kaggle/working/exp/pool --output /kaggle/working/exp/scores-A --max-seconds 7200
python training/prepare_v1_pruned.py select --pool /kaggle/working/exp/pool --scores /kaggle/working/exp/scores-A --method student --output /kaggle/working/exp/prepared/A
python training/run_v1_ab.py --prepared /kaggle/working/exp/prepared/B --output /kaggle/working/exp/runs/B/seed-42 --plan B --pilot-steps 50 --dry-run
```

Bỏ --dry-run để thực thi trên Kaggle. Bỏ --pilot-steps để chạy ngân sách chính
đã khóa sau pilot. Chuyển B sang A cần đổi cả --prepared, --output và --plan.
Không đưa F1 test vào quyết định chuyển nhánh. Khi đã xem test, thay phương pháp
phải công bố là thay đổi hậu kiểm; không gọi đó là kết quả kiểm chứng độc lập.

Nếu phiên thực tế có hai GPU và đủ RAM, có thể chạy B và scoring A đồng thời
bằng hai tiến trình riêng (thư mục output khác nhau). Ví dụ trên Kaggle:

```bash
CUDA_VISIBLE_DEVICES=0 python training/run_v1_ab.py --prepared /kaggle/working/exp/prepared/B --output /kaggle/working/exp/runs/B/seed-42 --plan B --pilot-steps 50
CUDA_VISIBLE_DEVICES=1 python training/prepare_v1_pruned.py score --pool /kaggle/working/exp/pool --output /kaggle/working/exp/scores-A --max-seconds 7200
```

Khởi chạy trong hai terminal/cell tiến trình độc lập; không cần hai GPU để dùng
pipeline. Hai lệnh trên không tự chạy song song trong một cell tuần tự.

## Dữ liệu và giới hạn

Loader stream JSON arrays/JSONL. KNOWDIST dùng prompt/llm_response và instr_type
(hoặc tên file analysis_/rewrite_). ICLDIST dùng instruction/input/output;
phân biệt TD/sentiment theo tên file. Giữ nguyên đáp án và nội dung trường.
Source dùng source_text nếu có, input riêng không rỗng, hoặc marker cuối đã hỗ
trợ. Template lạ bị loại có lý do, không tự hash cả prompt rồi nhận split an toàn.
Preview full corpus phải xác nhận tỷ lệ loại và độ phủ trước tokenization.

Split source key toàn cục giữa hai stage trước chọn mẫu. Exact prompt-response
dedup và optional exact-normalized final-input overlap với test; chưa audit near
duplicates hoặc văn bản nằm trong demonstrations. Không tuyên bố contamination-free.
Candidate reservoir theo hash và nhóm; dev tối đa 8 mẫu/nhóm/split/stage, train
có giới hạn tổng. Pool được cap trước loại overlength, không bù âm thầm; thiếu
nhóm phải tăng pool hoặc đổi context rồi tạo experiment mới.

Train/probe/holdout có thể thiếu nhóm trong corpus nhỏ; fail nếu split stage rỗng.
R/C/A dùng cùng dev kể cả nhóm không được chọn; uniform hoạt động được, các mode
adaptive có thể cần lọc/khóa group mapping riêng (chưa bật trong kế hoạch mặc định).
Tokenization phải làm lại nếu đổi student; không gọi tập lọc theo Qwen là tối ưu
riêng cho SmolLM2. Scoring ghi model revision, pool hash và thời gian, có resume.

## Chuyển phương án

- A không đủ score/bị lỗi/mất nhóm/vượt ngân sách scoring: giữ log, chạy B đã chuẩn bị.
- A/B chạy đều được: so cùng probe/dev, budget và tổng score+train cost; giữ holdout
  độc lập, báo cả nhánh thất bại/kết quả âm. Không tự đổi winner dựa trên test.
- Giữ cùng baseline R. Nếu thêm progress weighting, tạo config/ablation mới;
  không đồng thời đổi selector và fine-tuning rồi quy hết tác dụng cho selector.

Evaluator ASQP/latency/adapter-cache vẫn cần audit trước so sánh benchmark cuối.
Không ghép điểm từ notebook cũ với checkpoint mới sau khi đổi giao thức.

## Kiểm tra local không chạy model

```bash
python3 -B -m unittest discover -s tests -v
python3 tools/build_v1_ab_notebook.py
```

Các kiểm tra dùng fixtures tổng hợp và dữ liệu v1 mẫu; không cài dependency,
download, train hay inference model local. GPU/throughput/F1 chỉ xác nhận được
sau pilot thực trên Kaggle.
