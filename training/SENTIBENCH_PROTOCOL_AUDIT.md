# Yêu cầu cho coding agent: kiểm chứng giao thức SentiBench v1

Trạng thái 2026-11-22: đã đọc arXiv:2503.03225v2 và source official tại commit
`2810fc2826e8f7c2ae6a0d6dd328ab7d8979744a`. Tên **arXiv v2** là revision của
paper Targeted Distillation v1, không phải phương pháp/corpus CompEffDist v2.
Chỉ dùng corpus `zhang-yice/sentiment-distillation-v1` trong nghiên cứu hiện tại.

## Những điều phải đính chính

- Table2 Qwen2.5-1.5B+Know&ICLDist: ASQP22.93, SST295.99,
  Emotion2072.57, Avg1264.01 (% F1). Qwen base ASQP18.47.
  Không lấy ASQP0 của một rerun làm số paper hoặc chứng cứ model paper bất lực.
- Chưa tái hiện/audit raw output của lượt0 để xác định nguyên nhân. Các lỗi
  prompt/order/extraction/parser là ứng viên, không được viết “chắc chắn do X”.
- Section4.1: full training corpus1M K+400K I+100K general tasks cho models
  1.2B/1.5B; model3.2B dùng subset200K K+100K I. Kho v1 reduced đang chạy và
  token budget2M của nhóm phải tách khỏi số sample toàn paper.
- AppendixC.2:671 A100 GPU-hours là **teacher data generation**, không phải
  student fine-tuning wall hoặc baseline để tính tỷ lệ tiết kiệm train nhóm.
- Limitations nêu2 hạn chế (teacher response quality và tuple extraction);
  reasoning/domain combination là hướng triển vọng ởD.2. Không tự suy ra chọn
  Qwen3/no_think đã kiểm chứng đóng góp reasoning-enhanced.
- MPCB/RPR/STFA/top5% outlier/decaying replay/BCTD chưa phải thuật toán đang
  chạy trong repo này. Pilot A/B là selection student-proxy/coverage trên
  Qwen2.5-0.5B. Không đổi mô tả thành Qwen3 hoặc coi tên mới là novelty đã chứng minh.

## Mục tiêu giao thức ghi trong paper

Toàn bộ12 official test files/11.974 lượt mẫu; khóa SHA256 theo
`configs/sentibench_v1_test_manifest.json`, không dùng MAX_SAMPLES hay data_prop
để cắt test. BSA/MSA macroF1; ATSA/ACSA/ASQP microF1; Opener sentiment graphF1.
Avg12 chỉ khi đủ12 dataset cho mọi seed. Accuracy là chỉ số bổ sung nếu thích hợp.

Section4.1:4-shot randomly sampled **Dev** demos, greedy (paper mô tả temperature0),
trung bình3 evaluation seeds. Script bash tác giả dùng42/52/62; đây là seed nguồn
code, paper không ghi cụ thể ba giá trị. BSA/MSA dùng first-generated-token
label scores trong code; không tự đổi thành string generation rồi gọi y nguyên.
Trong Transformers, dùng do_sample=False,num_beams=1; không buộc truyền literal
temperature=0 nếu phiên bản thư viện không nhận hoặc bỏ qua khi không sampling.
Đó không phải đảm bảo bitwise reproducibility giữa mọi GPU/software.

Prompt gốc BSA/MSA/FSA ở Tables13–15 và source phải được đối chiếu, lưu raw
rendered prompts và demo IDs cho từng test row/seed/model. ASQP source gold/demo
category-first trong khi câu chỉ dẫn ghi aspect-first; không sửa ngầm. Chọn
protocol rõ ràng cho mỗi run và apply cùng cho base/adapter/paper checkpoint.

## Mâu thuẫn paper/source hiện đã quan sát

1. Official `fsa.py:load_dataset` và `icl_multifaced.py:load_dataset` đọc
   `train.json` cho demo, trái mô tả Dev ở Section4.1.
2. Official ATSA/ACSA chỉ có train2000 và test676, không có val/dev400; Table1
   ghi train1600/dev400. Không có bằng chứng400 dòng cuối/train split seed nào
   chính là dev, nên không tự chia/đặt tên để tuyên bố khôi phục Dev paper.
3. Dataset khác có val300; ASQP val316, Opener val249. SST2 val có1 input trùng
   test theo audit trước. Lọc train/val/demo-side, giữ nguyên test. Pool Dev đã
   lọc là clean protocol variant, không còn identical original demo pool.
4. `tools/run_kaggle_evaluation.py` hiện default train4-shot/seed42 và parser
   sửa r1. Đây là corrected re-evaluation, không phải strict reproduction.
   `--seeds 42 52 62` có sẵn nhưng không tự khắc phục thiếu Dev hoặc prompt mismatch.

## Công việc cần thực hiện, giữ nguyên kết quả lịch sử

1. Lưu protocol manifest theo từng run, tách `paper_text_target`,
   `released_code_reference`, `clean_corrected_r1`. Không gắn nhãn
   paper-equivalent nếu thiếu Dev provenance/prompt/scorer parity.
2. Audit expected/actual train/dev/test counts và file hashes; tạo báo cáo lỗi
   thiếu Dev, cấm fallback train hoặc tự resplit trong strict mode. Có thể tìm
   split preparation artifacts của tác giả/read-only; không gửi liên hệ tác giả
   nếu chưa có chỉ thị. Nếu không tìm thấy, công bố gap, không bịa matching100%.
3. Đối chiếu fsa.py/parse_utils bản gốc với wrapper hiện tại. Lưu generated
   token tail, raw text, extracted text, parsed tuples, valid-format flag,
   prediction tuple count và metric; giữ parser gốc để chấm diagnostic cùng
   raw outputs với parser sửa. Phân biệt đổi prompt (cần inference mới) với
   đổi parser (có thể rescore cùng raw outputs), không ghép F1 hai protocol.
4. Test parser bằng gold literals, multiline/list/tuple, >3 tuples, thiếu ngoặc,
   NULL và thứ tự ASQP; test metric oracle/sanity riêng, nhất là SSA span graph.
   Gold literal round-trip2.395 dòng đã pass không chứng minh end-to-end FSA đúng.
5. Kiểm tra tokenizer truncation, padding và decoding only-generated-tail;
   không lấy regex label đầu trong demos làm model prediction. Không đoán
   nguyên nhân0 trước kiểm tra raw artifacts. Không thêm constrained decoding
   chỉ riêng nhánh cải tiến để so với baseline không ràng buộc.
6. Final reporting: ba evaluation seeds42/52/62, mean+std per dataset và Avg12;
   một seed chỉ preliminary. Seed chọn demos khác với training seed. Tất cả
   model dùng cùng row/demo IDs, format, parser và generation budget trong run.
7. Test không chọn A/B/threshold/replay ratio/checkpoint. Clean dev tuning dùng
   final demo pool theo manifest; không lấy chính row đang chấm làm demo của
   nó khi đánh giá validation (leave-one-source-out hoặc pool độc lập).
8. Deadline/resume từng batch trong12h Kaggle; nếu thiếu giờ, báo incomplete
   và resume phiên sau, không bớt test hoặc giảm3seeds rồi gọi final average.

Không đổi student chính sang Qwen3 chỉ dựa vào lời dẫn trên. Tính khả thi hiện
có evidence Qwen2.5-0.5B; thay model cần config/run riêng và pilot mới.
Phương án cải tiến đang đề xuất là ICL stratification+K replay đơn giản;
sample-normalized decaying replay hay STFA là biến thể cần định nghĩa/ablate riêng.

## Nguồn

- https://arxiv.org/html/2503.03225v2 (Tables1/2/4–6/13–15, Sections4.1/C.2/D.2).
- https://github.com/HITSZ-HLT/Sentiment-Distillation/tree/2810fc2826e8f7c2ae6a0d6dd328ab7d8979744a
- Các source snapshot `git show origin/main:fsa.py`,
  `git show origin/main:icl_multifaced.py`, `git ls-tree -r origin/main datasets/absa`.

Mốc này chỉ là source/data audit và yêu cầu triển khai. Chưa chạy ASQP model
mới, chưa khôi phục đủ Dev và chưa chứng nhận strict paper reproduction.
