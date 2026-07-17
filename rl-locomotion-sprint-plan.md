# RL Locomotion — Sim-to-Real Robustness Project

## 1. Mục tiêu dự án

Train một RL policy điều khiển robot 2 chân/quadruped đi/đứng thăng bằng trong MuJoCo bằng PPO, sau đó chứng minh **domain randomization giúp policy robust hơn** khi gặp physics params khác với lúc train — đây là proxy gần nhất cho sim-to-real gap mà không cần hardware thật.

**Câu chuyện kể được trong interview:** "Tôi không chỉ train một policy chạy được trong sim sạch — tôi đo được nó fail thế nào khi world thay đổi, và chứng minh randomization thu hẹp gap đó bao nhiêu %."

## 2. Tech Stack

| Thành phần | Công cụ |
|---|---|
| Environment | Gymnasium + MuJoCo (official `mujoco` bindings, không dùng mujoco-py đã deprecated) |
| RL algorithm | PPO qua Stable-Baselines3 |
| Cross-engine eval (bonus) | PyBullet |
| Logging/tracking | TensorBoard hoặc Weights & Biases |
| Visualization | Matplotlib/Seaborn |
| Video recording | Gymnasium `RecordVideo` wrapper + ffmpeg |

## 3. Success Metrics (Definition of Done)

- Policy đạt được locomotion ổn định (đi/đứng không ngã trong ≥80% episode) trên môi trường train
- Có ít nhất 2 policy để so sánh: **baseline** (không randomization) vs **DR** (có randomization)
- Robustness eval: DR policy phải cho success rate cao hơn rõ rệt baseline khi test trên physics params bị perturb
- Repo reproducible: `pip install -r requirements.txt` + 1 lệnh chạy training/eval

## 4. Sprint Breakdown

### Sprint 0 — Setup (2-3 ngày)
**Task:**
- Tạo venv, cài `gymnasium[mujoco]`, `stable-baselines3`, `pybullet`, `tensorboard`
- Verify MuJoCo chạy được (render thử 1 episode random policy)
- Setup git repo: `src/`, `configs/`, `experiments/`, `README.md`
- Setup logging (TensorBoard local là đủ, không cần W&B account nếu muốn đơn giản)

**Deliverable:** repo skeleton chạy được, video 1 episode random policy (baseline "trước khi train" để so sánh sau này)

---

### Sprint 1 — Baseline PPO Training (Tuần 1)
**Task:**
- Viết training script PPO trên `Ant-v4` trước (dễ converge hơn `Humanoid-v4` nhiều, nên dùng làm primary target — xem phần Rủi ro bên dưới)
- Dùng hyperparameters mặc định của SB3 làm baseline run đầu tiên
- Train tới khi reward curve plateau, log qua TensorBoard
- Eval: đo forward velocity trung bình, số bước sống sót trước khi ngã/lật

**Deliverable:** policy baseline đã train, training curve plot, video demo

---

### Sprint 2 — Reward Shaping & Hyperparameter Tuning (Tuần 2)
**Task:**
- Thử các biến thể reward: forward velocity + alive bonus + energy penalty (phạt action lớn) + orientation penalty (phạt nghiêng/lật)
- Ablation: train 3-4 config reward khác nhau, so sánh kết quả bằng bảng
- Tune PPO hyperparams quan trọng nhất: `learning_rate`, `n_steps`, `batch_size`, `gamma`, `gae_lambda`
- Chọn config tốt nhất làm "final baseline policy"

**Deliverable:** bảng ablation reward configs, policy baseline cuối cùng, phần README giải thích lý do chọn reward shaping

---

### Sprint 3 — Domain Randomization (đầu Tuần 3)
**Task:**
- Viết `gym.Wrapper` custom, random hóa mỗi lần reset episode:
  - Body mass (±15-20%)
  - Joint friction coefficient
  - Motor gear/strength (actuator force scale)
  - Action delay (giả lập độ trễ control loop thật — queue action N steps)
  - Observation noise (Gaussian noise lên joint position/velocity readings)
- Train policy thứ 2 với wrapper này bật lên (cùng reward config đã chọn ở Sprint 2, để so sánh công bằng)

**Deliverable:** code randomization wrapper (đây là phần code "chất" nhất để show trong interview), policy đã train với DR

---

### Sprint 4 — Robustness Evaluation (cuối Tuần 3 → đầu Tuần 4)
**Task:**
- Xây eval harness: chạy cả 2 policy (baseline vs DR) trên grid các mức perturbation (vd: friction ở 3 mức, mass ±10/20/30%, mỗi cell chạy N episode)
- Metric: success rate (% episode không ngã trong T steps), reward trung bình, quãng đường đi được
- Vẽ heatmap/bar chart so sánh 2 policy qua từng mức perturbation

**Deliverable:** eval script, heatmap so sánh robustness — đây là **kết quả chính** của cả dự án, chứng minh được luận điểm "randomization giúp generalize"

---

### Sprint 5 — Cross-Engine Sim-to-Real Proxy (Bonus, Tuần 4)
**Task:**
- Load policy đã train trong MuJoCo, eval zero-shot (không train lại) trên env tương đương trong PyBullet (`pybullet_envs` có sẵn Ant/Humanoid tương tự, hoặc dùng `pybullet-gym`)
- Đo performance drop khi chuyển engine — đây là proxy gần nhất cho "sim-to-real gap" mà không cần robot thật
- So sánh: policy DR có degrade ít hơn policy baseline khi chuyển engine không?

**Deliverable:** script cross-engine eval, số liệu so sánh degradation (nếu kết quả không như kỳ vọng cũng vẫn là finding đáng nói trong README — quan trọng là đo được, không phải phải "thắng")

**Lưu ý:** đây là phần rủi ro kỹ thuật cao nhất (xem mục Rủi ro) — nếu hết thời gian, có thể cắt bỏ mà vẫn có dự án hoàn chỉnh từ Sprint 1-4.

---

### Sprint 6 — Packaging & Documentation (2-3 ngày cuối)
**Task:**
- Quay video demo side-by-side: baseline vs DR policy dưới cùng 1 perturbation
- Viết README đầy đủ: problem statement → method → reward design → DR implementation → kết quả → limitations → "nếu có hardware thật thì bước tiếp theo là gì"
- Clean code, `requirements.txt`, hướng dẫn reproduce
- Chuẩn bị 3-4 câu tóm tắt để nói trong interview (vấn đề, cách giải, kết quả, số liệu cụ thể)

**Deliverable:** repo hoàn chỉnh, README, video demo — sẵn sàng đính kèm portfolio/CV

## 5. Rủi ro & Mitigation

| Rủi ro | Mitigation |
|---|---|
| `Humanoid-v4` rất khó converge trong thời gian/compute giới hạn | Dùng `Ant-v4` hoặc `Walker2d-v4` làm target chính, để Humanoid là stretch goal nếu còn thời gian |
| Training PPO tốn compute (hàng triệu steps) | Nếu máy cá nhân yếu, dùng Google Colab (free GPU) hoặc giảm `total_timesteps`, ưu tiên Ant-v4 (train nhanh hơn Humanoid nhiều) |
| Cross-engine port (Sprint 5) phức tạp do khác URDF/XML giữa MuJoCo và PyBullet | Coi đây là bonus, không phải core — cắt bỏ nếu cần mà không ảnh hưởng deliverable chính |
| Domain randomization implement sai (randomize quá mạnh khiến không train được gì) | Bắt đầu với randomization range nhỏ (±10%), tăng dần sau khi confirm policy vẫn học được |

## 6. Deliverables Checklist

- [ ] Repo GitHub public, README đầy đủ
- [ ] Baseline policy + training curves
- [ ] DR policy + randomization wrapper code
- [ ] Robustness comparison plot (kết quả chính)
- [ ] Video demo side-by-side
- [ ] (Bonus) Cross-engine eval kết quả

## 7. Ước lượng timeline tổng

**~4 tuần full-time** (Sprint 0-4 là bắt buộc = ~3 tuần; Sprint 5 bonus + Sprint 6 packaging = ~1 tuần). Nếu làm part-time song song với việc khác, tính khoảng **6-7 tuần**. Phần tốn thời gian nhất thực tế không phải viết code mà là **chờ training chạy** (Ant-v4 PPO train tới hội tụ thường mất vài giờ đến nửa ngày mỗi run) — nên plan song song: trong lúc chờ 1 run train, chuẩn bị code cho sprint tiếp theo.
