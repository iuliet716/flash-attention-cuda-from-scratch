# Step 8. WMMA TensorCore (한글 초안)

> 최종 영어 문서 작성용 참고 초안.

## 1. 개념

### 텐서코어란

SM 안에 있는 행렬 곱 전용 하드웨어 유닛. 일반 CUDA 코어가 곱-합 1개씩 처리하는
동안, 텐서코어는 **16×16×16 행렬 곱-누적(D = A·B + C) 전체**를 몇 사이클에
처리한다. FP16 입력 × FP16 입력 + FP32 누적이 기본 조합이다 — step 07에서 fp16
저장을 준비한 이유가 이것이다.

CUDA에서의 인터페이스가 **wmma API**다:

- 연산 단위가 스레드가 아니라 **워프**(32스레드가 협동)
- 데이터는 `fragment`라는 타입에 담긴다 — 16×16 타일을 워프의 32스레드가
  나눠 갖는 조각으로, **어떤 스레드가 어떤 원소를 갖는지는 비공개**
- `load_matrix_sync`(smem→fragment) → `mma_sync`(곱-누적) →
  `store_matrix_sync`(fragment→smem)

### 커널 구조 재설계: 워프당 1행 → 워프당 16×16 타일

지금까지의 "워프당 Q행 1개 + 스칼라 내적" 구조는 텐서코어와 맞지 않는다.
연산의 단위가 행렬 타일이 되도록 재설계한다: BR=16(Q행), BC=64(KV행), 4워프.

- **QK^T**: S(16×64)를 4개의 16×16 블록으로 나눠 워프당 1블록.
  d 방향으로 16씩 잘라 `mma_sync`로 누적
- **softmax**: S를 smem에 내려놓고 워프 0가 행별로 online softmax (알고리즘은 그대로)
- **PV**: O(16×d)의 컬럼을 4워프가 d/4씩 분담

### 이 단계의 세금: O 누산기가 SRAM에 있어야 하는 이유 (중요)

online softmax는 타일마다 **행별로** 누산기를 `alpha = exp(m_old - m_new)`배 해야
한다. 그런데 wmma fragment는 "내 레지스터가 몇 행 원소인지"를 알려주지 않는다
(아키텍처마다 배치가 다르고 의도적으로 봉인됨). 행을 모르면 행별 rescale을
레지스터에서 할 수 없다.

우회책: **O 누산기를 shared memory에 둔다.** 타일마다

```
스칼라 코드로 O_smem 행별 rescale
→ load_matrix_sync(O fragment ← O_smem)
→ mma_sync(P·V 누적)
→ store_matrix_sync(O_smem ← fragment)
```

라운드트립을 돈다. 이 왕복이 이 단계의 구조적 세금이고, step 10에서
"레이아웃이 공개된 mma.sync"로 갈아타며 회수한다.

### 부속 장치들

- **SKEW(행당 +16 halves 패딩)**: wmma는 포인터 32B 정렬을 요구하고, 행마다
  시작 뱅크를 어긋나게 하는 효과도 있다(step 06 패딩 계열의 wmma 버전).
- **K^T 만들기**: row-major로 저장된 K를 `col_major`로 load하면 그게 곧 전치다.
  B(k,n) = K[n][k]가 되도록 포인터와 leading dimension(ldm)만 맞추면 된다.
- **smem 48KB 초과**: 동적 smem이 48KB를 넘으면
  `cudaFuncSetAttribute(..., MaxDynamicSharedMemorySize, ...)`로 명시적 opt-in.

### 결과와 의미

62.9 → 41.6ms (1.51x). 이 지점에서 수제 단일 커널이 **PyTorch matmul+softmax
(cuBLAS 기반 eager 파이프라인)와 동급**이 된다. 다만 텐서코어 활용률은 아직
한 자릿수 % — S/P/O가 전부 smem을 왕복하느라 텐서코어가 대부분 굶는다.
그 관찰이 step 09~10의 동기다.

## 2. 코드 구현 설명

### smem 카빙 (fp32 먼저, fp16 뒤)

```cuda
extern __shared__ __align__(16) unsigned char smem_raw[];
float* Osm = ...;  // O 누산기 [BR][d]
float* Ssm = ...;  // 이번 타일의 raw 점수 [BR][BC]
float* m_sm, l_sm, a_sm;  // 행별 softmax 상태
__half* Qs, Ks, Vs;       // [.. ][d+SKEW] 타일
__half* Ps;               // softmax 결과(half) [BR][BC]
```

정렬 문제를 피하려고 fp32 배열을 앞에, fp16을 뒤에 배치한다.

### QK^T — 워프당 16×16 블록

```cuda
wmma::fragment<wmma::accumulator, 16, 16, 16, float> s_frag;
wmma::fill_fragment(s_frag, 0.0f);
for (int k = 0; k < d; k += 16) {
    wmma::fragment<wmma::matrix_a, ..., wmma::row_major> a_frag;  // Q
    wmma::fragment<wmma::matrix_b, ..., wmma::col_major> b_frag;  // K → K^T
    wmma::load_matrix_sync(a_frag, Qs + k, ldh);
    wmma::load_matrix_sync(b_frag, Ks + (warp * 16) * ldh + k, ldh);
    wmma::mma_sync(s_frag, a_frag, b_frag, s_frag);
}
wmma::store_matrix_sync(Ssm + warp * 16, s_frag, BC, wmma::mem_row_major);
```

`ldm`(마지막 인자)은 "행과 행 사이 간격"으로, SKEW가 포함된 `d + 16`이다.

### softmax — 워프 0의 스칼라 코드

lane r이 S의 r행 전체(BC개)를 순회하며 online softmax 갱신. `scale`은 이때 곱한다.
alpha를 `a_sm[]`에 적어 다른 워프들이 rescale에 쓰게 한다.

### 경계 처리에서 배운 것

Q의 범위 밖 행을 **0으로 채우는** 것이 중요하다. 쓰레기 값이면 S가 쓰레기가 되고
`exp(쓰레기)`가 inf/NaN을 만든다. 0이면 S=0 → softmax가 유한하게 돌고, 결과는
마지막에 행 가드로 버린다. (wmma는 항상 16×16 전체를 계산하므로 "계산은 하되
무해하게"가 원칙)

### PV — rescale 후 누적

```cuda
// 각 워프: 자기 컬럼 슬라이스를 스칼라로 rescale
Osm[r * d + c] *= a_sm[r];
__syncwarp();
// 그 위에 텐서코어로 P·V 누적
o_frag ← load(Osm) → mma_sync(p_frag, v_frag) → store(Osm)
```

### 배리어

타일마다 `__syncthreads()` 4개: 로드 후 / S 저장 후 / softmax 후 / PV 후.
워프 간에 smem으로 데이터가 오가는 지점마다 하나씩 필요하다.
이 배리어들 역시 step 10에서 대부분 사라진다.

## 3. 벤치마크 참고

- **필수**: 메인 표 (07→08, ~1.5x). "PyTorch eager와 동급 도달"을 강조.
- **강력 추천**: ncu의 텐서 파이프 활용률
  (`sm__pipe_tensor_op_hmma_cycles_active...` 계열). "텐서코어를 도입했는데
  활용률은 몇 %"라는 수치가 step 09~10의 동기 부여를 완성한다.
