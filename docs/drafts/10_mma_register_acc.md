# Step 10. Register-Resident Accumulators — PTX mma.sync (한글 초안)

> 최종 영어 문서 작성용 참고 초안. 이 리포의 클라이맥스 단계:
> 25.3ms → 3.52ms (7.2x), SDPA FlashAttention 대비 70%대 도달.

## 1. 개념

### wmma의 남은 세금 계산서

step 09까지의 커널이 타일 반복마다 내던 shared memory 왕복 (d=64 기준):

| 항목 | 왜 필요한가 | 반복당 비용 |
|---|---|---:|
| S 저장 + 읽기 | softmax가 smem에서 스칼라로 돌므로 | ~16 KB |
| P 저장 + 읽기 | softmax 결과를 다시 fragment로 올리므로 | ~8 KB |
| O rescale + load/store | fragment의 행을 몰라 레지스터 rescale 불가 | ~64 KB |
| `__syncthreads` | 워프 간 smem 인계 지점마다 | 4회 |

반복 128번이면 블록당 ~11MB의 smem 트래픽. 정작 유용한 K/V 데이터는 반복당
8KB다. 커널이 절반쯤 "SRAM 복사 기계"였던 셈이고, 이 모든 세금의 뿌리는 하나 —
**wmma fragment의 레지스터 배치가 비공개**라는 것.

### mma.sync: 같은 하드웨어, 공개된 배치도

PTX 명령 `mma.sync.aligned.m16n8k16`은 wmma와 **같은 텐서코어**를 쓰지만,
레지스터↔행렬 매핑이 PTX ISA 문서에 명시돼 있다:

```
groupID = lane / 4  (0..7)      tig = lane % 4  (0..3)

C/D (16x8, fp32 4개):
  c0, c1 → (groupID행,     2*tig열, 2*tig+1열)
  c2, c3 → (groupID+8행,   같은 열)

     열:  0  1  2  3  4  5  6  7
행 0:    L0 L0 L1 L1 L2 L2 L3 L3     (L = lane, 각 2원소)
행 1:    L4 L4 L5 L5 L6 L6 L7 L7
...
행 8:    L0 L0 ...  (c2, c3로)
```

즉 **모든 스레드가 자기 누산기 레지스터의 (행, 열)을 안다.**

### 배치도가 열어주는 세 가지 자유

1. **softmax가 레지스터에서**: 한 행이 quad(같은 lane/4의 스레드 4개)에 걸쳐
   있으므로, 행 max/sum은 스레드 내 2원소 처리 + XOR 셔플 2번(offset 1, 2)이면
   끝. smem도, 워프 0 독점도, 배리어도 필요 없다.
2. **C→A 재사용 (FA2의 핵심 트릭)**: 누산기(C) 배치가 A 오퍼랜드 배치와
   호환된다. softmax를 마친 P를 레지스터에서 half2로 재포장하면 그대로 PV의
   A 입력이 된다 — P가 smem에 내려갈 일이 없다.
3. **O·Q도 레지스터 상주**: O는 행별 rescale이 레지스터 곱셈이 되고, Q는
   커널 시작 때 HBM에서 fragment 원소를 직접 한 번 읽어 끝까지 보유.
   결국 **smem에는 K/V 더블 버퍼만 남는다** (BC=64에서 40~72KB) → SM당 블록
   여러 개 복귀, 배리어는 4→2개.

### 템플릿 특수화가 필수인 이유

레지스터 "배열"(`float o[OB][4]` 등)은 모든 인덱스가 컴파일 타임에 풀려야
레지스터에 남는다. 크기가 런타임 값(d)에 의존하면 local memory(사실상 HBM)로
스필되어 전부 물거품이 된다. 그래서 `template<int D>`로 d=64/128을 각각
컴파일한다.

### 진단 서사: 1.00x → 6.7x (이 문서의 하이라이트)

- **BC=32 첫 구현: step 09와 23.67ms 정확한 동률.** 구조가 전혀 다른 두 커널의
  소수점 동률은 우연이 아니라 **공통 게이트의 지문**이다.
- 게이트: 반복당 고정 오버헤드(배리어 + cp.async 대기 + smem 지연) × 128회.
  임계 경로가 게이트로 결정되는 동안에는, 게이트 뒤에서 SRAM 왕복을 아무리
  없애도 총 시간이 안 줄어든다. 개선은 사라진 게 아니라 **가려져** 있었다.
- **BC=64**: 반복 수(=게이트 횟수) 절반 → 지연 지배에서 처리량 지배로 체제
  전환 → 숨어 있던 개선이 한꺼번에 현금화되어 3.52ms.
- 최종: 156 TFLOPS ≈ fp16 피크의 ~74%, SDPA 대비 70~79% (N 스윕).

교훈: **기법의 측정된 효과는 기법 자체가 아니라 현재 병목과의 관계가 결정한다.**
(step 03·07·09에 이어 네 번째 반복되는 주제 — 리포의 관통 교훈으로 묶을 것)

### 이 커널의 정체

FlashAttention-2 forward의 설계 그 자체다: split-Q(워프당 행 소유), 레지스터
상주 누산기, cp.async 파이프라인. 참고로 RTX 5090(sm_120)에는 데이터센터 전용
명령(wgmma, tcgen05)이 없어 fp16 기준 mma.sync가 최상위 명령이다 — 즉 SDPA의
커널과 **같은 명령 세트**로 겨뤄서 70%대가 나온 것이다.

## 2. 코드 구현 설명

### 인라인 PTX 프리미티브

```cuda
__device__ __forceinline__ void mma_16816(
    float* acc, const uint32_t* a, const uint32_t* b)
{
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
        : "+f"(acc[0]), "+f"(acc[1]), "+f"(acc[2]), "+f"(acc[3])
        : "r"(a[0]), ..., "r"(b[1]));
}
```

- D(출력)와 C(누적 입력)에 같은 레지스터를 써서 in-place 누적 — `"+f"`가
  "읽고 쓰는" 제약이다.
- A는 fp16 8개(= b32 레지스터 4개), B는 fp16 4개(b32 2개), C/D는 fp32 4개.

### 스레드 좌표계

```cuda
const int g = lane / 4;     // 16x8 타일 안에서 내 행
const int tig = lane % 4;   // 내 열 페어 (2*tig, 2*tig+1)
const int r_lo = q_base + warp * 16 + g;  // c0,c1이 속한 전역 Q행
const int r_hi = r_lo + 8;                // c2,c3이 속한 전역 Q행
```

이후 모든 코드가 "index 0 = r_lo행, index 1 = r_hi행" 관례를 따른다
(m[2], l[2], alpha[2], o[jo][0..1]/[2..3]).

### Q fragment를 HBM에서 직접 로드

Q smem 타일 자체가 없다. 각 스레드가 자기 fragment 원소(연속 half 2개 = 4B)를
글로벌에서 직접 읽는다. 커널당 한 번이라 비용은 무시 가능, 범위 밖 행은 0.

### QK^T와 softmax

- K의 B fragment: 원소 (k=2tig, n=g)가 row-major K에서 연속 half 2개라
  4B 로드 두 번이면 fragment 완성.
- softmax: 스레드 내 max/sum → `__shfl_xor_sync(…, 1)`, `(…, 2)`로 quad 리덕션
  → alpha 계산 → `o[jo][*] *= alpha[*]` (레지스터 rescale).
- scale은 `exp(scale * (s - m))` 형태로 exp 안에서 곱한다
  (max는 raw 점수로 추적해도 동치).

### P 재포장과 PV

```cuda
pa[0] = pack_float2(s[2kk][0], s[2kk][1]);   // fp32 2개 → half2 → b32
...
```

인접한 16×8 S 블록 두 개가 16×16 A 오퍼랜드 하나로 합쳐진다.
V의 B fragment는 row-major V의 "열"을 걷는 패턴이라 스칼라 half 4개를 읽어
조립한다 — **의도적으로 남긴 비효율**이며, `ldmatrix.trans`(차기 단계 후보)의
타깃이라고 문서에 명시할 것.

### 에필로그

`1/l`을 곱한 뒤 half2로 변환해 HBM에 직접 store. O가 smem을 거치는 일은
처음부터 끝까지 없다.

### 런처

```cuda
if (d == 64)  launch_impl<64>(...);
else if (d == 128) launch_impl<128>(...);
```

smem은 K/V 더블 버퍼만이라 40KB(d=64)/72KB(d=128).

## 3. 벤치마크 참고

- **필수 3종**:
  1. 메인 표 (09→10, 7.2x)
  2. **BC=32 시절의 동률 기록** (09: 23.685ms vs 10: 23.672ms, 1.00x) —
     "공통 게이트" 논증의 증거물. 이 표가 없으면 6.7x가 마법처럼 보인다
  3. 일반화 검증: N 스윕(70~79%), d=128(llm 프리셋), 경계 shape(N=1000) PASS
- **선택**:
  - `-Xptxas -v`로 레지스터 수/스필 0 확인 (템플릿 특수화 논증 뒷받침)
  - ncu HMMA 파이프 활용률의 08 대비 상승 폭
